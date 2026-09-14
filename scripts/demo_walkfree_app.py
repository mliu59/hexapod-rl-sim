"""Interactive browser demo for the WalkFree policy.

Runs the trained Spidertron-WalkFree policy on one env on open flat ground,
renders a fixed wide view (or a chase view) offscreen, and serves a local web UI at
http://127.0.0.1:<port>: main pane is the live sim render, sidebar has a
joystick (world-frame direction + speed, hold-last-command) and a base-height
slider. The sidebar values are written into the task's own command terms every
step, so the policy sees exactly the command interface it was trained on
(DirectionSpeedCommand heading/speed + UniformHeightCommand height).

Architecture: the sim loop stays on the main thread (Kit requirement), paced
to wall clock; an aiohttp server on a daemon thread streams the latest JPEG
frame + telemetry over a websocket and receives command updates. GUI-mode Kit
is not an option here (h5py DLL bug, see CLAUDE.md), and a browser UI needs no
display attached to the sim process.

Run (PowerShell, from repo root):
    cmd /c "set OMNI_KIT_ACCEPT_EULA=YES&& call C:\\git_ws\\env_isaaclab\\Scripts\\activate.bat && python -u scripts\\demo_walkfree_app.py --headless"
"""

import argparse
import sys

from isaaclab.app import AppLauncher

DEFAULT_CHECKPOINT = r"logs\rsl_rl\spidertron_walk_free\2026-09-14_00-58-29\model_1999.pt"

parser = argparse.ArgumentParser(description="Interactive WalkFree demo app.")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Full path to the model checkpoint.")
parser.add_argument("--port", type=int, default=8765, help="HTTP port for the web UI (localhost only).")
parser.add_argument("--render_fps", type=float, default=12.0, help="Wall-clock frame rate of the streamed render.")
parser.add_argument("--render_width", type=int, default=1152, help="Render width in pixels.")
parser.add_argument("--render_height", type=int, default=648, help="Render height in pixels.")
parser.add_argument("--jpeg_quality", type=int, default=82, help="JPEG quality for streamed frames.")
parser.add_argument(
    "--sync_render",
    action="store_true",
    default=False,
    help="Force the deterministic sync-render settings (check_collapse.py). Costs ~4x sim throughput; without "
    "them the frame.std() watchdog silently drops the occasional blank frame, which a live stream tolerates.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# default to CPU physics: the GPU pipeline is launch-bound (~20 ms/step wall
# for 1 env and 4096 alike -> ~30 Hz); one robot on CPU steps in ~7 ms and the
# demo runs real-time (measured 92-98/100 Hz). Pass --device cuda to override.
if "--device" not in sys.argv:
    args_cli.device = "cpu"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Deterministic offscreen rendering: without these, sim.render() races frame
# completion and the annotator can return blank buffers (see check_collapse.py).
# Optional here: measured ~4x sim-rate cost, and the live stream just drops the
# occasional blank frame via the frame.std() watchdog instead.
import carb

_settings = carb.settings.get_settings()
if args_cli.sync_render:
    _settings.set("/app/asyncRendering", False)
    _settings.set("/app/asyncRenderingLowLatency", False)
    _settings.set("/app/hydraEngine/waitIdle", True)
    _settings.set("/rtx/materialDb/syncLoads", True)
    _settings.set("/rtx/hydra/materialSyncLoads", True)
    _settings.set("/omni.kit.plugin/syncUsdLoads", True)
else:
    # the headless-camera experience blocks ~90 ms per render call with its
    # defaults; forcing the async pipeline shaves it (measured 2026-09-14 at
    # 540p/10 fps: 19 -> 30 Hz sim rate; remaining cost profiled per 200 frames)
    _settings.set("/app/asyncRendering", True)
    _settings.set("/app/hydraEngine/waitIdle", False)

import asyncio
import importlib.metadata as metadata
import io
import json
import math
import threading
import time
from pathlib import Path

import gymnasium as gym
import hexapod_lab.tasks  # noqa: F401
import torch
from aiohttp import WSMsgType, web
from PIL import Image
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from hexapod_lab.tasks.manager_based.hexapod_lab.spidertron_base_env_cfg import NOMINAL_HEIGHT
from hexapod_lab.tasks.manager_based.hexapod_lab.spidertron_height_env_cfg import HEIGHT_RANGE
from hexapod_lab.tasks.manager_based.hexapod_lab.spidertron_walk_free_env_cfg import MAX_SPEED

TASK = "Spidertron-WalkFree-Play-v0"
UNIFORM_STD = 2.0  # frames below this pixel std are the known blank-buffer race
# Fixed camera: elevated three-quarter view from -Y so screen-up = world +Y
# and screen-right = world +X (joystick axes match).
CAM_EYE = (0.0, -6.8, 4.8)
CAM_TARGET = (0.0, 0.3, 0.0)
# Chase camera: constant WORLD-frame offset from the robot (it does not rotate
# with the heading), so the joystick's world-axis mapping stays valid on screen.
CHASE_OFFSET = (1.8, -1.8, 0.9)


class SharedState:
    """Cross-thread state. Plain attribute reads/writes of floats/bytes are
    GIL-atomic; the sim loop reads commands, the server reads frames/telemetry."""

    def __init__(self) -> None:
        self.cmd_heading = 0.0  # rad, world frame
        self.cmd_speed = 0.0  # m/s
        self.cmd_height = NOMINAL_HEIGHT  # m
        self.reset_requested = False
        # live camera override (dev tuning hook, see the "camera" ws message)
        self.cam_eye = CAM_EYE
        self.cam_target = CAM_TARGET
        self.cam_dirty = False
        self.chase = False
        self.jpeg: bytes | None = None
        self.seq = 0
        self.telemetry: dict = {}


SHARED = SharedState()


# ---------------------------------------------------------------- web server


def _apply_command(data: dict) -> None:
    if "heading" in data:
        SHARED.cmd_heading = float(data["heading"])
    if "speed" in data:
        SHARED.cmd_speed = min(max(float(data["speed"]), 0.0), MAX_SPEED)
    if "height" in data:
        SHARED.cmd_height = min(max(float(data["height"]), HEIGHT_RANGE[0]), HEIGHT_RANGE[1])
    if data.get("reset"):
        SHARED.reset_requested = True
    if "chase" in data:
        SHARED.chase = bool(data["chase"])
        SHARED.cam_dirty = not SHARED.chase  # restore the fixed view on toggle-off
    if "camera" in data:  # dev tuning: {"camera": {"eye": [x,y,z], "target": [x,y,z]}}
        cam = data["camera"]
        SHARED.cam_eye = tuple(float(v) for v in cam.get("eye", SHARED.cam_eye))
        SHARED.cam_target = tuple(float(v) for v in cam.get("target", SHARED.cam_target))
        SHARED.cam_dirty = True


async def _index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(Path(__file__).with_suffix(".html"))


async def _push_loop(ws: web.WebSocketResponse) -> None:
    last_seq, last_tel = -1, 0.0
    while not ws.closed:
        await asyncio.sleep(0.025)
        if SHARED.seq != last_seq and SHARED.jpeg is not None:
            last_seq = SHARED.seq
            await ws.send_bytes(SHARED.jpeg)
        now = time.monotonic()
        if now - last_tel > 0.1 and SHARED.telemetry:
            last_tel = now
            await ws.send_str(json.dumps(SHARED.telemetry))


async def _ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    pusher = asyncio.ensure_future(_push_loop(ws))
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                _apply_command(json.loads(msg.data))
    finally:
        pusher.cancel()
    return ws


def start_server(port: int) -> None:
    def run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = web.Application()
        app.router.add_get("/", _index)
        app.router.add_get("/ws", _ws_handler)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
        print(f"[app    ] UI ready at http://127.0.0.1:{port}")
        loop.run_forever()

    threading.Thread(target=run, name="webui", daemon=True).start()


# ------------------------------------------------------------------ sim side


def make_env():
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 3600.0  # no timeouts mid-demo
    # honor --device: GPU-pipeline physics is launch-bound (~20 ms/step wall
    # for 1 env and 4096 alike); a single robot runs far faster on CPU
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    # the UI owns the commands: never let the task resample them mid-run
    env_cfg.commands.base_motion.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_motion.rel_standing_envs = 0.0
    env_cfg.commands.base_height.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.viewer.eye = CAM_EYE
    env_cfg.viewer.lookat = CAM_TARGET
    env_cfg.viewer.resolution = (args_cli.render_width, args_cli.render_height)

    # 10 m arena marker: visual-only disks (no collision APIs), robot spawns at
    # the center. Registered on the scene cfg exactly like a light asset.
    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    return env, agent_cfg


def main() -> None:
    env, agent_cfg = make_env()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[app    ] loaded {args_cli.checkpoint}")

    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    motion = raw_env.command_manager.get_term("base_motion")
    height_cmd = raw_env.command_manager.get_term("base_height")
    origin = raw_env.scene.env_origins[0]
    raw_env.sim.set_camera_view(eye=list(CAM_EYE), target=list(CAM_TARGET))
    step_dt = raw_env.step_dt

    start_server(args_cli.port)

    obs = env.get_observations()
    falls = 0
    last_ep_len = 0
    render_period = 1.0 / args_cli.render_fps
    last_render = 0.0
    render_count = 0
    fps_window_t, fps_window_n, measured_fps = time.perf_counter(), 0, 0.0
    frames_window_n, measured_stream_fps = 0, 0.0
    next_tick = time.perf_counter()
    step_ms_accum, step_ms_n = 0.0, 0
    last_telemetry = 0.0
    while simulation_app.is_running():
        t0 = time.perf_counter()

        if SHARED.cam_dirty:
            SHARED.cam_dirty = False
            raw_env.sim.set_camera_view(eye=list(SHARED.cam_eye), target=list(SHARED.cam_target))

        if SHARED.reset_requested:
            SHARED.reset_requested = False
            SHARED.cmd_speed = 0.0
            # inference_mode to match step(): the sim buffers are inference
            # tensors, and reset() writes them in-place (crashes otherwise)
            with torch.inference_mode():
                raw_env.reset()
                obs = env.get_observations()
            last_ep_len = 0  # a manual reset is not a fall

        # inject UI commands into the task's own command terms; the term's
        # _update_command inside step() derives yaw-rate and heading error
        motion.heading_target[0] = SHARED.cmd_heading
        motion.speed[0] = SHARED.cmd_speed
        height_cmd.height_command[0, 0] = SHARED.cmd_height

        t_s0 = time.perf_counter()
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        step_ms_accum += time.perf_counter() - t_s0
        step_ms_n += 1

        if t0 - last_render >= render_period:
            last_render = t0
            if SHARED.chase:
                p = robot.data.root_pos_w[0]
                raw_env.sim.set_camera_view(
                    eye=[float(p[0]) + CHASE_OFFSET[0], float(p[1]) + CHASE_OFFSET[1], float(p[2]) + CHASE_OFFSET[2]],
                    target=[float(p[0]), float(p[1]), float(p[2])],
                )
            t_r0 = time.perf_counter()
            raw_env.sim.render()
            t_r1 = time.perf_counter()
            frame = raw_env.render(recompute=True)  # annotator read only; sim.render done above
            t_r2 = time.perf_counter()
            render_count += 1
            if render_count % 200 == 1 and step_ms_n:
                print(
                    f"[perf   ] sim.render {1e3 * (t_r1 - t_r0):.1f} ms,"
                    f" annotator {1e3 * (t_r2 - t_r1):.1f} ms,"
                    f" env.step avg {1e3 * step_ms_accum / step_ms_n:.1f} ms"
                )
                step_ms_accum, step_ms_n = 0.0, 0
            # blank-frame watchdog: without sync rendering the annotator can
            # race frame completion; a dropped frame is invisible in a stream
            if frame is not None and float(frame.std()) > UNIFORM_STD:
                buf = io.BytesIO()
                Image.fromarray(frame).save(buf, format="JPEG", quality=args_cli.jpeg_quality)
                SHARED.jpeg = buf.getvalue()
                SHARED.seq += 1
                frames_window_n += 1

        fps_window_n += 1
        if t0 - fps_window_t >= 1.0:
            measured_fps = fps_window_n / (t0 - fps_window_t)
            measured_stream_fps = frames_window_n / (t0 - fps_window_t)
            fps_window_t, fps_window_n, frames_window_n = t0, 0, 0

        # telemetry at ~10 Hz, not per step: each float() here is a GPU->CPU
        # sync, and a dozen of them per 10 ms step is real overhead
        if t0 - last_telemetry >= 0.1:
            last_telemetry = t0
            ep_len = int(raw_env.episode_length_buf[0])
            if ep_len < last_ep_len:  # env auto-reset: the robot fell
                falls += 1
            last_ep_len = ep_len

            cmd = motion.command[0]
            pos = robot.data.root_pos_w[0]
            SHARED.telemetry = {
                "v_cmd": round(float(cmd[0]), 3),
                "v_act": round(float(robot.data.root_lin_vel_b[0, 0]), 3),
                "w_cmd": round(float(cmd[1]), 3),
                "w_act": round(float(robot.data.root_ang_vel_b[0, 2]), 3),
                "h_cmd": round(SHARED.cmd_height, 4),
                "h_act": round(float(pos[2] - origin[2]), 4),
                "heading_act": round(float(robot.data.heading_w[0]), 3),
                "heading_cmd": round(SHARED.cmd_heading, 3),
                "heading_err": round(math.atan2(float(cmd[3]), float(cmd[2])), 3),
                "speed_cmd": round(SHARED.cmd_speed, 3),
                "x": round(float(pos[0] - origin[0]), 2),
                "y": round(float(pos[1] - origin[1]), 2),
                "falls": falls,
                "sim_hz": round(measured_fps, 1),
                "rt_target_hz": round(1.0 / step_dt, 1),
                "stream_fps": round(measured_stream_fps, 1),
            }

        # real-time pacing against an absolute schedule: Windows sleep can
        # overshoot by a timer quantum (~15 ms), which capped a naive
        # sleep-per-step loop at ~30 Hz. With an absolute next_tick, an
        # oversleep just makes the following iterations run back-to-back
        # until the schedule is caught up.
        next_tick += step_dt
        lag = time.perf_counter() - next_tick
        if lag < 0:
            time.sleep(-lag)
        elif lag > 0.25:
            next_tick = time.perf_counter()  # resync after a long stall

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
