"""Record a single-env walk rollout with a command HUD.

Runs the walk policy on one env with the task's own command sampling, a chase
camera, and the base-frame axis marker; logs the command/state per step and
post-processes the video with a HUD: live command numbers (speed, yaw rate,
height) vs actuals, mode flag, and a compass arrow showing the target direction
in the robot's base frame (up = robot forward).

Run:
    python -u scripts/demo_walk_rollout.py --headless --checkpoint <model.pt>
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Walk rollout demo with command HUD.")
parser.add_argument("--checkpoint", type=str, required=True, help="Full path to the model checkpoint.")
parser.add_argument("--duration_s", type=float, default=18.0, help="Rollout length in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import math
import os
from pathlib import Path

import gymnasium as gym
import hexapod_lab.tasks  # noqa: F401
import torch
from hexapod_lab.viz import base_frame_marker
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK = "Spidertron-Walk-Play-v0"
UNIFORM_STD = 2.0  # frames below this pixel std are the known blank-buffer race
CAM_OFFSET = (1.6, -1.6, 0.7)
ARROW_LEN = 0.6  # m, world-space length of the target-direction arrow
CAM_HFOV = math.radians(60.0)  # default Kit viewport camera (focal 18.15 mm / aperture 20.955 mm)


def _camera_projector(width: int, height: int):
    """Pinhole projection into the chase camera, whose pose relative to the robot
    is constant (eye = robot + CAM_OFFSET, looking at the robot). Takes vectors
    relative to the EYE, returns pixel coordinates."""
    off = torch.tensor(CAM_OFFSET, dtype=torch.float64)
    fwd = -off / off.norm()
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    right = torch.linalg.cross(fwd, world_up)
    right = right / right.norm()
    cam_up = torch.linalg.cross(right, fwd)
    tan_h = math.tan(CAM_HFOV / 2)
    tan_v = tan_h * height / width

    def project(v_xyz) -> tuple[float, float]:
        v = torch.tensor(v_xyz, dtype=torch.float64)
        x, y, z = float(v @ right), float(v @ cam_up), float(v @ fwd)
        return (x / z / tan_h * (width / 2) + width / 2, -y / z / tan_v * (height / 2) + height / 2)

    return project


def overlay_hud(frames_with_data, out_path: str) -> None:
    """Draw command HUD (text panel + base-frame target compass) onto frames."""
    import imageio.v3 as iio
    import numpy as np
    from PIL import Image, ImageDraw

    out_frames = []
    project = None
    for frame, d in frames_with_data:
        img = Image.fromarray(frame)
        if project is None:
            project = _camera_projector(img.width, img.height)
        draw = ImageDraw.Draw(img)
        mode = "IDLE" if not d["active"] else ("TURN" if d["v_cmd"] < 0.02 else "WALK")
        lines = [
            f"mode {mode}",
            f"v_fwd  cmd {d['v_cmd']:+.2f}  act {d['v_act']:+.2f} m/s",
            f"yaw_w  cmd {d['w_cmd']:+.2f}  act {d['w_act']:+.2f} rad/s",
            f"height cmd  {d['h_cmd']:.3f}  act  {d['h_act']:.3f} m",
            f"heading err {math.degrees(d['err']):+.0f} deg",
        ]
        x0, y0 = 10, 8
        draw.rectangle([x0 - 4, y0 - 4, x0 + 300, y0 + 16 * len(lines) + 4], fill=(0, 0, 0))
        for i, line in enumerate(lines):
            draw.text((x0, y0 + 16 * i), line, fill=(255, 255, 255))
        # solid world-space arrow from the robot's center along the target
        # direction. Both endpoints are expressed relative to the camera eye:
        # the robot center is always at -CAM_OFFSET (the camera tracks it).
        theta = d["theta_t"]
        tail = [-CAM_OFFSET[0], -CAM_OFFSET[1], -CAM_OFFSET[2]]
        tip = [
            tail[0] + ARROW_LEN * math.cos(theta),
            tail[1] + ARROW_LEN * math.sin(theta),
            tail[2],
        ]
        x0, y0 = project(tail)
        x1, y1 = project(tip)
        draw.line([x0, y0, x1, y1], fill=(255, 60, 60), width=6)
        # arrowhead: two barbs swept back from the tip in screen space
        ang = math.atan2(y1 - y0, x1 - x0)
        for barb in (ang + 2.6, ang - 2.6):
            draw.line([x1, y1, x1 + 18 * math.cos(barb), y1 + 18 * math.sin(barb)], fill=(255, 60, 60), width=6)
        out_frames.append(np.asarray(img))
    iio.imwrite(out_path, out_frames, fps=50, codec="libx264")


def main() -> None:
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = args_cli.duration_s + 10.0  # no timeout mid-recording

    video_dir = os.path.join(os.path.dirname(args_cli.checkpoint), "videos", "demo")
    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
    total_steps = int(args_cli.duration_s / env.unwrapped.step_dt)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
        video_length=total_steps,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    motion = env.unwrapped.command_manager.get_term("base_motion")
    height_cmd = env.unwrapped.command_manager.get_term("base_height")
    robot = env.unwrapped.scene["robot"]
    origin = env.unwrapped.scene.env_origins[0]
    update_frame_marker = base_frame_marker(env.unwrapped)
    sim = env.unwrapped.sim

    obs = env.get_observations()
    step_data = []
    for _ in range(total_steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        update_frame_marker()
        pos = robot.data.root_pos_w[0]
        sim.set_camera_view(
            eye=[float(pos[0]) + CAM_OFFSET[0], float(pos[1]) + CAM_OFFSET[1], float(pos[2]) + CAM_OFFSET[2]],
            target=[float(pos[0]), float(pos[1]), float(pos[2])],
        )
        cmd = motion.command[0]
        step_data.append(
            {
                "v_cmd": float(cmd[0]),
                "w_cmd": float(cmd[1]),
                "cos_err": float(cmd[2]),
                "sin_err": float(cmd[3]),
                "err": math.atan2(float(cmd[3]), float(cmd[2])),
                "theta_t": float(motion.heading_target[0]),
                "active": bool(motion.is_active[0]),
                "h_cmd": float(height_cmd.command[0, 0]),
                "v_act": float(robot.data.root_lin_vel_b[0, 0]),
                "w_act": float(robot.data.root_ang_vel_b[0, 2]),
                "h_act": float(robot.data.root_pos_w[0, 2] - origin[2]),
            }
        )
    env.close()

    import imageio.v3 as iio

    raw = sorted(Path(video_dir).glob("rl-video-*.mp4"), key=lambda p: p.stat().st_mtime)[-1]
    frames = iio.imread(raw)
    # drop the known alternating blank frames, keeping data aligned by index
    keep = [(f, step_data[i]) for i, f in enumerate(frames) if i < len(step_data) and float(f.std()) > UNIFORM_STD]
    out_path = str(Path(video_dir) / "walk_rollout_hud.mp4")
    overlay_hud(keep, out_path)
    print(f"[demo   ] {len(keep)} frames with HUD -> {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
