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
parser.add_argument(
    "--task",
    type=str,
    default="Spidertron-Walk-Play-v0",
    help="Play task id (must register a 'base_motion' command; 'base_height' is optional).",
)
parser.add_argument(
    "--diagnostics",
    action="store_true",
    default=False,
    help="Also log per-joint target-vs-measured position and torque saturation, and write a "
    "gait-diagnostics plot. Separates 'policy intentionally holds this pose' (target tracks "
    "measured, moderate torque) from 'joint is stalled/jammed' (large tracking error with "
    "torque pinned at the effort limit).",
)
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


# Okabe-Ito colorblind-safe hues, fixed leg order (never cycled)
_LEG_ORDER = ["LF", "LM", "LR", "RF", "RM", "RR"]
_LEG_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def write_gait_diagnostics(joint_names, dt, positions, targets, torques, limits, out_path: str) -> None:
    """Plot tracking error per leg, then target-vs-measured and torque saturation
    for the worst-tracking leg. Printed table covers all joints."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pos, tgt, tau = np.asarray(positions), np.asarray(targets), np.asarray(torques)
    err = np.abs(tgt - pos)  # (T, 18)
    sat = np.abs(tau) / limits  # (T, 18)
    t = np.arange(pos.shape[0]) * dt
    leg_of = [name[:2] for name in joint_names]

    print(f"{'joint':>16} {'mean|err| rad':>14} {'p95|err|':>10} {'%steps sat>0.9':>15}")
    for j, name in enumerate(joint_names):
        print(f"{name:>16} {err[:, j].mean():14.4f} {np.percentile(err[:, j], 95):10.4f} "
              f"{(sat[:, j] > 0.9).mean() * 100:14.1f}%")

    leg_err = {leg: err[:, [j for j, lg in enumerate(leg_of) if lg == leg]].mean(axis=1) for leg in _LEG_ORDER}
    worst = max(_LEG_ORDER, key=lambda leg: leg_err[leg].mean())
    worst_ids = [j for j, lg in enumerate(leg_of) if lg == worst]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for leg, color in zip(_LEG_ORDER, _LEG_COLORS):
        axes[0].plot(t, leg_err[leg], color=color, lw=1.2, label=leg)
    axes[0].set_ylabel("mean |target - measured| (rad)")
    axes[0].set_title(f"Per-leg joint tracking error (worst: {worst})")
    axes[0].legend(ncol=6, fontsize=8, frameon=False)

    for j, color in zip(worst_ids, _LEG_COLORS[:3]):
        axes[1].plot(t, pos[:, j], color=color, lw=1.4, label=f"{joint_names[j]} measured")
        axes[1].plot(t, tgt[:, j], color=color, lw=1.0, ls="--", alpha=0.7, label=f"{joint_names[j]} target")
    axes[1].set_ylabel("joint position (rad)")
    axes[1].set_title(f"{worst} leg: commanded target (dashed) vs measured (solid) -- "
                      "overlap = intentional pose, divergence = joint not reaching its target")
    axes[1].legend(ncol=3, fontsize=7, frameon=False)

    for j, color in zip(worst_ids, _LEG_COLORS[:3]):
        axes[2].plot(t, sat[:, j], color=color, lw=1.2, label=joint_names[j])
    axes[2].axhline(1.0, color="#888888", lw=0.8, ls=":")
    axes[2].set_ylabel("|torque| / effort limit")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title(f"{worst} leg torque saturation (pinned at 1.0 + tracking error = stalling)")
    axes[2].legend(ncol=3, fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"[demo   ] gait diagnostics -> {out_path}")


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    # keep the play cfg's env count: the chase camera follows env 0, the rest
    # stay visible in the background
    env_cfg.episode_length_s = args_cli.duration_s + 10.0  # no timeout mid-recording

    video_dir = os.path.join(os.path.dirname(args_cli.checkpoint), "videos", "demo")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
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
    try:
        height_cmd = env.unwrapped.command_manager.get_term("base_height")
    except KeyError:
        height_cmd = None  # march-style task: fixed nominal height
    robot = env.unwrapped.scene["robot"]
    origin = env.unwrapped.scene.env_origins[0]
    update_frame_marker = base_frame_marker(env.unwrapped)
    sim = env.unwrapped.sim

    diag = None
    if args_cli.diagnostics:
        diag = {"pos": [], "tgt": [], "tau": [], "contact": [], "foot_xy": [], "base_vel_b": []}
        contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
        # sensor body ids for tibia feet, and articulation body ids (same names,
        # different index spaces)
        foot_names = [n for n in contact_sensor.body_names if n.endswith("_tibia")]
        diag["foot_names"] = foot_names
        diag["foot_ids"] = [contact_sensor.body_names.index(n) for n in foot_names]
        diag["foot_body_ids"] = [robot.body_names.index(n) for n in foot_names]

    obs = env.get_observations()
    step_data = []
    for _ in range(total_steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        update_frame_marker()
        if diag is not None:
            diag["pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
            diag["tgt"].append(robot.data.joint_pos_target[0].cpu().numpy().copy())
            diag["tau"].append(robot.data.applied_torque[0].cpu().numpy().copy())
            contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
            diag["contact"].append(
                (contact_sensor.data.current_contact_time[0, diag["foot_ids"]] > 0.0).cpu().numpy().copy()
            )
            diag["foot_xy"].append(robot.data.body_pos_w[0, diag["foot_body_ids"], :2].cpu().numpy().copy())
            diag["base_vel_b"].append(robot.data.root_lin_vel_b[0, :2].cpu().numpy().copy())
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
                "h_cmd": float(height_cmd.command[0, 0]) if height_cmd is not None else 0.2745,
                "v_act": float(robot.data.root_lin_vel_b[0, 0]),
                "w_act": float(robot.data.root_ang_vel_b[0, 2]),
                "h_act": float(robot.data.root_pos_w[0, 2] - origin[2]),
            }
        )
    if diag is not None:
        import numpy as np

        # effort limits by joint name (robots/spidertron.py: STS3215 coxa, STS3250 pitch)
        limits = np.array([2.06 if "_coxa_" in n else 3.43 for n in robot.joint_names])
        write_gait_diagnostics(
            robot.joint_names,
            env.unwrapped.step_dt,
            diag["pos"],
            diag["tgt"],
            diag["tau"],
            limits,
            os.path.join(video_dir, "walk_gait_diagnostics.png"),
        )

        # ---- translation analysis: why does the body (not) advance? --------
        dt = env.unwrapped.step_dt
        contact = np.asarray(diag["contact"])  # (T, 6) bool
        foot_xy = np.asarray(diag["foot_xy"])  # (T, 6, 2) world
        vel_b = np.asarray(diag["base_vel_b"])  # (T, 2)
        T = contact.shape[0]
        print(f"[transl ] mean base vel: fwd {vel_b[:, 0].mean():+.3f} m/s, lat {vel_b[:, 1].mean():+.3f} m/s")
        print(f"{'foot':>10} {'duty%':>6} {'steps/s':>8} {'step_len_cm':>12} {'stance_drift_cm':>16}")
        for f, name in enumerate(diag["foot_names"]):
            c = contact[:, f]
            duty = c.mean() * 100
            touchdowns = np.flatnonzero(~c[:-1] & c[1:]) + 1
            liftoffs = np.flatnonzero(c[:-1] & ~c[1:]) + 1
            rate = len(touchdowns) / (T * dt)
            # step length: foot displacement over each swing (liftoff -> next touchdown)
            step_lens = []
            for lo in liftoffs:
                nxt = touchdowns[touchdowns > lo]
                if len(nxt):
                    step_lens.append(np.linalg.norm(foot_xy[nxt[0], f] - foot_xy[lo, f]))
            # stance drift: how far the planted foot slides over each stance period
            drifts = []
            for td in touchdowns:
                nxt = liftoffs[liftoffs > td]
                if len(nxt):
                    drifts.append(np.linalg.norm(foot_xy[nxt[0], f] - foot_xy[td, f]))
            sl = np.mean(step_lens) * 100 if step_lens else 0.0
            dr = np.mean(drifts) * 100 if drifts else 0.0
            print(f"{name:>10} {duty:6.1f} {rate:8.2f} {sl:12.2f} {dr:16.2f}")

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
