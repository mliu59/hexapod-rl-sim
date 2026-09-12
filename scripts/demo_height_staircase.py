"""Drive the trained height-tracking policy with a scripted setpoint staircase.

The deployment-side counterpart of the UniformHeightCommand training sampler:
one env, and an external function (here a staircase) writes the command tensor
directly every step, overriding the random resampler. Records the rollout to
mp4 and prints per-step settling statistics.

Run:
    python -u scripts/demo_height_staircase.py --headless \
        --checkpoint <path to model_*.pt>
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Height-setpoint staircase demo for the spidertron.")
parser.add_argument("--checkpoint", type=str, required=True, help="Full path to the model checkpoint.")
parser.add_argument("--hold_s", type=float, default=2.5, help="Seconds to hold each setpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Deterministic rendering for the capture path (same block as check_collapse.py):
# without waitIdle/syncLoads the annotator readback races frame completion and
# alternate frames come back as uniform clear-colour buffers.
import carb

_settings = carb.settings.get_settings()
_settings.set("/app/asyncRendering", False)
_settings.set("/app/asyncRenderingLowLatency", False)
_settings.set("/app/hydraEngine/waitIdle", True)
_settings.set("/rtx/materialDb/syncLoads", True)
_settings.set("/rtx/hydra/materialSyncLoads", True)
_settings.set("/omni.kit.plugin/syncUsdLoads", True)

import importlib.metadata as metadata
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

TASK = "Spidertron-HeightTrack-Play-v0"
# staircase: start at nominal, sweep the trained 0.18-0.35 m range with steps
# of varying size and direction, return to nominal
HEIGHTS = [0.2745, 0.20, 0.30, 0.35, 0.25, 0.18, 0.2745]
SETTLE_TOL = 0.01  # m


def main() -> None:
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env_cfg.scene.num_envs = 1
    # longer than the demo so no timeout reset interrupts the recording
    env_cfg.episode_length_s = len(HEIGHTS) * args_cli.hold_s + 10.0
    env_cfg.viewer.eye = (1.3, -1.3, 0.55)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.25)

    video_dir = os.path.join(os.path.dirname(args_cli.checkpoint), "videos", "demo")
    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
    total_steps = int(len(HEIGHTS) * args_cli.hold_s / env.unwrapped.step_dt)
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

    command_term = env.unwrapped.command_manager.get_term("base_height")
    robot = env.unwrapped.scene["robot"]
    update_frame_marker = base_frame_marker(env.unwrapped)
    origin_z = env.unwrapped.scene.env_origins[0, 2]
    dt = env.unwrapped.step_dt
    steps_per_hold = int(args_cli.hold_s / dt)

    obs = env.get_observations()
    print(f"[demo   ] {len(HEIGHTS)} setpoints, {args_cli.hold_s} s each, tolerance {SETTLE_TOL * 100:.0f} cm")
    for target in HEIGHTS:
        settle_step = None
        errors = []
        for i in range(steps_per_hold):
            command_term.height_command[:, 0] = target
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            update_frame_marker()
            error = abs(float(robot.data.root_pos_w[0, 2] - origin_z) - target)
            errors.append(error)
            if settle_step is None and error < SETTLE_TOL:
                settle_step = i
        tail = errors[-int(0.5 / dt) :]
        settle = f"{settle_step * dt:5.2f} s" if settle_step is not None else "  neverr"
        print(
            f"[demo   ] target {target:.4f} m: settled in {settle}, "
            f"steady-state |err| {sum(tail) / len(tail) * 1000:.1f} mm"
        )

    env.close()
    videos = sorted(Path(video_dir).glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    print(f"[demo   ] video: {videos[-1] if videos else 'MISSING -- RecordVideo wrote nothing'}")


if __name__ == "__main__":
    main()
    simulation_app.close()
