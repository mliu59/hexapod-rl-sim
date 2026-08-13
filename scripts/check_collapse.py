"""Zero-torque collapse test for the placeholder hexapod (milestone M1).

`check_robot.py` shows the hexapod stands under a PD hold. That alone does not
prove the actuators are what hold it up — a fixed base, locked joints, or a
stray implicit drive would look identical. This test removes the actuation and
checks the robot actually falls:

- phase A: PD hold at the default pose (control) -> stays at standing height
- phase B: stiffness = damping = 0, zero effort target -> collapses to the deck

Passing means gravity, joint freedom and actuator authority are all wired up.

Usage:
    python scripts/check_collapse.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-torque collapse test for the hexapod.")
parser.add_argument("--video", action="store_true", default=False, help="Record the test to an mp4.")
parser.add_argument(
    "--video_path", type=str, default="docs/m1_collapse.mp4", help="Where to write the recording."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# a camera sensor is only created when cameras are enabled
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext

from hexapod_lab.robots import HEXAPOD_CFG

# body half-height is 0.03 m, so a fully collapsed base sits near there
COLLAPSED_MAX_HEIGHT = 0.06  # m
STANDING_MIN_HEIGHT = 0.09  # m
SETTLE_STEPS = 400  # 2.0 s at dt = 0.005

# record every Nth physics step -> 200 Hz / 4 = 50 fps of real-time footage
VIDEO_EVERY = 4
VIDEO_FPS = 50
RENDER_PASSES = 3  # RTX passes per captured frame; fewer gives uniform buffers


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/light", light_cfg)

    robot = Articulation(HEXAPOD_CFG.replace(prim_path="/World/Robot"))

    camera = None
    if args_cli.video:
        camera = Camera(
            CameraCfg(
                prim_path="/World/Camera",
                update_period=0.0,
                height=720,
                width=1280,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)),
                offset=CameraCfg.OffsetCfg(pos=(0.5, -0.5, 0.25), rot=(0.0, 0.0, 0.0, 1.0)),
            )
        )

    sim.reset()

    frames: list = []
    if camera is not None:
        camera.set_world_poses_from_view(
            eyes=torch.tensor([[0.55, -0.55, 0.22]], device=sim.device),
            targets=torch.tensor([[0.0, 0.0, 0.06]], device=sim.device),
        )
        # the RTX pipeline needs a few frames before the annotator returns
        # anything but black -- without this the clip opens on dead frames
        for _ in range(20):
            sim.render()
        camera.update(sim.get_physics_dt(), force_recompute=True)

    def grab(step: int) -> None:
        """Render and keep a frame every VIDEO_EVERY-th physics step.

        The RTX pipeline needs several passes before the annotator holds a
        complete image -- read it too early and you get a uniform buffer
        (black, or a flat clear-colour 243). So the physics loop runs with
        rendering off and every sampled step is rendered RENDER_PASSES times
        before the frame is read.
        """
        if camera is None or step % VIDEO_EVERY != 0:
            return
        for _ in range(RENDER_PASSES):
            sim.render()
        camera.update(sim.get_physics_dt(), force_recompute=True)
        frames.append(camera.data.output["rgb"][0, ..., :3].cpu().numpy().copy())

    # ---- phase A: PD hold (control) -----------------------------------------
    for i in range(SETTLE_STEPS):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
        grab(i)

    held_h = robot.data.root_pos_w[0, 2].item()
    held_torque = robot.data.applied_torque[0].abs().max().item()
    print(f"[phase A] PD hold      : root height {held_h:.4f} m, max |torque| {held_torque:.3f} N*m")

    # ---- phase B: kill the actuators ----------------------------------------
    zeros = torch.zeros_like(robot.data.joint_stiffness)
    robot.write_joint_stiffness_to_sim(zeros)
    robot.write_joint_damping_to_sim(zeros)
    # write_joint_*_to_sim deliberately does not touch the actuator models
    # (IsaacLab #128), so zero those too. Otherwise PhysX applies nothing while
    # `applied_torque` still reports the model's PD result clipped to 3 N*m.
    for actuator in robot.actuators.values():
        actuator.stiffness[:] = 0.0
        actuator.damping[:] = 0.0
    print("[phase B] stiffness and damping zeroed (sim + actuator models); zero effort target")

    for i in range(SETTLE_STEPS):
        robot.set_joint_effort_target(torch.zeros_like(robot.data.joint_pos))
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
        grab(SETTLE_STEPS + i)
        if i % 100 == 0:
            print(f"[phase B] step {i:3d}: root height {robot.data.root_pos_w[0, 2].item():.4f} m")

    fell_h = robot.data.root_pos_w[0, 2].item()
    fell_torque = robot.data.applied_torque[0].abs().max().item()
    joint_drift = (robot.data.joint_pos[0] - robot.data.default_joint_pos[0]).abs().max().item()
    print(f"[phase B] zero torque  : root height {fell_h:.4f} m, max |torque| {fell_torque:.3f} N*m")
    print(f"[phase B] max joint drift from default pose: {joint_drift:.3f} rad")
    print(f"[result ] height drop: {held_h:.4f} -> {fell_h:.4f} m ({held_h - fell_h:.4f} m)")

    # write the recording before the assertions, so a failing run still leaves
    # footage to look at
    if camera is not None:
        import imageio.v3 as iio

        out_path = Path(args_cli.video_path)
        if not out_path.is_absolute():
            out_path = Path(__file__).resolve().parents[1] / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(out_path, frames, fps=VIDEO_FPS, codec="libx264")
        print(f"[video  ] wrote {len(frames)} frames ({len(frames) / VIDEO_FPS:.1f} s) to {out_path}")

    assert held_h > STANDING_MIN_HEIGHT, f"phase A: robot did not stand ({held_h:.4f} m)"
    assert fell_torque < 1e-3, f"phase B: actuators still applying torque ({fell_torque:.4f} N*m)"
    assert fell_h < COLLAPSED_MAX_HEIGHT, (
        f"phase B: robot did not collapse ({fell_h:.4f} m) -- something other than the actuators is holding it up"
    )
    assert joint_drift > 0.1, f"phase B: joints barely moved ({joint_drift:.3f} rad) -- are they locked?"
    print("[check  ] OK: hexapod is held up by actuator torque and collapses without it")


if __name__ == "__main__":
    main()
    simulation_app.close()
