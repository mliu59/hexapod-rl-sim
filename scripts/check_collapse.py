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
parser.add_argument(
    "--robot",
    choices=("hexapod", "spidertron"),
    default="hexapod",
    help="Which articulation config to test (thresholds and camera scale with it).",
)
parser.add_argument("--video", action="store_true", default=False, help="Record the test to an mp4.")
parser.add_argument(
    "--video_path", type=str, default="", help="Where to write the recording (default: docs/m1_collapse_<robot>.mp4)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# a camera sensor is only created when cameras are enabled
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if args_cli.video:
    # Make rendering DETERMINISTIC instead of racing the async pipeline.
    # Two async sources produce uniform clear-colour (243) frames:
    #   1. sim.render() enqueues GPU work and returns; the annotator readback
    #      races frame completion -> waitIdle blocks until the frame is done.
    #   2. Materials/shaders load asynchronously; frames render as bare clear
    #      colour until compiled (why cold runs were far worse than warm) ->
    #      syncLoads forces them to finish before the frame is drawn.
    # With these set, one render call = one complete frame, and the verify
    # loop in grab() is a watchdog that should count zero retries.
    import carb

    _settings = carb.settings.get_settings()
    _settings.set("/app/asyncRendering", False)
    _settings.set("/app/asyncRenderingLowLatency", False)
    _settings.set("/app/hydraEngine/waitIdle", True)
    _settings.set("/rtx/materialDb/syncLoads", True)
    _settings.set("/rtx/hydra/materialSyncLoads", True)
    _settings.set("/omni.kit.plugin/syncUsdLoads", True)

import time
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext

from hexapod_lab.robots import HEXAPOD_CFG, SPIDERTRON_CFG

# Per-robot scaling: (cfg, collapsed max, standing min, camera eye, camera target).
# hexapod: body half-height 0.03 m, stands at ~0.11 m.
# spidertron: base origin at the hip plane (0.27 m standing); collapsed, the
# pod underside (0.04 m below origin) rests near the deck -> origin ~0.05-0.10.
_ROBOTS = {
    "hexapod": (HEXAPOD_CFG, 0.06, 0.09, (0.55, -0.55, 0.22), (0.0, 0.0, 0.06)),
    "spidertron": (SPIDERTRON_CFG, 0.14, 0.24, (1.35, -1.35, 0.55), (0.0, 0.0, 0.18)),
}
ROBOT_CFG, COLLAPSED_MAX_HEIGHT, STANDING_MIN_HEIGHT, CAM_EYE, CAM_TARGET = _ROBOTS[args_cli.robot]
SETTLE_STEPS = 400  # 2.0 s at dt = 0.005

# record every Nth physics step -> 200 Hz / 4 = 50 fps of real-time footage
VIDEO_EVERY = 4
VIDEO_FPS = 50
# With async rendering disabled and waitIdle/syncLoads on (see the settings
# block after AppLauncher), one render call yields one complete frame. The
# verify loop in grab() remains as a WATCHDOG: retries should be zero, and a
# nonzero count in the summary means the determinism settings regressed.
RENDER_PASSES = 1
MAX_EXTRA_PASSES = 64
UNIFORM_STD = 2.0  # frame std below this = no scene content in the buffer


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/light", light_cfg)

    robot = Articulation(ROBOT_CFG.replace(prim_path="/World/Robot"))

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
            eyes=torch.tensor([list(CAM_EYE)], device=sim.device),
            targets=torch.tensor([list(CAM_TARGET)], device=sim.device),
        )
        # the RTX pipeline needs a few frames before the annotator returns
        # anything but black -- without this the clip opens on dead frames
        for _ in range(20):
            sim.render()
        camera.update(sim.get_physics_dt(), force_recompute=True)

    retry_stats = {"frames_retried": 0, "max_extra": 0, "still_uniform": 0, "render_s": 0.0}

    def grab(step: int) -> None:
        """Render and keep a VERIFIED frame every VIDEO_EVERY-th physics step.

        The RTX pipeline needs several passes before the annotator holds a
        complete image -- read it too early and you get a uniform buffer
        (black, or a flat clear-colour 243) with no error raised. How many
        passes suffice is scene-dependent, so instead of trusting a fixed
        count, each frame is checked for content (std > UNIFORM_STD) and
        re-rendered until it has some, up to MAX_EXTRA_PASSES.
        """
        if camera is None or step % VIDEO_EVERY != 0:
            return
        t0 = time.perf_counter()
        for _ in range(RENDER_PASSES):
            sim.render()
        camera.update(sim.get_physics_dt(), force_recompute=True)
        frame = camera.data.output["rgb"][0, ..., :3].cpu().numpy().copy()
        extra = 0
        while float(frame.std()) < UNIFORM_STD and extra < MAX_EXTRA_PASSES:
            sim.render()
            camera.update(sim.get_physics_dt(), force_recompute=True)
            frame = camera.data.output["rgb"][0, ..., :3].cpu().numpy().copy()
            extra += 1
        if extra:
            retry_stats["frames_retried"] += 1
            retry_stats["max_extra"] = max(retry_stats["max_extra"], extra)
        if float(frame.std()) < UNIFORM_STD:
            retry_stats["still_uniform"] += 1
        retry_stats["render_s"] += time.perf_counter() - t0
        frames.append(frame)

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

        out_path = Path(args_cli.video_path or f"docs/m1_collapse_{args_cli.robot}.mp4")
        if not out_path.is_absolute():
            out_path = Path(__file__).resolve().parents[1] / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(out_path, frames, fps=VIDEO_FPS, codec="libx264")
        print(f"[video  ] wrote {len(frames)} frames ({len(frames) / VIDEO_FPS:.1f} s) to {out_path}")
        print(
            f"[video  ] render retries: {retry_stats['frames_retried']}/{len(frames)} frames needed extra "
            f"passes (max {retry_stats['max_extra']}); still uniform after cap: {retry_stats['still_uniform']}; "
            f"render time {retry_stats['render_s']:.1f} s"
        )
        if retry_stats["frames_retried"] or retry_stats["still_uniform"]:
            print("[video  ] WARNING: retries nonzero -- the deterministic-render settings have regressed")

    assert held_h > STANDING_MIN_HEIGHT, f"phase A: robot did not stand ({held_h:.4f} m)"
    assert fell_torque < 1e-3, f"phase B: actuators still applying torque ({fell_torque:.4f} N*m)"
    assert fell_h < COLLAPSED_MAX_HEIGHT, (
        f"phase B: robot did not collapse ({fell_h:.4f} m) -- something other than the actuators is holding it up"
    )
    assert joint_drift > 0.1, f"phase B: joints barely moved ({joint_drift:.3f} rad) -- are they locked?"
    print(f"[check  ] OK: {args_cli.robot} is held up by actuator torque and collapses without it")


if __name__ == "__main__":
    main()
    simulation_app.close()
