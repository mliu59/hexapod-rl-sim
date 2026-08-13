"""Random-policy throughput benchmark for the hexapod (milestone M1).

Measures how fast the hexapod steps at training scale, so the number can be
compared against the ANYmal-C rough baseline (~35-37k steps/s at 4096 envs).
"steps" here means *policy* steps, matching what rsl-rl reports: one policy
step = DECIMATION physics steps at the same action, i.e. 50 Hz control on a
200 Hz sim.

No gym env is involved -- this drives the articulation directly, so it works
before the hexapod task config exists.

Usage:
    python scripts/bench_hexapod.py --headless
    python scripts/bench_hexapod.py --headless --num_envs 1024 --iters 300
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hexapod random-policy throughput benchmark.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel hexapods.")
parser.add_argument("--iters", type=int, default=500, help="Timed policy steps.")
parser.add_argument("--warmup", type=int, default=50, help="Untimed policy steps before timing starts.")
parser.add_argument("--action_scale", type=float, default=0.25, help="Random action amplitude (rad).")
parser.add_argument(
    "--decimation",
    type=int,
    default=4,
    help="Physics steps per policy step. 4 matches the rsl-rl convention the ANYmal number was quoted in; "
    "use 2 to match Hexapod-Flat-v0.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from hexapod_lab.robots import HEXAPOD_CFG

PHYSICS_DT = 0.005  # 200 Hz, same as the ANYmal baseline
DECIMATION = args_cli.decimation  # 4 -> 50 Hz control, 2 -> 100 Hz (the M2 env)


@configclass
class HexapodBenchSceneCfg(InteractiveSceneCfg):
    """Flat ground, one hexapod per env."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = HEXAPOD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def gpu_used_gb() -> float:
    """Total GPU memory in use, driver-reported (includes PhysX and Kit, not just torch)."""
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device))

    scene_cfg = HexapodBenchSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0)
    t0 = time.perf_counter()
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print(f"[bench] scene built in {time.perf_counter() - t0:.1f} s")

    robot = scene["robot"]
    default_pos = robot.data.default_joint_pos
    print(f"[bench] {args_cli.num_envs} envs x {robot.num_joints} joints, device {sim.device}")
    print(f"[bench] GPU in use after build: {gpu_used_gb():.2f} GB")

    def run(iters: int) -> float:
        start = time.perf_counter()
        for _ in range(iters):
            # fresh random action once per policy step, held across decimation
            targets = default_pos + args_cli.action_scale * (2.0 * torch.rand_like(default_pos) - 1.0)
            for _ in range(DECIMATION):
                robot.set_joint_position_target(targets)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(PHYSICS_DT)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    run(args_cli.warmup)
    print(f"[bench] warmup done ({args_cli.warmup} policy steps)")

    elapsed = run(args_cli.iters)

    policy_steps = args_cli.iters * args_cli.num_envs
    physics_steps = policy_steps * DECIMATION
    print(f"\n[result] {args_cli.iters} policy steps x {args_cli.num_envs} envs in {elapsed:.2f} s")
    print(f"[result] {policy_steps / elapsed:,.0f} policy steps/s   (rsl-rl 'steps/s' convention)")
    print(f"[result] {physics_steps / elapsed:,.0f} physics steps/s")
    print(f"[result] {elapsed / args_cli.iters * 1e3:.2f} ms per policy step ({args_cli.num_envs} envs)")
    print(f"[result] sim time simulated: {args_cli.iters * DECIMATION * PHYSICS_DT:.1f} s, "
          f"real-time factor {args_cli.iters * DECIMATION * PHYSICS_DT * args_cli.num_envs / elapsed:,.0f}x")
    print(f"[result] peak GPU in use: {gpu_used_gb():.2f} GB")

    fallen = (robot.data.root_pos_w[:, 2] < 0.06).float().mean().item()
    print(f"[note  ] {fallen * 100:.0f}% of envs are on the deck after random actions (expected -- no policy)")


if __name__ == "__main__":
    main()
    simulation_app.close()
