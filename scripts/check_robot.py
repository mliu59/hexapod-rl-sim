"""Load the placeholder hexapod in Isaac Sim and sanity-check it.

Checks (precursor to milestone M1):
- URDF -> USD conversion succeeds and the articulation spawns
- joint count/names match the 18-DOF layout
- robot settles under gravity from the default pose instead of exploding

Usage:
    python scripts/check_robot.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sanity-check the hexapod articulation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from hexapod_lab.robots import HEXAPOD_CFG


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/light", light_cfg)

    robot = Articulation(HEXAPOD_CFG.replace(prim_path="/World/Robot"))

    sim.reset()

    print(f"[check] bodies ({robot.num_bodies}): {robot.body_names}")
    print(f"[check] joints ({robot.num_joints}): {robot.joint_names}")
    assert robot.num_joints == 18, f"expected 18 joints, got {robot.num_joints}"
    assert robot.num_bodies == 19, f"expected 19 bodies, got {robot.num_bodies}"

    start_h = robot.data.root_pos_w[0, 2].item()
    # settle for 2 s of sim time with PD holding the default pose
    for _ in range(400):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
    end_h = robot.data.root_pos_w[0, 2].item()

    print(f"[check] root height: spawn {start_h:.3f} m -> settled {end_h:.3f} m")
    names = robot.joint_names
    target = robot.data.default_joint_pos[0]
    actual = robot.data.joint_pos[0]
    torque = robot.data.applied_torque[0]
    for i, n in enumerate(names):
        print(f"[joint] {n:16s} target {target[i]:+.3f}  actual {actual[i]:+.3f}  torque {torque[i]:+.3f}")
    print(f"[check] stiffness: {robot.data.joint_stiffness[0, :3].tolist()}")
    print(f"[check] damping:   {robot.data.joint_damping[0, :3].tolist()}")
    assert 0.06 < end_h < 0.4, f"implausible settled height {end_h:.3f} m (robot not standing)"
    print("[check] OK: hexapod loads, articulates, and stands under PD hold")


if __name__ == "__main__":
    main()
    simulation_app.close()
