"""March task: walk straight ahead at constant speed — locomotion isolated.

Diagnostic task, built after five walk versions (docs/SPIDERTRON_TASKS.md
iteration 3) fixed every measurable pathology (loopholes, flat kernels,
shuffle pricing, dithering, hopping) without a real gait emerging. Hypothesis:
the full task's competing objectives (heading + speed + height + idle mode)
let reorienting-and-creeping absorb the reward mass. Here, everything except
walking is amputated:

* one fixed command: world +x heading, 0.15 m/s, resampled never;
* no height command (fixed nominal), no idle envs, no curriculum;
* spawn yaw narrowed to ±0.3 rad so episodes are walking, not turning;
* the reward is locomotion + the v9 smoothness/honesty guards + minimal
  regularizers — every term either drives walking or protects hardware.

If a gait develops here, add the command channels back one at a time to find
which kills it. If it does not, the reward-only approach is exhausted and the
gait clock (v10 plan) is next. Keep this task around either way: it is the
fastest possible A/B rig for gait-reward experiments (~2 min/100 iters).
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .spidertron_base_env_cfg import NOMINAL_HEIGHT, SpidertronBaseEnvCfg
from .spidertron_base_env_cfg import ObservationsCfg as BaseObservationsCfg
from .spidertron_base_env_cfg import TerminationsCfg as BaseTerminationsCfg
from .spidertron_walk_env_cfg import FOOT_TIP_OFFSET

# v4: 0.2 -> 0.25 (user direction: faster march; still inside the 0.3 budget)
MARCH_SPEED = 0.25  # m/s

# Run 3: align the swing-time band with the policy's revealed cadence. Across
# every configuration (walk v4-v9, march r1-r2) the policy converges to
# ~0.13-0.14 s swings -- the plant's natural frequency -- and the old
# 0.15 floor / 0.4 target sat entirely above it, making air-time and duty
# antagonists (duty gains were paid for with shrinking swings). Floor 0.08
# still kills genuine taps; target 0.25 is earnable from the natural gait.
MARCH_MIN_AIR_TIME = 0.08  # s
# v4: target 0.25 -> 0.3 -- push toward larger (but stable: slip/clearance
# still enforced) swings now that the floor sits below the natural cadence
MARCH_TARGET_SWING_TIME = 0.3  # s
MARCH_FOOT_CLEARANCE = 0.05  # m, raised from 0.04 for the bigger swings


@configclass
class CommandsCfg:
    # constant command: face world +x, walk at MARCH_SPEED. The huge
    # resampling window means it never changes within an episode; the heading
    # P-controller only corrects push-induced drift.
    base_motion = mdp.DirectionSpeedCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e6, 1.0e6),
        heading_control_stiffness=0.5,
        max_yaw_rate=0.8,
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=mdp.DirectionSpeedCommandCfg.Ranges(heading=(0.0, 0.0), speed=(MARCH_SPEED, MARCH_SPEED)),
    )


@configclass
class ObservationsCfg(BaseObservationsCfg):
    """Base proprioception + motion command + binary foot contacts (73 dims)."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_motion"})
        foot_contacts = ObsTerm(
            func=mdp.foot_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia")},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class MarchRewardsCfg:
    """Locomotion + v9 smoothness/honesty guards + minimal regularizers."""

    # -- the objective
    track_forward_vel = RewTerm(
        func=mdp.track_forward_vel_exp,
        weight=3.0,
        params={"std": 0.5 * MARCH_SPEED, "command_name": "base_motion"},
    )
    track_heading = RewTerm(func=mdp.track_heading_cos, weight=1.0, params={"command_name": "base_motion"})
    base_height_exp = RewTerm(
        func=mdp.base_height_target_exp,
        weight=1.0,
        params={"target_height": NOMINAL_HEIGHT, "std": 0.05},
    )

    # -- gait (the terms under study; weights inherited from the walk task)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_target_active,
        # run 2: 6 -> 8, pushing the slow-cadence solution (run 1 satisfied
        # duty with fast shallow cycles whose swings sat under the floor)
        weight=8.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "command_name": "base_motion",
            "target_time": MARCH_TARGET_SWING_TIME,
            "min_air_time": MARCH_MIN_AIR_TIME,
        },
    )
    stance_progress = RewTerm(
        func=mdp.stance_progress,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_tibia"),
            "slip_tol": 0.05,
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_tibia"),
            "target_height": MARCH_FOOT_CLEARANCE,
            "tip_offset": FOOT_TIP_OFFSET,
        },
    )
    # -- v4 gait structure: alternating tripods, no flight phases
    tripod_antiphase = RewTerm(
        func=mdp.tripod_antiphase,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "command_name": "base_motion",
        },
    )
    # v5: two-sided replacement for too_many_feet_airborne -- exactly 3 feet
    # planted is the tripod invariant; hops (<3) keep the old pricing and
    # drag/overlap phases (4-6) now cost the same
    contact_count = RewTerm(
        func=mdp.contact_count_deviation,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "command_name": "base_motion",
            "target": 3,
        },
    )
    # v5: the three legs of each tripod must share contact time equally
    # (antiphase only constrains group means; this constrains within-group)
    tripod_balance = RewTerm(
        func=mdp.tripod_contact_time_balance,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "command_name": "base_motion",
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_tibia"),
        },
    )
    feet_airborne_too_long = RewTerm(
        func=mdp.feet_airborne_too_long,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "command_name": "base_motion",
            "max_air_time": 1.0,
        },
    )
    # load coupling (the v9 tap-dance fix): every foot's contact duty must sit
    # in [0.45, 0.75] while moving -- decorative 2%-duty legs and two-leg
    # scuttling both become illegal per foot
    foot_duty = RewTerm(
        func=mdp.foot_duty_deviation,
        # run 2: -2 -> -4 -- run 1 plateaued at duty_min ~0.22 with tracking
        # secured (2.8/3.0), so the auction can bear heavier load-sharing
        # pressure
        weight=-4.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "command_name": "base_motion",
            "target": 0.6,
            "tol": 0.15,
        },
    )

    # -- smoothness (the v9 wins, unchanged)
    torque_saturation = RewTerm(func=mdp.torque_saturation, weight=-5.0, params={"threshold": 0.9})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.2)

    # -- stability + regularizers, minimal set
    lat_vel_l2 = RewTerm(func=mdp.base_lin_vel_y_l2, weight=-2.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    base_ang_acc_l2 = RewTerm(
        func=mdp.base_ang_acc_l2,
        weight=-2.5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_femur", ".*_coxa"]),
            "threshold": 1.0,
        },
    )


@configclass
class MarchMaxRewardsCfg(MarchRewardsCfg):
    """Max-speed variant: identical constraint suite, but forward speed is a
    LINEAR reward instead of setpoint tracking -- finds the fastest gait the
    tripod/honesty/smoothness constraints permit."""

    # weight 8 -> 5 after the first attempt: at 8 the linear term dominated
    # everything and the optimal policy was a 0.85 s lunge-and-fall
    forward_velocity = RewTerm(func=mdp.forward_velocity, weight=5.0)
    # dying must cost more than a lunge earns: without this, early termination
    # SAVES accumulated penalties and the fall itself is free (max-variant
    # first attempt: 100% bad_orientation terminations)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    def __post_init__(self):
        self.track_forward_vel = None


@configclass
class MarchFreeRewardsCfg:
    """Free-run variant (emergent-gait control experiment): maximize forward
    velocity with NO foot-contact/gait-pattern terms at all -- no tripod, no
    duty, no air-time, no slip, no contact count. Only survival (heavy
    termination penalty), direction, height, hardware-feasibility smoothness,
    and basic regularizers remain. Question under test: is speed + survival
    sufficient for a mechanically stable alternating gait to EMERGE, or was
    the prior doing real work?"""

    forward_velocity = RewTerm(func=mdp.forward_velocity, weight=5.0)
    # "heavily penalize falling over" -- 2x the max-variant's penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    track_heading = RewTerm(func=mdp.track_heading_cos, weight=1.0, params={"command_name": "base_motion"})
    base_height_exp = RewTerm(
        func=mdp.base_height_target_exp,
        weight=1.0,
        params={"target_height": NOMINAL_HEIGHT, "std": 0.05},
    )

    # hardware feasibility (not gait pattern): no dithering, no stall
    torque_saturation = RewTerm(func=mdp.torque_saturation, weight=-5.0, params={"threshold": 0.9})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.2)
    # RECLASSIFIED as a physics constraint (march-free2 post-mortem): the
    # solver under-enforces friction on brief high-speed contacts, so sliding
    # loaded feet are cheaper in sim than physics allows. This term is the
    # reward-side backstop for real contact economy, alongside the sim-side
    # contact fixes (offset, solver iterations) in robots/spidertron.py.
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_tibia"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_tibia"),
        },
    )

    # basic regularizers; keeps chassis/femur off the ground (body posture,
    # not foot placement)
    lat_vel_l2 = RewTerm(func=mdp.base_lin_vel_y_l2, weight=-2.0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_femur", ".*_coxa"]),
            "threshold": 1.0,
        },
    )


@configclass
class MarchTerminationsCfg(BaseTerminationsCfg):
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class MarchMetricsCfg:
    """Translation probe riding the curriculum logging channel: gait-structure
    curves in TensorBoard every iteration (see mdp.foot_duty_metric)."""

    metric_foot_duty_min = CurrTerm(
        func=mdp.foot_duty_metric, params={"sensor_name": "contact_forces", "reduce": "min"}
    )
    metric_foot_duty_mean = CurrTerm(
        func=mdp.foot_duty_metric, params={"sensor_name": "contact_forces", "reduce": "mean"}
    )
    metric_tripod_antiphase = CurrTerm(
        func=mdp.tripod_antiphase_metric, params={"sensor_name": "contact_forces"}
    )
    # harness-level friction-evasion tripwire (march-free2 post-mortem)
    metric_foot_slip = CurrTerm(func=mdp.foot_slip_metric, params={"sensor_name": "contact_forces"})


@configclass
class SpidertronMarchEnvCfg(SpidertronBaseEnvCfg):
    """Spidertron marching straight ahead at constant speed."""

    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: MarchRewardsCfg = MarchRewardsCfg()
    terminations: MarchTerminationsCfg = MarchTerminationsCfg()
    curriculum: MarchMetricsCfg = MarchMetricsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        # start roughly facing the march direction: episodes should be spent
        # walking, not demonstrating the (already-mastered) reorientation
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.3, 0.3)
        # v9 exploration pairing (product 0.245 <= proven 0.25)
        self.actions.joint_pos.scale = 0.35


@configclass
class SpidertronMarchMaxEnvCfg(SpidertronMarchEnvCfg):
    """Max-speed march: linear velocity reward, same constraints."""

    rewards: MarchMaxRewardsCfg = MarchMaxRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # commanded speed becomes a constant "go fast" observation; the reward
        # no longer references it
        self.commands.base_motion.ranges.speed = (0.5, 0.5)


@configclass
class SpidertronMarchMaxEnvCfg_PLAY(SpidertronMarchMaxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class SpidertronMarchFreeEnvCfg(SpidertronMarchMaxEnvCfg):
    """Free-run: max speed, survival, no gait prior. Keeps the duty/tripod
    curriculum METRICS (pure logging) so emergence is measurable."""

    rewards: MarchFreeRewardsCfg = MarchFreeRewardsCfg()


@configclass
class SpidertronMarchFreeEnvCfg_PLAY(SpidertronMarchFreeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class SpidertronMarchEnvCfg_PLAY(SpidertronMarchEnvCfg):
    """Small, deterministic version for watching rollouts."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
