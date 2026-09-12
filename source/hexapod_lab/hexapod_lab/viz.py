"""Visualization helpers for rollout videos."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


def base_frame_marker(env, scale: float = 0.1) -> Callable[[], None]:
    """Attach an RGB axis triad to the robot base for orientation readability.

    Renders the base frame (x red, y green, z blue) in every env, so recorded
    rollouts show heading and tilt directly instead of leaving orientation to
    be guessed from the leg silhouette. Returns an ``update()`` callable to
    invoke once per env step (markers do not track prims by themselves).

    ``env`` is the unwrapped ManagerBasedRLEnv (pass ``env.unwrapped`` when
    holding a wrapper).
    """
    cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/base_frame")
    cfg.markers["frame"].scale = (scale, scale, scale)
    markers = VisualizationMarkers(cfg)
    robot = env.scene["robot"]

    def update() -> None:
        markers.visualize(robot.data.root_pos_w, robot.data.root_quat_w)

    return update
