"""Module 8.2: place the two arm bases with the Ego2Robot search.

This file only forwards the gripper trajectory to the search. The grid, prune,
keyframe selection, and score live in ``anchor``. IK lives in module 8.1.
"""

from __future__ import annotations

from anchor.candidates import AnchorSearchConfig
from anchor.search import search_arm_anchors
from arm_ik.model import ArmModel
from arm_ik.solver import ArmSolver
from contracts import ArmAnchors, GripperPoses


def auto_anchor(
    poses: GripperPoses,
    solvers: dict[str, ArmSolver],
    models: dict[str, ArmModel],
    config: AnchorSearchConfig | None = None,
) -> ArmAnchors:
    """Return one independent base for each arm that has valid frames.

    Args:
        poses: Module 7 poses in the left rectified camera.
        solvers: Feasibility solvers keyed by ``left`` and ``right``.
        models: Arm models keyed the same way. They supply reach only.
        config: Paper constants. ``None`` uses the appendix grid.
    Returns:
        Left and right anchors. A side with no valid frames is absent.
    """

    return search_arm_anchors(poses, solvers, models, AnchorSearchConfig() if config is None else config)
