"""Module 8.1: solve one gripper pose in one arm base frame.

The iteration lives in ``arm_ik``. This file checks the pose and calls the
injected solver. It does not search a base or draw a video.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arm_ik.model import ArmModel
from arm_ik.solver import ArmSolver, IkConfig, IkResult, MinkArmSolver


def solve_gripper_in_base(target_in_base: np.ndarray, solver: ArmSolver) -> IkResult:
    """Solve one jaw pose with the supplied arm solver.

    Args:
        target_in_base: 4x4 jaw pose in the arm base. Translation is meters.
            Quaternion order is not used here; the rotation is the matrix.
        solver: Feasibility solver, normally ``MinkArmSolver``.
    Returns:
        The solver result. Failure carries NaN joints.
    """

    pose = np.asarray(target_in_base, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("target_in_base must be a finite 4x4 transform.")
    return solver.solve(pose)


def make_mink_solver(urdf_path: Path, side: str, config: IkConfig | None = None) -> MinkArmSolver:
    """Build the paper's mink solver for one G1 arm.

    Args:
        urdf_path: G1 URDF with Dex1 grippers.
        side: ``left`` or ``right``.
        config: Iteration budget. ``None`` uses 100 iterations and 1e-5.
    """

    return MinkArmSolver(ArmModel.from_urdf(urdf_path, side), IkConfig() if config is None else config)
