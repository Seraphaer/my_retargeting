"""One-frame arm IK behind a small solver protocol.

``MinkArmSolver`` follows Ego2Robot: mink, the quadprog backend, at most 100
iterations, and a 1e-5 pose threshold. A frame is feasible when the jaw
position error is below 0.05 m. Failure returns NaN joints, never a zero
vector. This module does not choose a robot base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from arm_ik.model import ArmModel
from se3 import orientation_error_rad


@dataclass(frozen=True)
class IkConfig:
    """Iteration budget and the two paper thresholds.

    ``convergence_tolerance`` stops the loop early. ``feasible_position_error_m``
    is the L1 tracking gate that decides success. ``damping`` is the QP damping
    passed to mink and is not a second success threshold.
    """

    maximum_iterations: int = 100
    convergence_tolerance: float = 1e-5
    feasible_position_error_m: float = 0.05
    time_step_s: float = 0.02
    damping: float = 1e-3
    qp_solver: str = "quadprog"

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive.")
        if self.convergence_tolerance <= 0.0 or self.feasible_position_error_m <= 0.0:
            raise ValueError("IK tolerances must be positive.")
        if self.time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive.")


@dataclass(frozen=True)
class IkResult:
    """One attempt to place the jaw on a target pose in the arm base.

    ``joint_rad`` is NaN when ``succeeded`` is false. A failed result is not
    allowed to look like the zero configuration.
    """

    succeeded: bool
    position_error_m: float
    orientation_error_rad: float
    joint_rad: np.ndarray

    def __post_init__(self) -> None:
        angles = np.asarray(self.joint_rad, dtype=float)
        if angles.shape != (7,):
            raise ValueError(f"joint_rad must have shape (7,), received {angles.shape}.")
        object.__setattr__(self, "joint_rad", angles)
        if self.succeeded:
            if not np.all(np.isfinite(angles)):
                raise ValueError("A successful IK result requires finite joint angles.")
            return
        if np.all(np.isfinite(angles)) and np.allclose(angles, 0.0):
            raise ValueError("A failed IK result must not report an all-zero joint vector.")


class ArmSolver(Protocol):
    """Solve one jaw pose expressed in the arm base."""

    def solve(self, target_in_base: np.ndarray) -> IkResult:
        """Return feasibility for a 4x4 jaw pose. Translation is meters."""


class MinkArmSolver:
    """Mink differential IK on the MuJoCo arm built from ``ArmModel``."""

    def __init__(self, model: ArmModel, config: IkConfig | None = None) -> None:
        self._arm = model
        self._config = IkConfig() if config is None else config
        mink, mujoco = _require_mink()
        _require_quadprog(self._config.qp_solver)
        self._mink = mink
        self._mujoco_model = mujoco.MjModel.from_xml_string(model.mjcf())
        self._configuration = mink.Configuration(self._mujoco_model)
        self._task = mink.FrameTask(
            frame_name="jaw",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        self._limit = mink.ConfigurationLimit(self._mujoco_model)

    def solve(self, target_in_base: np.ndarray) -> IkResult:
        """Iterate from the mid-range seed toward one jaw pose.

        Args:
            target_in_base: 4x4 jaw pose in the arm base. Translation is meters.
        Returns:
            Success when the jaw position is within 0.05 m and the joints lie
            inside the URDF limits. Otherwise joints are NaN.
        """

        target = _require_pose(target_in_base)
        seed = self._arm.nominal_joint_rad()
        self._configuration.update(seed)
        self._task.set_target(_se3(self._mink, target))
        position_error = float("inf")
        orientation_error = float("inf")
        joint_rad = seed
        for _iteration in range(self._config.maximum_iterations):
            velocity = self._mink.solve_ik(
                self._configuration,
                [self._task],
                self._config.time_step_s,
                self._config.qp_solver,
                self._config.damping,
                limits=[self._limit],
            )
            self._configuration.integrate_inplace(velocity, self._config.time_step_s)
            joint_rad = np.asarray(self._configuration.q, dtype=float).reshape(-1)[:7].copy()
            jaw, _links = self._arm.forward(joint_rad)
            position_error = float(np.linalg.norm(jaw[:3, 3] - target[:3, 3]))
            orientation_error = orientation_error_rad(jaw[:3, :3], target[:3, :3])
            if (
                position_error < self._config.convergence_tolerance
                and orientation_error < self._config.convergence_tolerance
            ):
                break
        if position_error < self._config.feasible_position_error_m and self._arm.within_limits(joint_rad):
            return IkResult(
                succeeded=True,
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
                joint_rad=joint_rad,
            )
        return IkResult(
            succeeded=False,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            joint_rad=np.full(7, np.nan),
        )


def _se3(mink, pose: np.ndarray):
    if hasattr(mink.SE3, "from_matrix"):
        return mink.SE3.from_matrix(np.asarray(pose, dtype=float))
    return mink.SE3.from_rotation_and_translation(pose[:3, :3], pose[:3, 3])


def _require_mink():
    try:
        import mink
        import mujoco
    except ImportError as error:
        raise RuntimeError("MinkArmSolver requires the mink and mujoco packages.") from error
    return mink, mujoco


def _require_quadprog(solver_name: str) -> None:
    import qpsolvers

    if solver_name not in qpsolvers.available_solvers:
        installed = ", ".join(qpsolvers.available_solvers) or "none"
        raise RuntimeError(
            f"Ego2Robot IK expects the {solver_name} backend. Installed QP solvers: {installed}."
        )


def _require_pose(target_in_base: np.ndarray) -> np.ndarray:
    pose = np.asarray(target_in_base, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("target_in_base must be a finite 4x4 transform.")
    return pose
