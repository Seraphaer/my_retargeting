"""FK round trip and an unreachable target for the mink arm solver."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
URDF_PATH = Path(__file__).resolve().parents[1] / "urdf" / "g1_29dof_with_dex1_1.urdf"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from arm_ik.model import ArmModel


def test_forward_kinematics_reaches_a_finite_jaw() -> None:
    """The URDF arm has a positive reach and a finite jaw at the nominal pose."""

    model = ArmModel.from_urdf(URDF_PATH, "left")
    jaw, links = model.forward(model.nominal_joint_rad())
    assert 0.3 < model.reach_m < 1.2
    assert np.all(np.isfinite(jaw))
    assert "left_wrist_yaw_link" in links


def test_roundtrip_position_error_stays_under_five_centimeters() -> None:
    """A pose produced by FK is feasible for the same arm."""

    stage = _load_stage()
    solver = stage.make_mink_solver(URDF_PATH, "right")
    model = ArmModel.from_urdf(URDF_PATH, "right")
    joints = model.clip_joint_rad(model.nominal_joint_rad() + np.array([0.2, -0.15, 0.1, -0.25, 0.2, -0.1, 0.15]))
    jaw, _links = model.forward(joints)
    result = stage.solve_gripper_in_base(jaw, solver)
    assert result.succeeded
    assert result.position_error_m < 0.05
    assert model.within_limits(result.joint_rad)


def test_unreachable_target_is_a_failure_without_zero_joints() -> None:
    """A target many meters away does not come back as the zero configuration."""

    stage = _load_stage()
    solver = stage.make_mink_solver(URDF_PATH, "left")
    target = np.eye(4)
    target[:3, 3] = [8.0, 0.0, 0.0]
    result = stage.solve_gripper_in_base(target, solver)
    assert not result.succeeded
    assert np.isnan(result.joint_rad).all()


def _load_stage():
    path = PIPELINE_DIR / "8.1_solve_arm_ik.py"
    spec = spec_from_file_location("stage_8_1_solve_arm_ik", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}.")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
