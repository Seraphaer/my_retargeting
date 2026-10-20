"""Synthetic checks for the paper base search. The IK solver is a fake."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from anchor.candidates import AnchorSearchConfig, base_positions
from anchor.keyframes import select_keyframes
from anchor.prune import is_base_admissible
from anchor.score import score_placement
from arm_ik.solver import IkResult
from contracts import GripperPoseTrack, GripperPoses, load_arm_anchors, save_arm_anchors
from se3 import transform


class _Reach:
    def __init__(self, reach_m: float) -> None:
        self.reach_m = reach_m


class _NearTargetSolver:
    """Feasible only when the jaw origin is 0.65 m in front of the base."""

    def solve(self, target_in_base: np.ndarray) -> IkResult:
        error = float(np.linalg.norm(target_in_base[:3, 3] - np.array([0.0, 0.0, 0.65])))
        if error < 0.02:
            return IkResult(True, error, 0.0, np.full(7, 0.2))
        return IkResult(False, error, 1.0, np.full(7, np.nan))


def test_prune_rejects_camera_near_and_stretch_limits() -> None:
    """The appendix drops a base that is too close to the camera, the hand, or full stretch."""

    config = AnchorSearchConfig()
    points = np.array([[0.4, 0.0, 0.8]])
    assert not is_base_admissible(np.array([0.0, 0.0, 0.1]), points, 1.0, config)
    assert not is_base_admissible(np.array([0.4, 0.0, 0.75]), points, 1.0, config)
    assert not is_base_admissible(np.array([0.4, 0.0, 2.0]), points, 1.0, config)
    assert is_base_admissible(np.array([0.4, 0.0, 0.2]), points, 1.0, config)


def test_keyframes_keep_the_spatial_extreme_under_the_cap() -> None:
    """More than 20 valid frames still retain the point farthest from the centroid."""

    count = 30
    positions = np.zeros((count, 3))
    positions[-1] = [3.0, 0.0, 0.0]
    quaternions = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (count, 1))
    chosen = select_keyframes(positions, quaternions, np.ones(count, dtype=bool), 20)
    assert len(chosen) == 20
    assert 29 in chosen


def test_score_matches_the_paper_formula() -> None:
    """S is the feasibility rate minus five times the reach-fraction error."""

    config = AnchorSearchConfig()
    base = transform(np.eye(3), np.array([0.0, 0.0, 1.0]))
    positions = np.array([[0.0, 0.0, 1.65], [0.0, 0.0, 1.65]])
    quaternions = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1))
    score = score_placement(_NearTargetSolver(), base, positions, quaternions, np.array([0, 1]), 1.0, config)
    assert np.isclose(score.feasibility_rate, 1.0)
    assert np.isclose(score.mean_reach_fraction, 0.65)
    assert np.isclose(score.score, 1.0)


def test_grid_is_anchored_at_the_camera() -> None:
    """Appendix offsets start at the camera, not at the hand."""

    config = AnchorSearchConfig(lateral_fractions=(0.3,), forward_fractions=(0.5,), vertical_fractions=(0.2,))
    assert np.allclose(base_positions("right", 1.0, config), [[0.3, 0.2, 0.5]])
    assert np.allclose(base_positions("left", 1.0, config), [[-0.3, 0.2, 0.5]])


def test_pair_search_selects_the_only_feasible_bases() -> None:
    """Each arm keeps its own camera-frame base, and that pair outranks the others."""

    poses = GripperPoses(
        left=_track("left", [-0.4, 0.0, 0.95]),
        right=_track("right", [0.4, 0.0, 0.95]),
    )
    config = AnchorSearchConfig(
        lateral_fractions=(0.4, 0.8),
        forward_fractions=(0.3, 0.7),
        vertical_fractions=(0.0,),
        pitch_deg=(0.0,),
        yaw_deg=(0.0,),
        roll_deg=(0.0,),
    )
    solvers = {"left": _NearTargetSolver(), "right": _NearTargetSolver()}
    models = {"left": _Reach(1.0), "right": _Reach(1.0)}
    anchors = _stage82().auto_anchor(poses, solvers, models, config)
    assert anchors.left.present and anchors.right.present
    assert np.allclose(anchors.left.position_m, [-0.4, 0.0, 0.3], atol=1e-6)
    assert np.allclose(anchors.right.position_m, [0.4, 0.0, 0.3], atol=1e-6)
    assert anchors.pair_score > 0.5


def test_a_hand_with_no_valid_frames_stays_absent() -> None:
    """The remaining arm still receives its top placement."""

    poses = GripperPoses(
        left=_track("left", [-0.4, 0.0, 0.95]),
        right=_track("right", [0.4, 0.0, 0.95], valid=False),
    )
    config = AnchorSearchConfig(
        lateral_fractions=(0.4,),
        forward_fractions=(0.3,),
        vertical_fractions=(0.0,),
        pitch_deg=(0.0,),
        yaw_deg=(0.0,),
        roll_deg=(0.0,),
    )
    anchors = _stage82().auto_anchor(
        poses,
        {"left": _NearTargetSolver()},
        {"left": _Reach(1.0), "right": _Reach(1.0)},
        config,
    )
    assert anchors.left.present
    assert not anchors.right.present
    assert np.isnan(anchors.right.position_m).all()


def test_anchor_roundtrip_preserves_an_absent_side(tmp_path: Path) -> None:
    """Saving the search result does not invent a pose for a missing arm."""

    poses = GripperPoses(left=_track("left", [-0.4, 0.0, 0.95]), right=_track("right", [0.4, 0.0, 0.95], valid=False))
    config = AnchorSearchConfig(
        lateral_fractions=(0.4,),
        forward_fractions=(0.3,),
        vertical_fractions=(0.0,),
        pitch_deg=(0.0,),
        yaw_deg=(0.0,),
        roll_deg=(0.0,),
    )
    anchors = _stage82().auto_anchor(
        poses,
        {"left": _NearTargetSolver()},
        {"left": _Reach(1.0), "right": _Reach(1.0)},
        config,
    )
    loaded = load_arm_anchors(save_arm_anchors(anchors, tmp_path / "arm_anchors.npz"))
    assert loaded.left.present
    assert not loaded.right.present
    assert np.allclose(loaded.left.position_m, anchors.left.position_m)


def test_stage_81_returns_the_solver_result() -> None:
    """Module 8.1 does not replace a failed solve with zeros."""

    target = np.eye(4)
    target[:3, 3] = [5.0, 0.0, 0.0]
    result = _stage81().solve_gripper_in_base(target, _NearTargetSolver())
    assert not result.succeeded
    assert np.isnan(result.joint_rad).all()


def _track(side: str, position: list[float], valid: bool = True, frames: int = 4) -> GripperPoseTrack:
    indices = np.arange(frames, dtype=np.int64)
    positions = np.repeat(np.asarray(position, dtype=float)[None, :], frames, axis=0)
    quaternions = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (frames, 1))
    return GripperPoseTrack(
        side=side,
        frame_indices=indices,
        timestamps_ns=indices * 33_000_000 + 1_000,
        position_m=positions,
        quaternion_xyzw=quaternions,
        opening_distance_m=np.full(frames, 0.04),
        opening_width=np.full(frames, np.nan),
        invalid_mask=np.full(frames, not valid),
        jaw_degenerate_count=0,
        frame_degenerate_count=0,
    )


def _stage81():
    return _load_stage("8.1_solve_arm_ik.py")


def _stage82():
    return _load_stage("8.2_auto_anchor.py")


def _load_stage(filename: str):
    path = PIPELINE_DIR / filename
    spec = spec_from_file_location(filename.replace(".", "_"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}.")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
