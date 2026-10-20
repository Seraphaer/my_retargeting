"""Run HumanPlus preparation, WiLoR extraction, gripper poses, and base search.

Edit CONFIG before running. This script saves each stage and does not draw
video. Render keypoints with ``test/render_keypoint_stages.py`` and the solved
arms with ``test/render_arm_ik.py``.
"""

from __future__ import annotations

from datetime import datetime
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

from arm_ik.model import ArmModel
from contracts import (
    load_rectified_stereo,
    save_camera_keypoints,
    save_gripper_poses,
    save_rectified_stereo,
    save_repaired_hands,
    save_triangulated_hands,
    save_arm_anchors,
)


def _load_stage(filename: str):
    """Load a dotted stage filename such as ``8.1_solve_arm_ik.py``."""

    path = Path(__file__).with_name(filename)
    spec = spec_from_file_location(filename.replace(".", "_"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline stage {path}.")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_prepare = import_module("1_prepare_humanplus")
_extract = import_module("2_extract_wilor")
_smooth = import_module("3_smooth_keypoints")
_triangulate = import_module("4_triangulate_keypoints")
_fill = import_module("5_fill_keypoints")
_gripper = import_module("6_map_gripper_pose")
_filter = import_module("7_filter_gripper_pose")
PrepareConfig = _prepare.PrepareConfig
prepare_humanplus = _prepare.prepare_humanplus
WiLoRExtractionConfig = _extract.WiLoRExtractionConfig
extract_wilor = _extract.extract_wilor
SmoothConfig = _smooth.SmoothConfig
smooth_camera_keypoints = _smooth.smooth_camera_keypoints
rectified_video_fps = _smooth.rectified_video_fps
TriangulationConfig = _triangulate.TriangulationConfig
triangulate_smoothed_keypoints = _triangulate.triangulate_smoothed_keypoints
FillConfig = _fill.FillConfig
fill_triangulated_hands = _fill.fill_triangulated_hands
JawPoseConfig = _gripper.JawPoseConfig
map_gripper_poses = _gripper.map_gripper_poses
GripperFilterConfig = _filter.GripperFilterConfig
STAGE_ORDER = _filter.STAGE_ORDER
filter_gripper_poses = _filter.filter_gripper_poses
_arm_ik = _load_stage("8.1_solve_arm_ik.py")
_anchor = _load_stage("8.2_auto_anchor.py")
make_mink_solver = _arm_ik.make_mink_solver
auto_anchor = _anchor.auto_anchor


# =============================================================================
# CONFIG: edit these values before running this script.
# =============================================================================
MY_RETARGETING_ROOT = Path(__file__).resolve().parents[1]
SESSION_ROOT = (
    MY_RETARGETING_ROOT / "data" / "humanplus-1000" / "data" / "session_20260903_113804"
)
RUN_DIRECTORY = MY_RETARGETING_ROOT / "output" / f"humanplus_wilor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
WILOR_DEVICE = "cuda"
MINIMUM_DETECTION_CONFIDENCE = 0.30
MAXIMUM_SYNC_DELTA_NS = 0
MAXIMUM_FRAMES = None
MAXIMUM_REPROJECTION_ERROR_PX = 10.0
MAXIMUM_POINT_SPEED_MPS = 2.5
LONG_GAP_FRAMES = 10
SMOOTH_CLOSING_INPUTS = True
SMOOTH_REACH_INPUTS = True
ENABLED_GRIPPER_STAGES = STAGE_ORDER
URDF_PATH = MY_RETARGETING_ROOT / "urdf" / "g1_29dof_with_dex1_1.urdf"


def main() -> None:
    """Prepare one session and save raw, smoothed, triangulated, and filled keypoints."""

    prepare_config = PrepareConfig(
        session_root=SESSION_ROOT,
        output_directory=RUN_DIRECTORY / "rectified",
        maximum_sync_delta_ns=MAXIMUM_SYNC_DELTA_NS,
        maximum_frames=MAXIMUM_FRAMES,
    )
    capture = prepare_humanplus(prepare_config)
    save_rectified_stereo(capture, RUN_DIRECTORY)
    loaded_capture = load_rectified_stereo(RUN_DIRECTORY)
    left_keypoints, right_keypoints = extract_wilor(
        loaded_capture,
        WiLoRExtractionConfig(device=WILOR_DEVICE, minimum_detection_confidence=MINIMUM_DETECTION_CONFIDENCE),
    )
    save_camera_keypoints(left_keypoints, RUN_DIRECTORY / "keypoints_left.npz")
    save_camera_keypoints(right_keypoints, RUN_DIRECTORY / "keypoints_right.npz")
    smoothed_left, smoothed_right = smooth_camera_keypoints(
        left_keypoints,
        right_keypoints,
        loaded_capture.left_rectified_video,
        loaded_capture.right_rectified_video,
        SmoothConfig(),
    )
    save_camera_keypoints(smoothed_left, RUN_DIRECTORY / "keypoints_left_smoothed.npz")
    save_camera_keypoints(smoothed_right, RUN_DIRECTORY / "keypoints_right_smoothed.npz")
    tracks = triangulate_smoothed_keypoints(
        loaded_capture,
        smoothed_left,
        smoothed_right,
        TriangulationConfig(maximum_reprojection_error_px=MAXIMUM_REPROJECTION_ERROR_PX),
    )
    save_triangulated_hands(tracks, RUN_DIRECTORY / "hand_tracks.npz")
    filled = fill_triangulated_hands(
        tracks,
        FillConfig(
            frames_per_second=rectified_video_fps(loaded_capture.left_rectified_video),
            maximum_speed_mps=MAXIMUM_POINT_SPEED_MPS,
            long_gap_frames=LONG_GAP_FRAMES,
        ),
    )
    save_repaired_hands(filled, RUN_DIRECTORY / "filled_hands.npz")
    raw_pose = map_gripper_poses(
        filled,
        JawPoseConfig(smooth_closing_inputs=SMOOTH_CLOSING_INPUTS, smooth_reach_inputs=SMOOTH_REACH_INPUTS),
    )
    save_gripper_poses(raw_pose, RUN_DIRECTORY / "gripper_pose_raw.npz")
    pose_snapshots = filter_gripper_poses(
        raw_pose,
        GripperFilterConfig(
            frames_per_second=rectified_video_fps(loaded_capture.left_rectified_video),
            enabled_stages=ENABLED_GRIPPER_STAGES,
            long_gap_frames=LONG_GAP_FRAMES,
        ),
    )
    final_pose = raw_pose
    for stage_name, stage_pose in pose_snapshots:
        save_gripper_poses(stage_pose, RUN_DIRECTORY / f"gripper_pose_after_{stage_name}.npz")
        final_pose = stage_pose
    save_gripper_poses(final_pose, RUN_DIRECTORY / "gripper_poses.npz")
    anchors = auto_anchor(
        final_pose,
        {
            "left": make_mink_solver(URDF_PATH, "left"),
            "right": make_mink_solver(URDF_PATH, "right"),
        },
        {
            "left": ArmModel.from_urdf(URDF_PATH, "left"),
            "right": ArmModel.from_urdf(URDF_PATH, "right"),
        },
    )
    save_arm_anchors(anchors, RUN_DIRECTORY / "arm_anchors.npz")
    latest_run_path = MY_RETARGETING_ROOT / "output" / "latest_run.txt"
    latest_run_path.write_text(str(RUN_DIRECTORY), encoding="utf-8")
    print(f"HumanPlus WiLoR keypoints saved: {RUN_DIRECTORY}")
    print(f"  frames: {len(left_keypoints.frame_indices)}")
    print(f"  left measured frames: {int(tracks.left.point_valid_mask.any(axis=1).sum())}")
    print(f"  right measured frames: {int(tracks.right.point_valid_mask.any(axis=1).sum())}")
    print(f"  left invalid frames after gate: {int(filled.left.invalid_mask.sum())}")
    print(f"  right invalid frames after gate: {int(filled.right.invalid_mask.sum())}")
    print(f"  gripper stages saved: {len(pose_snapshots)}")
    print(f"  left anchor present: {anchors.left.present} score={anchors.left.score:.3f}")
    print(f"  right anchor present: {anchors.right.present} score={anchors.right.score:.3f}")
    print("Render arms with test/render_arm_ik.py")


if __name__ == "__main__":
    main()
