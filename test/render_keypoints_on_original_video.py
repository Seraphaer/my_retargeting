"""Project triangulated rig-meter hand tracks onto the original HumanPlus fish-eye videos.

This script only reads ``capture.json`` and ``hand_tracks.npz``. It does not
rectify video, run WiLoR, triangulate, or fill missing points. Projection uses
the raw fish-eye calibration, not the rectified pinhole intrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from contracts import (
    KEYPOINT_COUNT,
    CameraModel,
    RepairedHands,
    load_rectified_stereo,
    load_repaired_hands,
    load_triangulated_hands,
)


# =============================================================================
# CONFIG: edit these values before running this script.
# =============================================================================
MY_RETARGETING_ROOT = Path(__file__).resolve().parents[1]
# None uses output/latest_run.txt written by run_humanplus_wilor.py.
RUN_DIRECTORY: Path | None = None
OUTPUT_SCALE = 0.5
OUTPUT_FPS = None
VIDEO_CODEC = "mp4v"

_LEFT_COLOR = (0, 165, 255)
_RIGHT_COLOR = (255, 144, 30)
_INTERPOLATED_COLOR = (0, 255, 255)
_HELD_COLOR = (255, 0, 255)
_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass(frozen=True)
class RenderConfig:
    """Display settings for the side-by-side original-video overlay.

    Args:
        output_path: MP4 written by this script. It is not a geometric result.
        output_scale: Uniform scale applied after drawing. ``1.0`` keeps the raw resolution.
        output_fps: Playback rate. ``None`` keeps the left source video's FPS.
        video_codec: Four-character OpenCV codec.
    """

    output_path: Path
    output_scale: float = 0.5
    output_fps: float | None = None
    video_codec: str = "mp4v"

    def __post_init__(self) -> None:
        if self.output_scale <= 0.0:
            raise ValueError("output_scale must be positive.")
        if self.output_fps is not None and self.output_fps <= 0.0:
            raise ValueError("output_fps must be positive when set.")
        if len(self.video_codec) != 4:
            raise ValueError("video_codec must contain four characters.")


def render_tracks_on_original_video(run_directory: Path, config: RenderConfig) -> Path:
    """Draw repaired 3D tracks on the two original fish-eye videos.

    Args:
        run_directory: Directory produced by ``run_humanplus_wilor.py``.
        config: Output path and display scale.
    Returns:
        Path of the side-by-side MP4.
    Raises:
        ValueError: If the tracks are not in the rig frame or the raw cameras are not fish-eye.
        FileNotFoundError: If a source video or saved artifact is missing.
    """

    cv2 = _require_cv2()
    capture = load_rectified_stereo(run_directory)
    tracks = _load_drawable_tracks(run_directory)
    _validate_fisheye(capture.left_raw_fisheye)
    _validate_fisheye(capture.right_raw_fisheye)
    frame_count = len(tracks.left.raw.frame_indices)
    left_decoder = _open_video(cv2, capture.left_raw_video)
    right_decoder = _open_video(cv2, capture.right_raw_video)
    writer = None
    try:
        source_fps = float(left_decoder.get(cv2.CAP_PROP_FPS))
        output_fps = config.output_fps if config.output_fps is not None else source_fps
        if output_fps <= 0.0:
            raise ValueError("The source video has no positive FPS; set OUTPUT_FPS.")
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        source_indices = np.asarray(tracks.left.raw.frame_indices, dtype=np.int64)
        for frame_index, source_index in enumerate(source_indices):
            if int(source_index) != frame_index:
                left_decoder.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
                right_decoder.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
            left_ok, left_frame = left_decoder.read()
            right_ok, right_frame = right_decoder.read()
            if not left_ok or left_frame is None or not right_ok or right_frame is None:
                raise RuntimeError(f"Original video ended before track frame {frame_index}.")
            left_canvas = _draw_frame(cv2, left_frame, capture.left_raw_fisheye, tracks, frame_index, "左目原始鱼眼")
            right_canvas = _draw_frame(cv2, right_frame, capture.right_raw_fisheye, tracks, frame_index, "右目原始鱼眼")
            combined = np.hstack((left_canvas, right_canvas))
            rendered = _resize(cv2, combined, config.output_scale)
            if writer is None:
                height_px, width_px = rendered.shape[:2]
                writer = cv2.VideoWriter(
                    str(config.output_path),
                    cv2.VideoWriter_fourcc(*config.video_codec),
                    output_fps,
                    (width_px, height_px),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open overlay writer: {config.output_path}")
            writer.write(rendered)
    finally:
        left_decoder.release()
        right_decoder.release()
        if writer is not None:
            writer.release()
    return config.output_path


def _draw_frame(cv2, frame_bgr, camera: CameraModel, tracks: RepairedHands, frame_index: int, label: str):
    canvas = frame_bgr.copy()
    summaries = []
    for track, base_color in ((tracks.left, _LEFT_COLOR), (tracks.right, _RIGHT_COLOR)):
        projected = _project_fisheye(cv2, track.points_m[frame_index], camera)
        colors = _colors(track, frame_index, base_color)
        _draw_hand(cv2, canvas, projected, colors)
        side = "左" if track.raw.side == "left" else "右"
        summaries.append(
            f"{side}:测{int(track.measurement_valid_mask[frame_index].sum())}"
            f" 插{int(track.interpolated_mask[frame_index].sum())}"
            f" 持{int(track.held_mask[frame_index].sum())}"
        )
    cv2.putText(canvas, f"{label}  frame={frame_index}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "  ".join(summaries), (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "橙/蓝=直接测量  黄=短插值  紫=长保持",
        (8, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _colors(track, frame_index: int, base_color: tuple[int, int, int]):
    colors = []
    for joint in range(KEYPOINT_COUNT):
        if track.held_mask[frame_index, joint]:
            colors.append(_HELD_COLOR)
        elif track.interpolated_mask[frame_index, joint]:
            colors.append(_INTERPOLATED_COLOR)
        elif track.measurement_valid_mask[frame_index, joint]:
            colors.append(base_color)
        else:
            colors.append(None)
    return colors


def _project_fisheye(cv2, points_m: np.ndarray, camera: CameraModel) -> np.ndarray:
    projected = np.full((KEYPOINT_COUNT, 2), np.nan, dtype=np.float64)
    finite = np.isfinite(points_m).all(axis=1)
    if not np.any(finite):
        return projected
    transform = np.linalg.inv(np.asarray(camera.T_rig_from_camera, dtype=np.float64))
    homogeneous = np.column_stack((points_m[finite], np.ones(int(finite.sum()), dtype=np.float64)))
    camera_points = (transform @ homogeneous.T).T[:, :3]
    in_front = camera_points[:, 2] > 1e-6
    if not np.any(in_front):
        return projected
    image_points, _ = cv2.fisheye.projectPoints(
        np.ascontiguousarray(camera_points[in_front].reshape(-1, 1, 3)),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.asarray(camera.intrinsics, dtype=np.float64),
        np.asarray(camera.distortion, dtype=np.float64).reshape(4, 1),
    )
    selected = np.flatnonzero(finite)[in_front]
    projected[selected] = image_points.reshape(-1, 2)
    return projected


def _draw_hand(cv2, canvas, projected: np.ndarray, colors) -> None:
    for start, end in _CONNECTIONS:
        if colors[start] is None or colors[end] is None:
            continue
        if _drawable(projected[start]) and _drawable(projected[end]):
            cv2.line(canvas, _pixel(projected[start]), _pixel(projected[end]), colors[start], 2, cv2.LINE_AA)
    for joint, point in enumerate(projected):
        if colors[joint] is not None and _drawable(point):
            cv2.circle(canvas, _pixel(point), 4, colors[joint], -1, cv2.LINE_AA)


def _load_drawable_tracks(run_directory: Path) -> RepairedHands:
    repaired_path = run_directory / "repaired_hands.npz"
    if repaired_path.is_file():
        tracks = load_repaired_hands(repaired_path)
    else:
        raw = load_triangulated_hands(run_directory / "hand_tracks.npz")
        tracks = _repaired_from_raw_only(raw)
    for track in (tracks.left, tracks.right):
        if track.raw.coordinate_frame != "rig":
            raise ValueError(f"{track.raw.side} track is not in the rig frame.")
    return tracks


def _repaired_from_raw_only(raw) -> RepairedHands:
    """Draw an older triangulation file that has no repaired companion."""

    from contracts import RepairedHandTrack

    def wrap(track):
        finite = np.isfinite(track.points_m).all(axis=2)
        measurement = track.point_valid_mask & finite
        empty = np.zeros(measurement.shape, dtype=bool)
        return RepairedHandTrack(
            raw=track,
            points_m=track.points_m,
            measurement_valid_mask=measurement,
            jump_mask=empty,
            interpolated_mask=empty,
            held_mask=empty,
        )

    return RepairedHands(left=wrap(raw.left), right=wrap(raw.right))


def _validate_fisheye(camera: CameraModel) -> None:
    if "fisheye" not in camera.camera_model.lower():
        raise ValueError(f"Original-video rendering requires a fish-eye camera, got {camera.camera_model!r}.")
    if len(camera.distortion) != 4:
        raise ValueError(f"{camera.camera_id} needs four fish-eye distortion coefficients.")


def _open_video(cv2, video_path: Path):
    if not video_path.is_file():
        raise FileNotFoundError(f"Original video does not exist: {video_path}")
    decoder = cv2.VideoCapture(str(video_path))
    if not decoder.isOpened():
        decoder.release()
        raise FileNotFoundError(f"Cannot open original video: {video_path}")
    return decoder


def _drawable(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all())


def _pixel(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _resize(cv2, image, scale: float):
    if scale == 1.0:
        return image
    height_px, width_px = image.shape[:2]
    size = (max(1, int(round(width_px * scale))), max(1, int(round(height_px * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to render the original-video overlay.") from error
    return cv2


def _resolve_run_directory() -> Path:
    if RUN_DIRECTORY is not None:
        return RUN_DIRECTORY
    pointer = MY_RETARGETING_ROOT / "output" / "latest_run.txt"
    if not pointer.is_file():
        raise FileNotFoundError("Set RUN_DIRECTORY, or run run_humanplus_wilor.py first so latest_run.txt exists.")
    return Path(pointer.read_text(encoding="utf-8").strip())


def main() -> None:
    """Render the configured run's triangulated tracks onto the original fish-eye videos."""

    raise RuntimeError("Per-point repair rendering was removed. Use test/render_keypoint_stages.py.")
    run_directory = _resolve_run_directory()
    output_path = render_tracks_on_original_video(
        run_directory,
        RenderConfig(
            output_path=run_directory / "original_fisheye_keypoints.mp4",
            output_scale=OUTPUT_SCALE,
            output_fps=OUTPUT_FPS,
            video_codec=VIDEO_CODEC,
        ),
    )
    print(f"Original-video overlay written: {output_path}")


if __name__ == "__main__":
    main()
