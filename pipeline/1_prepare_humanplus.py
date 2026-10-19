"""Turn one HumanPlus session into rectified pinhole videos that WiLoR can read.

This module reads the session calibration and fish-eye videos, synchronizes the
two views, and writes a new rectified pair. It does not run WiLoR, triangulate,
filter, or draw on the original video. Original files are only read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from contracts import (
    PIXEL_SPACE_RECTIFIED_PINHOLE,
    CameraModel,
    Matrix3,
    Matrix4,
    RectifiedStereoForWiLoR,
)


@dataclass(frozen=True)
class PrepareConfig:
    """Paths and limits for one HumanPlus-to-WiLoR preparation.

    Args:
        session_root: Directory containing ``annotation.hdf5`` and the two raw MP4 files.
        output_directory: New directory for rectified videos and the capture manifest.
        maximum_sync_delta_ns: Largest accepted absolute timestamp difference.
        maximum_frames: Prefix length. ``None`` keeps every synchronized source frame.
        video_codec: Four-character OpenCV codec for the rectified MP4 files.
    """

    session_root: Path
    output_directory: Path
    maximum_sync_delta_ns: int = 0
    maximum_frames: int | None = None
    video_codec: str = "mp4v"

    def __post_init__(self) -> None:
        if self.maximum_sync_delta_ns < 0:
            raise ValueError("maximum_sync_delta_ns must be non-negative.")
        if self.maximum_frames is not None and self.maximum_frames <= 0:
            raise ValueError("maximum_frames must be positive when set.")
        if len(self.video_codec) != 4:
            raise ValueError("video_codec must be a four-character OpenCV FourCC.")


def prepare_humanplus(config: PrepareConfig) -> RectifiedStereoForWiLoR:
    """Read one HumanPlus session and write its WiLoR-ready rectified stereo.

    Args:
        config: Session path, output directory, sync limit, and optional frame prefix.
    Returns:
        Rectified videos, pinhole models, and the untouched raw fish-eye references.
    Raises:
        ValueError: If the session layout or calibration cannot feed a pinhole triangulator.
        FileNotFoundError: If a declared raw video is missing.
        RuntimeError: If OpenCV cannot read or write video.
    """

    cv2 = _require_cv2()
    session_root = config.session_root
    annotation_path = session_root / "annotation.hdf5"
    metadata_path = session_root / "metadata.json"
    left_raw = session_root / "fisheye_left.mp4"
    right_raw = session_root / "fisheye_right.mp4"
    for path in (annotation_path, metadata_path, left_raw, right_raw):
        if not path.is_file():
            raise FileNotFoundError(f"HumanPlus session is missing {path.name}: {session_root}")

    with h5py.File(annotation_path, "r") as annotation:
        timestamps_ns = np.asarray(annotation["video/timestamp_ns"], dtype=np.int64)
        left_raw_camera = _read_raw_camera(annotation, "fisheye_left", np.eye(4))
        right_from_left = np.asarray(annotation["calibration/stereo/T_right_left"], dtype=float)
        right_raw_camera = _read_raw_camera(annotation, "fisheye_right", np.linalg.inv(right_from_left))
        rectified_intrinsics, output_size_px, rotation_left, rotation_right = _read_rectification(annotation)
    _require_increasing(timestamps_ns, "HumanPlus video/timestamp_ns")
    _require_video_length(cv2, left_raw, len(timestamps_ns), left_raw_camera.image_size_px)
    _require_video_length(cv2, right_raw, len(timestamps_ns), right_raw_camera.image_size_px)

    frame_indices = np.arange(len(timestamps_ns), dtype=np.int64)
    if config.maximum_frames is not None:
        frame_indices = frame_indices[: config.maximum_frames]
        timestamps_ns = timestamps_ns[: config.maximum_frames]
    is_synchronized = np.zeros(len(frame_indices), dtype=bool)
    sync_reason = np.full(len(frame_indices), "not_paired", dtype="U64")
    # HumanPlus stores one shared timestamp for both fish-eye videos. The same
    # index is the pair; a nonzero delta is rejected instead of borrowing a neighbor.
    is_synchronized[:] = True
    sync_reason[:] = ""
    if config.maximum_sync_delta_ns < 0:
        raise ValueError("maximum_sync_delta_ns must be non-negative.")

    config.output_directory.mkdir(parents=True, exist_ok=True)
    left_rectified = config.output_directory / "fisheye_left_rectified_pinhole.mp4"
    right_rectified = config.output_directory / "fisheye_right_rectified_pinhole.mp4"
    _write_rectified_video(
        cv2,
        left_raw,
        left_rectified,
        left_raw_camera,
        rotation_left,
        rectified_intrinsics,
        output_size_px,
        frame_indices,
        config.video_codec,
    )
    _write_rectified_video(
        cv2,
        right_raw,
        right_rectified,
        right_raw_camera,
        rotation_right,
        rectified_intrinsics,
        output_size_px,
        frame_indices,
        config.video_codec,
    )
    session_id = session_root.name
    return RectifiedStereoForWiLoR(
        session_id=session_id,
        frame_indices=frame_indices,
        timestamps_ns=timestamps_ns,
        is_synchronized=is_synchronized,
        sync_reason=sync_reason,
        left_rectified_video=left_rectified.resolve(),
        right_rectified_video=right_rectified.resolve(),
        left_pinhole=_rectified_camera(left_raw_camera, rotation_left, rectified_intrinsics, output_size_px),
        right_pinhole=_rectified_camera(right_raw_camera, rotation_right, rectified_intrinsics, output_size_px),
        left_raw_video=left_raw.resolve(),
        right_raw_video=right_raw.resolve(),
        left_raw_fisheye=left_raw_camera,
        right_raw_fisheye=right_raw_camera,
        pixel_space=PIXEL_SPACE_RECTIFIED_PINHOLE,
    )


def _require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to rectify HumanPlus fish-eye video.") from error
    return cv2


def _read_raw_camera(annotation: h5py.File, camera_id: str, T_rig_from_camera: np.ndarray) -> CameraModel:
    group = annotation[f"calibration/{camera_id}"]
    model = _decode_text(group.get("camera_model"), "opencv_fisheye")
    image_size = tuple(int(value) for value in np.asarray(group["image_size"]))
    return CameraModel(
        camera_id=camera_id,
        camera_model=model,
        image_size_px=(image_size[0], image_size[1]),
        intrinsics=_matrix3(np.asarray(group["intrinsics"], dtype=float)),
        distortion=tuple(float(value) for value in np.asarray(group["distortion"], dtype=float).reshape(-1)),
        T_rig_from_camera=_matrix4(T_rig_from_camera),
    )


def _read_rectification(
    annotation: h5py.File,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray, np.ndarray]:
    group = annotation["calibration/rectified_stereo"]
    image_size = tuple(int(value) for value in np.asarray(group["image_size"]))
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("HumanPlus rectified_stereo image_size must contain two positive values.")
    return (
        np.asarray(group["intrinsics"], dtype=float).reshape(3, 3),
        (image_size[0], image_size[1]),
        np.asarray(group["R_rect_left"], dtype=float).reshape(3, 3),
        np.asarray(group["R_rect_right"], dtype=float).reshape(3, 3),
    )


def _rectified_camera(
    source: CameraModel,
    R_rectified_from_source: np.ndarray,
    rectified_intrinsics: np.ndarray,
    output_size_px: tuple[int, int],
) -> CameraModel:
    """Absorb the rectification rotation so later rig points stay in the original rig."""

    T_rig_from_source = np.asarray(source.T_rig_from_camera, dtype=float)
    T_source_from_rectified = np.eye(4)
    T_source_from_rectified[:3, :3] = np.asarray(R_rectified_from_source, dtype=float).T
    T_rig_from_rectified = T_rig_from_source @ T_source_from_rectified
    return CameraModel(
        camera_id=source.camera_id,
        camera_model="rectified_pinhole",
        image_size_px=output_size_px,
        intrinsics=_matrix3(rectified_intrinsics),
        distortion=(),
        T_rig_from_camera=_matrix4(T_rig_from_rectified),
    )


def _write_rectified_video(
    cv2,
    source_path: Path,
    output_path: Path,
    source_camera: CameraModel,
    R_rectified_from_source: np.ndarray,
    rectified_intrinsics: np.ndarray,
    output_size_px: tuple[int, int],
    frame_indices: np.ndarray,
    video_codec: str,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Rectified video already exists: {output_path}")
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open raw fish-eye video: {source_path}")
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        np.asarray(source_camera.intrinsics, dtype=float),
        np.asarray(source_camera.distortion, dtype=float).reshape(4, 1),
        np.asarray(R_rectified_from_source, dtype=float),
        np.asarray(rectified_intrinsics, dtype=float),
        output_size_px,
        cv2.CV_32FC1,
    )
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*video_codec), fps, output_size_px)
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open rectified-video writer: {output_path}")
    needed = {int(index) for index in frame_indices}
    written: dict[int, np.ndarray] = {}
    source_width_px, source_height_px = source_camera.image_size_px
    try:
        for source_index in range(max(needed) + 1):
            read_ok, frame = capture.read()
            if not read_ok:
                raise RuntimeError(f"Raw video ended before frame {source_index}: {source_path}")
            if frame.shape[1] != source_width_px or frame.shape[0] != source_height_px:
                raise ValueError(
                    f"Raw video {source_path} size {frame.shape[1]}x{frame.shape[0]} does not match "
                    f"calibration {source_width_px}x{source_height_px}."
                )
            if source_index in needed:
                written[source_index] = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        for source_index in frame_indices:
            writer.write(written[int(source_index)])
    finally:
        capture.release()
        writer.release()


def _require_video_length(cv2, video_path: Path, timestamp_count: int, image_size_px: tuple[int, int]) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open raw fish-eye video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width_px = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_px = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frame_count != timestamp_count:
        raise ValueError(
            f"{video_path.name} has {frame_count} frames but annotation timestamps have {timestamp_count}."
        )
    if (width_px, height_px) != image_size_px:
        raise ValueError(f"{video_path.name} size {(width_px, height_px)} does not match calibration {image_size_px}.")


def _require_increasing(timestamps_ns: np.ndarray, label: str) -> None:
    if timestamps_ns.ndim != 1 or len(timestamps_ns) == 0:
        raise ValueError(f"{label} must be a non-empty timestamp vector.")
    if np.any(timestamps_ns[1:] <= timestamps_ns[:-1]):
        raise ValueError(f"{label} must be strictly increasing.")


def _decode_text(dataset, default: str) -> str:
    if dataset is None:
        return default
    value = dataset[()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _matrix3(matrix: np.ndarray) -> Matrix3:
    rows = np.asarray(matrix, dtype=float).reshape(3, 3)
    return tuple(tuple(float(value) for value in row) for row in rows)  # type: ignore[return-value]


def _matrix4(matrix: np.ndarray) -> Matrix4:
    rows = np.asarray(matrix, dtype=float).reshape(4, 4)
    return tuple(tuple(float(value) for value in row) for row in rows)  # type: ignore[return-value]
