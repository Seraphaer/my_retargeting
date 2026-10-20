"""Rigid transforms shared by the arm model and the base search.

Both packages need the same camera-frame pose convention. Quaternion order is
scipy xyzw. This module does not load a robot or solve IK.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def transform(rotation: np.ndarray, translation_m: np.ndarray) -> np.ndarray:
    """Build a 4x4 pose from a 3x3 rotation and a meter translation."""

    pose = np.eye(4)
    pose[:3, :3] = np.asarray(rotation, dtype=float)
    pose[:3, 3] = np.asarray(translation_m, dtype=float)
    return pose


def invert_transform(pose: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform."""

    inverse = np.eye(4)
    rotation = np.asarray(pose, dtype=float)[:3, :3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ np.asarray(pose, dtype=float)[:3, 3]
    return inverse


def pose_from_xyz_rpy(xyz_m: np.ndarray, rpy_rad: np.ndarray) -> np.ndarray:
    """URDF origin: fixed-axis roll, then pitch, then yaw, in radians."""

    rotation = Rotation.from_euler("XYZ", np.asarray(rpy_rad, dtype=float)).as_matrix()
    return transform(rotation, xyz_m)


def pose_from_translation_quaternion(translation_m: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Camera-frame pose. Quaternion is scipy xyzw."""

    rotation = Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=float)).as_matrix()
    return transform(rotation, translation_m)


def quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation to a scipy xyzw quaternion."""

    return Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_quat()


def orientation_error_rad(current: np.ndarray, target: np.ndarray) -> float:
    """Geodesic angle, in radians, between two 3x3 rotations."""

    difference = np.asarray(current, dtype=float).T @ np.asarray(target, dtype=float)
    cosine = float(np.clip((np.trace(difference) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def rotation_about_axis(axis_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    """Right-handed rotation about a unit axis by ``angle_rad``."""

    axis = np.asarray(axis_xyz, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * float(angle_rad)).as_matrix()
