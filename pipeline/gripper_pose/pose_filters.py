"""Sequence filters used to build and post-process a gripper pose.

These are local rewrites of the EMA, Hampel, One-Euro, Savitzky-Golay,
velocity-outlier, and Gaussian-SLERP steps. They do not import the SSH tree.
Each function returns a new array.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def normalize_rows(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Divide each trailing vector by its length, with a floor of ``epsilon``."""

    length = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(length, epsilon)


def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    """Fixed-alpha EMA. ``out[0] = values[0]``; later frames blend toward the new sample."""

    output = np.empty_like(values, dtype=float)
    output[0] = values[0]
    for frame_index in range(1, len(values)):
        output[frame_index] = output[frame_index - 1] + alpha * (values[frame_index] - output[frame_index - 1])
    return output


def hampel_filter(values: np.ndarray, window: int = 5, n_sigma: float = 3.0) -> np.ndarray:
    """Replace a sample with the local median when it exceeds ``n_sigma`` robust deviations."""

    source = np.asarray(values, dtype=np.float64).copy()
    frame_count = len(source)
    flat = source.reshape(frame_count, -1)
    for column in range(flat.shape[1]):
        series = flat[:, column]
        for frame_index in range(frame_count):
            low = max(0, frame_index - window)
            high = min(frame_count, frame_index + window + 1)
            neighbors = np.concatenate([series[low:frame_index], series[frame_index + 1:high]])
            if len(neighbors) < 2:
                continue
            median = float(np.median(neighbors))
            deviation = float(np.median(np.abs(neighbors - median)) * 1.4826)
            if deviation < 1e-6:
                continue
            if abs(float(series[frame_index]) - median) > n_sigma * deviation:
                series[frame_index] = median
    return flat.reshape(source.shape)


def one_euro_filter(values: np.ndarray, frames_per_second: float, minimum_cutoff_hz: float = 1.0, beta: float = 15.0, derivative_cutoff_hz: float = 1.0) -> np.ndarray:
    """Causal One-Euro, one independent filter per column, at a fixed frame rate."""

    source = np.asarray(values, dtype=float)
    if source.ndim == 1:
        return _filter_one_euro_column(source, frames_per_second, minimum_cutoff_hz, beta, derivative_cutoff_hz)
    output = np.empty_like(source, dtype=float)
    for column in range(source.shape[1]):
        output[:, column] = _filter_one_euro_column(
            source[:, column], frames_per_second, minimum_cutoff_hz, beta, derivative_cutoff_hz
        )
    return output


def savgol_smooth(values: np.ndarray, window: int = 21, polynomial_order: int = 3) -> np.ndarray:
    """Non-causal Savitzky-Golay smoothing. Short sequences are returned unchanged."""

    source = np.asarray(values, dtype=float)
    frame_count = len(source)
    if frame_count < 3:
        return source.copy()
    used_window = min(window, frame_count)
    if used_window % 2 == 0:
        used_window -= 1
    if used_window < 3:
        return source.copy()
    used_order = min(polynomial_order, used_window - 1)
    return savgol_filter(source, used_window, used_order, axis=0)


def velocity_outlier_filter(
    values: np.ndarray, frames_per_second: float, median_multiplier: float = 5.0, floor_per_second: float = 0.9, rounds: int = 2
) -> np.ndarray:
    """Replace frames whose step exceeds a whole-clip median threshold. Position uses meters."""

    source = np.asarray(values, dtype=float)
    output = source.copy()
    if len(output) < 3:
        return output
    for _ in range(rounds):
        differences = np.diff(output, axis=0)
        step = np.abs(differences) if output.ndim == 1 else np.linalg.norm(differences, axis=-1)
        threshold = max(median_multiplier * float(np.median(step)), floor_per_second / frames_per_second)
        outlier = np.zeros(len(output), dtype=bool)
        outlier[1:] = step > threshold
        if not outlier.any():
            break
        output = _interpolate_samples(output, outlier)
    return output


def orientation_velocity_outlier_filter(
    quaternions: np.ndarray,
    frames_per_second: float,
    median_multiplier: float = 5.0,
    floor_radians_per_second: float = 10.0,
    rounds: int = 2,
) -> np.ndarray:
    """Replace orientation frames whose angle step exceeds a whole-clip median threshold."""

    output = np.asarray(quaternions, dtype=float).copy()
    if len(output) < 3:
        return output
    for _ in range(rounds):
        dots = np.clip(np.abs(np.sum(output[1:] * output[:-1], axis=-1)), -1.0, 1.0)
        step = 2.0 * np.arccos(dots)
        threshold = max(median_multiplier * float(np.median(step)), floor_radians_per_second / frames_per_second)
        outlier = np.zeros(len(output), dtype=bool)
        outlier[1:] = step > threshold
        if not outlier.any():
            break
        output = _slerp_samples(output, outlier)
    return output


def gaussian_slerp_smooth(quaternions: np.ndarray, sigma_frames: float = 10.0, kernel: int = 21) -> np.ndarray:
    """Non-causal Gaussian-weighted SLERP. Sequences shorter than two frames are unchanged."""

    source = np.asarray(quaternions, dtype=float)
    frame_count = len(source)
    if frame_count < 2:
        return source.copy()
    half = kernel // 2
    offsets = np.arange(-half, half + 1)
    weights = np.exp(-0.5 * (offsets / sigma_frames) ** 2)
    weights = weights / weights.sum()
    corrected = source.copy()
    for frame_index in range(1, frame_count):
        if np.dot(source[frame_index], corrected[frame_index - 1]) < 0.0:
            corrected[frame_index] = -source[frame_index]
    output = np.empty_like(source)
    for frame_index in range(frame_count):
        indices = np.clip(frame_index + offsets, 0, frame_count - 1)
        accumulator = corrected[indices[0]].copy()
        accumulated_weight = float(weights[0])
        for tap in range(1, len(indices)):
            sample = corrected[indices[tap]]
            if np.dot(accumulator, sample) < 0.0:
                sample = -sample
            accumulated_weight += float(weights[tap])
            fraction = float(weights[tap]) / accumulated_weight
            blend = Slerp([0.0, 1.0], Rotation.from_quat([accumulator, sample]))
            accumulator = blend([fraction]).as_quat()[0]
            accumulator = accumulator / np.linalg.norm(accumulator)
        output[frame_index] = accumulator
    return output


def _filter_one_euro_column(
    values: np.ndarray, frames_per_second: float, minimum_cutoff_hz: float, beta: float, derivative_cutoff_hz: float
) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    previous = float(values[0])
    previous_derivative = 0.0
    output[0] = previous
    for frame_index in range(1, len(values)):
        derivative = (float(values[frame_index]) - previous) * frames_per_second
        derivative_alpha = _one_euro_alpha(derivative_cutoff_hz, frames_per_second)
        derivative_hat = derivative_alpha * derivative + (1.0 - derivative_alpha) * previous_derivative
        cutoff = minimum_cutoff_hz + beta * abs(derivative_hat)
        alpha = _one_euro_alpha(cutoff, frames_per_second)
        previous = alpha * float(values[frame_index]) + (1.0 - alpha) * previous
        previous_derivative = derivative_hat
        output[frame_index] = previous
    return output


def _one_euro_alpha(cutoff_hz: float, frames_per_second: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return 1.0 / (1.0 + tau * frames_per_second)


def _interpolate_samples(values: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    valid = ~invalid
    index = np.arange(len(values))
    if valid.sum() == 0:
        return values.copy()
    if values.ndim == 1:
        return np.interp(index, index[valid], values[valid])
    output = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        output[:, column] = np.interp(index, index[valid], values[valid, column])
    return output


def _slerp_samples(quaternions: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    output = quaternions.copy()
    frame_count = len(quaternions)
    frame_index = 0
    while frame_index < frame_count:
        if not invalid[frame_index]:
            frame_index += 1
            continue
        start = frame_index
        while frame_index < frame_count and invalid[frame_index]:
            frame_index += 1
        before = start - 1 if start > 0 else None
        after = frame_index if frame_index < frame_count else None
        if before is None and after is None:
            continue
        if before is None:
            output[start:frame_index] = quaternions[after]
            continue
        if after is None:
            output[start:frame_index] = quaternions[before]
            continue
        start_quaternion = quaternions[before]
        end_quaternion = quaternions[after]
        if np.dot(start_quaternion, end_quaternion) < 0.0:
            end_quaternion = -end_quaternion
        blend = Slerp([0.0, 1.0], Rotation.from_quat([start_quaternion, end_quaternion]))
        span = after - before
        for sample_index in range(start, frame_index):
            output[sample_index] = blend([(sample_index - before) / span]).as_quat()[0]
    return output
