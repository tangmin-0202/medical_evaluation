from __future__ import annotations

import cv2
import numpy as np


def color_ratio_hsv(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    hue_range: tuple[int, int],
    minimum_saturation: int = 40,
    minimum_value: int = 30,
) -> float | None:
    binary = _region(frame_bgr, mask)
    if not binary.any():
        return None
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    selected = hsv[binary]
    lower_hue, upper_hue = hue_range
    matches = (
        (selected[:, 0] >= lower_hue)
        & (selected[:, 0] <= upper_hue)
        & (selected[:, 1] >= minimum_saturation)
        & (selected[:, 2] >= minimum_value)
    )
    return float(matches.mean())


def dark_ratio(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    maximum_value: int = 55,
) -> float | None:
    binary = _region(frame_bgr, mask)
    if not binary.any():
        return None
    value = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    return float((value[binary] <= maximum_value).mean())


def blur_score(frame_bgr: np.ndarray, mask: np.ndarray) -> float | None:
    binary = _region(frame_bgr, mask)
    if not binary.any():
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian[binary].var())


def reference_cosine_similarity(
    embedding: np.ndarray,
    reference: np.ndarray,
) -> float | None:
    first = np.asarray(embedding, dtype=np.float32).ravel()
    second = np.asarray(reference, dtype=np.float32).ravel()
    if first.shape != second.shape:
        raise ValueError("embedding dimensions must match")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return None
    return float(np.clip(np.dot(first, second) / denominator, -1, 1))


def _region(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != frame_bgr.shape[:2]:
        raise ValueError("mask and frame dimensions must match")
    return binary
