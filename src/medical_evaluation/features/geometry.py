from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from medical_evaluation.storage import safe_child


def centroid_xy(mask: np.ndarray) -> tuple[float, float] | None:
    binary = _mask(mask)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def centroid_normalized(mask: np.ndarray) -> tuple[float, float] | None:
    binary = _mask(mask)
    centroid = centroid_xy(binary)
    if centroid is None:
        return None
    height, width = binary.shape
    x_scale = max(width - 1, 1)
    y_scale = max(height - 1, 1)
    return centroid[0] / x_scale, centroid[1] / y_scale


def bounding_box_center(mask: np.ndarray) -> tuple[float, float] | None:
    box = _bounding_box(_mask(mask))
    if box is None:
        return None
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _bounding_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def relative_bbox_center_offset(
    subject: np.ndarray,
    reference: np.ndarray,
) -> float | None:
    subject_mask, reference_mask = _matching_masks(subject, reference)
    subject_box = _bounding_box(subject_mask)
    reference_box = _bounding_box(reference_mask)
    if subject_box is None or reference_box is None:
        return None
    sx = (subject_box[0] + subject_box[2]) / 2
    sy = (subject_box[1] + subject_box[3]) / 2
    rx = (reference_box[0] + reference_box[2]) / 2
    ry = (reference_box[1] + reference_box[3]) / 2
    reference_width = reference_box[2] - reference_box[0] + 1
    reference_height = reference_box[3] - reference_box[1] + 1
    return float(np.hypot((sx - rx) / reference_width, (sy - ry) / reference_height))


def frame_center_offset(mask: np.ndarray) -> float | None:
    binary = _mask(mask)
    center = bounding_box_center(binary)
    if center is None:
        return None
    height, width = binary.shape
    frame_x = (width - 1) / 2
    frame_y = (height - 1) / 2
    return float(np.hypot((center[0] - frame_x) / width, (center[1] - frame_y) / height))


def area_ratio(mask: np.ndarray) -> float | None:
    binary = _mask(mask)
    if binary.size == 0:
        return None
    return float(binary.mean())


def intersection_ratio(subject: np.ndarray, reference: np.ndarray) -> float | None:
    subject_mask, reference_mask = _matching_masks(subject, reference)
    subject_area = int(subject_mask.sum())
    if subject_area == 0:
        return None
    return float(np.logical_and(subject_mask, reference_mask).sum() / subject_area)


def visible_ratio(mask: np.ndarray, expected_area_pixels: float) -> float | None:
    if expected_area_pixels <= 0:
        raise ValueError("expected_area_pixels must be positive")
    binary = _mask(mask)
    if not binary.any():
        return None
    return float(min(binary.sum() / expected_area_pixels, 1.0))


def mask_iou(first: np.ndarray, second: np.ndarray) -> float | None:
    first_mask, second_mask = _matching_masks(first, second)
    union = np.logical_or(first_mask, second_mask).sum()
    if union == 0:
        return None
    return float(np.logical_and(first_mask, second_mask).sum() / union)


def minimum_boundary_distance(first: np.ndarray, second: np.ndarray) -> float | None:
    first_mask, second_mask = _matching_masks(first, second)
    if not first_mask.any() or not second_mask.any():
        return None
    distances = distance_transform_edt(~second_mask)
    return float(distances[first_mask].min())


def boundary_contact_ratio(
    subject: np.ndarray,
    reference: np.ndarray,
    *,
    max_distance: float = 1.5,
) -> float | None:
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")
    subject_mask, reference_mask = _matching_masks(subject, reference)
    area = int(subject_mask.sum())
    if area == 0 or not reference_mask.any():
        return None
    distances = distance_transform_edt(~reference_mask)
    return float((distances[subject_mask] <= max_distance).mean())


def write_overlay(
    frame_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    evidence_root: Path,
    relative_output: str,
    *,
    opacity: float = 0.42,
) -> Path:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    if not 0 <= opacity <= 1:
        raise ValueError("opacity must be between zero and one")
    output_path = safe_child(evidence_root, relative_output)
    canvas = frame_bgr.copy()
    for object_id, raw_mask in masks.items():
        mask = _mask(raw_mask)
        if mask.shape != frame_bgr.shape[:2]:
            raise ValueError("mask and frame dimensions must match")
        if not mask.any():
            continue
        color = _stable_color(object_id)
        layer = np.empty_like(canvas)
        layer[:] = color
        canvas[mask] = cv2.addWeighted(canvas, 1 - opacity, layer, opacity, 0)[mask]
        centroid = centroid_xy(mask)
        if centroid is not None:
            cv2.putText(
                canvas,
                object_id,
                (round(centroid[0]), round(centroid[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"could not write evidence overlay: {output_path}")
    return output_path


def _mask(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    return array.astype(bool, copy=False)


def _matching_masks(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first_mask = _mask(first)
    second_mask = _mask(second)
    if first_mask.shape != second_mask.shape:
        raise ValueError("mask dimensions must match")
    return first_mask, second_mask


def _stable_color(object_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(object_id.encode("utf-8")).digest()
    return tuple(64 + int(value) % 192 for value in digest[:3])
