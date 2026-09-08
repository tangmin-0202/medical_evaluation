from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def _identity() -> np.ndarray:
    return np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


@dataclass(frozen=True)
class RegistrationLimits:
    min_matches: int = 10
    min_inliers: int = 6
    min_inlier_ratio: float = 0.35
    min_mask_iou: float = 0.20
    min_scale: float = 0.60
    max_scale: float = 1.60
    max_residual_px: float = 5.0
    search_dilation_diagonal_ratio: float = 0.08
    analysis_max_width: int = 960


@dataclass(frozen=True)
class RegistrationResult:
    accepted: bool
    reason: str | None
    matrix: np.ndarray = field(default_factory=_identity)
    match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    mask_iou: float = 0.0
    residual_px: float = float("inf")
    scale: float = 1.0
    angle_deg: float = 0.0


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (2, 3):
        raise ValueError("similarity matrix must have shape (2, 3)")
    return np.vstack((value, np.asarray([0.0, 0.0, 1.0])))


def compose_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (_homogeneous(left) @ _homogeneous(right))[:2]


def invert_similarity(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_homogeneous(matrix))[:2]


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    homogeneous = np.column_stack((values, np.ones(len(values), dtype=np.float64)))
    return homogeneous @ np.asarray(matrix, dtype=np.float64).T


def warp_mask(
    mask: np.ndarray,
    matrix: np.ndarray,
    *,
    output_shape: tuple[int, int],
) -> np.ndarray:
    height, width = output_shape
    if height <= 0 or width <= 0:
        raise ValueError("output_shape must be positive")
    return cv2.warpAffine(
        np.asarray(mask, dtype=np.uint8),
        np.asarray(matrix, dtype=np.float64),
        (width, height),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.count_nonzero(left & right)
    union = np.count_nonzero(left | right)
    return float(intersection / union) if union else 0.0


def _scale_angle(matrix: np.ndarray) -> tuple[float, float]:
    a, _b, _tx = matrix[0]
    c, _d, _ty = matrix[1]
    return float(np.hypot(a, c)), float(np.rad2deg(np.arctan2(c, a)))


def estimate_similarity_registration(
    reference_bgr: np.ndarray,
    reference_mask: np.ndarray,
    target_bgr: np.ndarray,
    target_mask: np.ndarray,
    limits: RegistrationLimits | None = None,
) -> RegistrationResult:
    limits = limits or RegistrationLimits()
    reference_full = np.asarray(reference_bgr)
    target_full = np.asarray(target_bgr)
    ref_mask_full = np.asarray(reference_mask, dtype=bool)
    dst_mask_full = np.asarray(target_mask, dtype=bool)
    if reference_full.shape != target_full.shape or reference_full.ndim != 3:
        raise ValueError("reference and target images must share HxWxC shape")
    if (
        ref_mask_full.shape != reference_full.shape[:2]
        or dst_mask_full.shape != target_full.shape[:2]
    ):
        raise ValueError("masks must match image height and width")

    if not 0 <= limits.search_dilation_diagonal_ratio <= 0.5:
        raise ValueError("search_dilation_diagonal_ratio must be within [0, 0.5]")
    if limits.analysis_max_width <= 0:
        raise ValueError("analysis_max_width must be positive")
    analysis_scale = min(1.0, limits.analysis_max_width / reference_full.shape[1])
    if analysis_scale < 1.0:
        analysis_size = (
            limits.analysis_max_width,
            max(1, round(reference_full.shape[0] * analysis_scale)),
        )
        reference = cv2.resize(reference_full, analysis_size, interpolation=cv2.INTER_AREA)
        target = cv2.resize(target_full, analysis_size, interpolation=cv2.INTER_AREA)
        ref_mask = cv2.resize(
            ref_mask_full.astype(np.uint8), analysis_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        dst_mask = cv2.resize(
            dst_mask_full.astype(np.uint8), analysis_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    else:
        reference, target = reference_full, target_full
        ref_mask, dst_mask = ref_mask_full, dst_mask_full
    detector = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.01)
    radius = max(
        1,
        round(np.hypot(*reference.shape[:2]) * limits.search_dilation_diagonal_ratio),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    ref_search = cv2.dilate(ref_mask.astype(np.uint8) * 255, kernel)
    dst_search = cv2.dilate(dst_mask.astype(np.uint8) * 255, kernel)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    ref_keypoints, ref_descriptors = detector.detectAndCompute(ref_gray, ref_search)
    dst_keypoints, dst_descriptors = detector.detectAndCompute(dst_gray, dst_search)
    if ref_descriptors is None or dst_descriptors is None:
        return RegistrationResult(False, "insufficient_feature_matches")

    candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        ref_descriptors, dst_descriptors, k=2
    )
    matches = [
        first
        for pair in candidates
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.78 * second.distance
    ]
    if len(matches) < limits.min_matches:
        return RegistrationResult(
            False,
            "insufficient_feature_matches",
            match_count=len(matches),
        )

    source = np.float32([ref_keypoints[item.queryIdx].pt for item in matches])
    destination = np.float32([dst_keypoints[item.trainIdx].pt for item in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.995,
        refineIters=20,
    )
    if matrix is None or inliers is None:
        return RegistrationResult(
            False,
            "transform_estimation_failed",
            match_count=len(matches),
        )

    matrix = np.asarray(matrix, dtype=np.float64)
    inlier_flags = inliers.ravel().astype(bool)
    inlier_count = int(inlier_flags.sum())
    inlier_ratio = float(inlier_count / len(matches))
    scale, angle_deg = _scale_angle(matrix)
    projected = transform_points(source, matrix)
    residual_px = (
        float(np.median(np.linalg.norm(projected[inlier_flags] - destination[inlier_flags], axis=1)))
        if inlier_count
        else float("inf")
    )
    if analysis_scale < 1.0:
        matrix[:, 2] /= analysis_scale
        residual_px /= analysis_scale
    projected_mask = warp_mask(
        ref_mask_full, matrix, output_shape=dst_mask_full.shape
    )
    mask_iou = _mask_iou(projected_mask, dst_mask_full)
    metrics = {
        "matrix": matrix,
        "match_count": len(matches),
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "mask_iou": mask_iou,
        "residual_px": residual_px,
        "scale": scale,
        "angle_deg": angle_deg,
    }
    if inlier_count < limits.min_inliers:
        return RegistrationResult(False, "insufficient_inliers", **metrics)
    if inlier_ratio < limits.min_inlier_ratio:
        return RegistrationResult(False, "low_inlier_ratio", **metrics)
    if not limits.min_scale <= scale <= limits.max_scale:
        return RegistrationResult(False, "scale_out_of_range", **metrics)
    if residual_px > limits.max_residual_px:
        return RegistrationResult(False, "high_reprojection_residual", **metrics)
    if mask_iou < limits.min_mask_iou:
        return RegistrationResult(False, "low_mask_overlap", **metrics)
    return RegistrationResult(True, None, **metrics)


def registration_is_continuous(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    image_shape: tuple[int, int],
    max_angle_delta_deg: float,
    max_scale_delta: float,
    max_translation_diagonal_ratio: float,
) -> bool:
    relative = compose_similarity(current, invert_similarity(previous))
    scale, angle = _scale_angle(relative)
    angle_delta = abs((angle + 180.0) % 360.0 - 180.0)
    height, width = image_shape
    center = np.asarray([[width / 2.0, height / 2.0]])
    moved_center = transform_points(center, relative)[0]
    translation_ratio = float(
        np.linalg.norm(moved_center - center[0]) / np.hypot(width, height)
    )
    return bool(
        angle_delta <= max_angle_delta_deg
        and abs(scale - 1.0) <= max_scale_delta
        and translation_ratio <= max_translation_diagonal_ratio
    )
