"""Stable display-frame selection and reversible CP04 hand crops."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DisplayFrameCandidate:
    frame_index: int
    frame_time_sec: float
    frame_bgr: np.ndarray
    hand_mask: np.ndarray
    seed_mask: np.ndarray
    sharpness: float
    boundary_clipped: bool
    score: float


@dataclass(frozen=True)
class HandObjectCrop:
    image_bgr: np.ndarray
    support_mask: np.ndarray
    background_bgr: tuple[int, int, int]
    original_shape: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    resized_shape: tuple[int, int]
    padding: tuple[int, int]

    def project_mask(self, mask: np.ndarray) -> np.ndarray:
        """Map an original-frame mask into the square local-crop coordinates."""
        value = np.asarray(mask, dtype=bool)
        if value.shape != self.original_shape:
            raise ValueError("mask shape does not match the original frame")
        x1, y1, x2, y2 = self.crop_box
        resized_height, resized_width = self.resized_shape
        pad_x, pad_y = self.padding
        local = cv2.resize(
            value[y1:y2, x1:x2].astype(np.uint8),
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        output = np.zeros(self.image_bgr.shape[:2], dtype=bool)
        output[
            pad_y : pad_y + resized_height,
            pad_x : pad_x + resized_width,
        ] = local
        return output

    def restore_mask(self, local_mask: np.ndarray) -> np.ndarray:
        """Map a local SAM mask back into its exact original-frame rectangle."""
        value = np.asarray(local_mask, dtype=bool)
        if value.shape != self.image_bgr.shape[:2]:
            value = cv2.resize(
                value.astype(np.uint8),
                (self.image_bgr.shape[1], self.image_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        x1, y1, x2, y2 = self.crop_box
        resized_height, resized_width = self.resized_shape
        pad_x, pad_y = self.padding
        content = value[
            pad_y : pad_y + resized_height,
            pad_x : pad_x + resized_width,
        ]
        restored_content = cv2.resize(
            content.astype(np.uint8),
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        restored = np.zeros(self.original_shape, dtype=bool)
        restored[y1:y2, x1:x2] = restored_content
        return restored


def build_hand_object_crop(
    frame_bgr: np.ndarray,
    hand_mask: np.ndarray,
    seed_mask: np.ndarray,
    *,
    output_size: int = 640,
) -> HandObjectCrop:
    """Keep the displayed glove and its metal seed while deleting the scene."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 image")
    if output_size < 64:
        raise ValueError("output_size must be at least 64")
    shape = frame_bgr.shape[:2]
    hand = _largest_component(_as_mask(hand_mask, shape))
    seed = _as_mask(seed_mask, shape)
    if not hand.any():
        raise ValueError("hand_mask must contain a hand component")
    if not seed.any():
        raise ValueError("seed_mask must contain a metal-object seed")

    filled_hand = _fill_enclosed_holes(hand)
    radius = max(1, round(min(shape) * 0.01))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    support = cv2.dilate(
        np.logical_or(filled_hand, seed).astype(np.uint8), kernel
    ).astype(bool)

    ys, xs = np.nonzero(hand)
    hand_width = int(xs.max() - xs.min() + 1)
    hand_height = int(ys.max() - ys.min() + 1)
    padding = max(2, round(max(hand_width, hand_height) * 0.08))
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(shape[1], int(xs.max()) + padding + 1)
    y2 = min(shape[0], int(ys.max()) + padding + 1)

    background_bgr = (127, 127, 127)
    isolated = np.full_like(frame_bgr, background_bgr)
    isolated[support] = frame_bgr[support]
    region = isolated[y1:y2, x1:x2]
    scale = min(output_size / region.shape[1], output_size / region.shape[0])
    resized_width = max(1, round(region.shape[1] * scale))
    resized_height = max(1, round(region.shape[0] * scale))
    resized = cv2.resize(
        region, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    resized_support = cv2.resize(
        support[y1:y2, x1:x2].astype(np.uint8),
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    resized[~resized_support] = background_bgr
    pad_x = (output_size - resized_width) // 2
    pad_y = (output_size - resized_height) // 2
    canvas = np.full((output_size, output_size, 3), background_bgr, dtype=np.uint8)
    canvas[
        pad_y : pad_y + resized_height,
        pad_x : pad_x + resized_width,
    ] = resized
    return HandObjectCrop(
        image_bgr=canvas,
        support_mask=support,
        background_bgr=background_bgr,
        original_shape=shape,
        crop_box=(x1, y1, x2, y2),
        resized_shape=(resized_height, resized_width),
        padding=(pad_x, pad_y),
    )


def select_stable_display_frames(
    candidates: list[DisplayFrameCandidate],
    *,
    maximum_frames: int = 5,
    minimum_stable_frames: int = 2,
) -> list[DisplayFrameCandidate]:
    """Select the strongest temporally stable hand-relative metal display."""
    if maximum_frames < minimum_stable_frames or minimum_stable_frames < 1:
        raise ValueError("invalid stable-frame limits")
    usable = [
        item
        for item in sorted(candidates, key=lambda value: value.frame_time_sec)
        if not item.boundary_clipped
        and item.sharpness >= 15.0
        and np.asarray(item.hand_mask, dtype=bool).any()
        and np.asarray(item.seed_mask, dtype=bool).any()
    ]
    runs: list[list[DisplayFrameCandidate]] = []
    current: list[DisplayFrameCandidate] = []
    previous_center: tuple[float, float] | None = None
    previous_time: float | None = None
    for item in usable:
        center = _relative_seed_center(item.hand_mask, item.seed_mask)
        is_adjacent = previous_time is None or item.frame_time_sec - previous_time <= 1.25
        is_stable = (
            previous_center is None
            or np.hypot(center[0] - previous_center[0], center[1] - previous_center[1])
            <= 0.12
        )
        if current and (not is_adjacent or not is_stable):
            runs.append(current)
            current = []
        current.append(item)
        previous_center = center
        previous_time = item.frame_time_sec
    if current:
        runs.append(current)
    stable = [run for run in runs if len(run) >= minimum_stable_frames]
    if not stable:
        return []
    best = max(
        stable,
        key=lambda run: (
            len(run),
            float(np.mean([item.score for item in run])),
            float(np.mean([item.sharpness for item in run])),
        ),
    )
    if len(best) > maximum_frames:
        best = sorted(
            best, key=lambda item: (item.score, item.sharpness), reverse=True
        )[:maximum_frames]
        best.sort(key=lambda item: item.frame_time_sec)
    return best


def _relative_seed_center(
    hand_mask: np.ndarray, seed_mask: np.ndarray
) -> tuple[float, float]:
    hand_y, hand_x = np.nonzero(np.asarray(hand_mask, dtype=bool))
    seed_y, seed_x = np.nonzero(np.asarray(seed_mask, dtype=bool))
    diagonal = max(
        1.0,
        float(
            np.hypot(
                int(hand_x.max() - hand_x.min() + 1),
                int(hand_y.max() - hand_y.min() + 1),
            )
        ),
    )
    return (
        float(seed_x.mean() - hand_x.min()) / diagonal,
        float(seed_y.mean() - hand_y.min()) / diagonal,
    )


def _as_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.shape != shape:
        value = cv2.resize(
            value.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return value


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    value = mask.astype(np.uint8)
    padded = cv2.copyMakeBorder(value, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    outside = (padded == 0).astype(np.uint8)
    flood = outside.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 2)
    holes = flood[1:-1, 1:-1] == 1
    return np.logical_or(mask, holes)
