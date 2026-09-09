from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ToothAnchor:
    x: int
    y: int
    width: int
    height: int
    score: float
    frame_position: int

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class FlossLine:
    x1: int
    y1: int
    x2: int
    y2: int
    length: float
    distance_to_tooth: float
    white_ratio: float
    contact_y_offset: float
    score: float


def locate_tooth_anchor(
    frames: Iterable[np.ndarray], *, minimum_score: float = 0.6
) -> ToothAnchor | None:
    best: ToothAnchor | None = None
    for position, image in enumerate(frames):
        _validate_image(image)
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        neutral = (hsv[..., 1] <= 95) & (hsv[..., 2] >= 135)
        roi = np.zeros((height, width), dtype=bool)
        roi[
            round(0.30 * height) : round(0.68 * height),
            round(0.28 * width) : round(0.62 * width),
        ] = True
        raw = cv2.morphologyEx(
            (neutral & roi).astype(np.uint8),
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        dark = hsv[..., 2] <= 115
        ring_size = max(9, round(min(height, width) * 0.038)) | 1
        image_area = height * width
        for label in range(1, count):
            x, y, component_width, component_height, area = map(int, stats[label])
            if not (
                0.0004 * image_area <= area <= 0.02 * image_area
                and 0.01 * width <= component_width <= 0.12 * width
                and 0.02 * height <= component_height <= 0.20 * height
            ):
                continue
            component = labels == label
            ring = cv2.dilate(
                component.astype(np.uint8), np.ones((ring_size, ring_size), np.uint8)
            ).astype(bool) & ~component
            dark_ratio = float(np.mean(dark[ring])) if ring.any() else 0.0
            target_area = max(1.0, 0.0012 * image_area)
            compactness = min(area / target_area, 1.0)
            score = dark_ratio * compactness + 0.10 * (
                (y + component_height / 2) / height
            )
            candidate = ToothAnchor(
                x=x,
                y=y,
                width=component_width,
                height=component_height,
                score=score,
                frame_position=position,
            )
            if best is None or candidate.score > best.score:
                best = candidate
    if best is None or best.score < minimum_score:
        return None
    return best


def segment_tooth(image: np.ndarray, anchor: ToothAnchor) -> np.ndarray:
    _validate_image(image)
    height, width = image.shape[:2]
    pad_x = max(6, round(anchor.width * 0.27))
    pad_y = max(6, round(anchor.height * 0.27))
    left, top = max(0, anchor.x - pad_x), max(0, anchor.y - pad_y)
    right = min(width, anchor.x + anchor.width + pad_x)
    bottom = min(height, anchor.y + anchor.height + pad_y)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidate = (hsv[..., 1] <= 110) & (hsv[..., 2] >= 105)
    roi = np.zeros((height, width), dtype=bool)
    roi[top:bottom, left:right] = True
    raw = cv2.morphologyEx(
        (candidate & roi).astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, 8)
    result = np.zeros((height, width), dtype=bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        center_x, center_y = centroids[label]
        if (
            area >= max(12, round(anchor.width * anchor.height * 0.015))
            and abs(center_x - anchor.center_x) <= anchor.width * 0.9
            and abs(center_y - anchor.center_y) <= anchor.height * 0.9
        ):
            result[labels == label] = True
    return result


def detect_floss_line(image: np.ndarray, anchor: ToothAnchor) -> FlossLine | None:
    _validate_image(image)
    height, width = image.shape[:2]
    half_width = max(round(0.20 * width), anchor.width * 4)
    half_height = max(round(0.21 * height), anchor.height * 3)
    left = max(0, round(anchor.center_x - half_width))
    top = max(0, round(anchor.center_y - half_height))
    right = min(width, round(anchor.center_x + half_width))
    bottom = min(height, round(anchor.center_y + half_height))
    crop = image[top:bottom, left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 120)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(12, round(width * 0.016)),
        minLineLength=max(14, round(width * 0.034)),
        maxLineGap=max(4, round(width * 0.009), round(anchor.width * 0.5)),
    )
    if lines is None:
        return None
    best: FlossLine | None = None
    minimum_length = max(width * 0.034, max(anchor.width, anchor.height) * 1.05)
    for local_x1, local_y1, local_x2, local_y2 in lines[:, 0]:
        x1, y1 = int(local_x1 + left), int(local_y1 + top)
        x2, y2 = int(local_x2 + left), int(local_y2 + top)
        vector_x, vector_y = x2 - x1, y2 - y1
        length = float(np.hypot(vector_x, vector_y))
        if length < minimum_length:
            continue
        projection = np.clip(
            (
                (anchor.center_x - x1) * vector_x
                + (anchor.center_y - y1) * vector_y
            )
            / (length * length),
            0,
            1,
        )
        closest_x = x1 + projection * vector_x
        closest_y = y1 + projection * vector_y
        distance = float(
            np.hypot(closest_x - anchor.center_x, closest_y - anchor.center_y)
        )
        if distance > max(anchor.width, anchor.height) * 1.15:
            continue
        band = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.line(
            band,
            (int(local_x1), int(local_y1)),
            (int(local_x2), int(local_y2)),
            1,
            3,
        )
        neutral = (hsv[..., 1] <= 105) & (hsv[..., 2] >= 105)
        white_ratio = float(np.mean(neutral[band.astype(bool)]))
        if white_ratio < 0.42:
            continue
        score = length * white_ratio / (20 + distance)
        candidate = FlossLine(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            length=length,
            distance_to_tooth=distance,
            white_ratio=white_ratio,
            contact_y_offset=float(closest_y - anchor.center_y),
            score=score,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def _validate_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a non-empty BGR frame")
