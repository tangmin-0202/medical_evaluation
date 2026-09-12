from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from medical_evaluation.annotations import VideoAnnotations


PROMPT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "blunt_instrument": (
        "blunt dental instrument",
        "dental instrument with a rounded blunt working tip",
        "metal dental instrument with a curved shaft and rounded working tip",
    ),
    "sharp_probe": (
        "sharp dental explorer probe",
        "pointed dental probe with a needle-like tip",
    ),
    "target_tooth": (
        "isolated white tooth inside the rubber dam clamp",
        "white molar enclosed by the metal rubber dam clamp",
    ),
    "rubber_dam_clamp": (
        "metal rubber dam clamp around the tooth",
        "complete stainless steel rubber dam clamp with two side wings",
    ),
    "rubber_dam": ("green dental rubber dam",),
}


@dataclass(frozen=True)
class GateWindows:
    cp08: tuple[float, float]
    cp09_tail: tuple[float, float]


def load_gate_windows(annotation_path: Path) -> GateWindows:
    annotations = VideoAnnotations.model_validate_json(
        annotation_path.read_text(encoding="utf-8")
    )
    ranges = {step.checkpoint_id: step.time_range for step in annotations.steps}
    for checkpoint_id in ("cp_08", "cp_09"):
        if checkpoint_id not in ranges:
            raise ValueError(f"Missing {checkpoint_id} annotation range")
    cp08 = ranges["cp_08"]
    cp09 = ranges["cp_09"]
    return GateWindows(
        cp08=(cp08.start_sec, cp08.end_sec),
        cp09_tail=(max(cp09.start_sec, cp09.end_sec - 3.0), cp09.end_sec),
    )


def _max_true_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _dominant_component_ratio(mask: np.ndarray) -> float:
    binary = np.asarray(mask, dtype=np.uint8)
    foreground = int(binary.sum())
    if foreground == 0:
        return 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    return largest / foreground


def summarize_masks(masks: list[np.ndarray], *, min_area_px: int) -> dict[str, Any]:
    areas = [int(np.asarray(mask, dtype=bool).sum()) for mask in masks]
    valid = [area >= min_area_px for area in areas]
    coherence = [
        _dominant_component_ratio(mask)
        for mask, is_valid in zip(masks, valid, strict=True)
        if is_valid
    ]
    median_coherence = float(np.median(coherence)) if coherence else 0.0
    longest = _max_true_run(valid)
    return {
        "frame_count": len(masks),
        "valid_frame_count": sum(valid),
        "max_consecutive_valid_frames": longest,
        "median_dominant_component_ratio": median_coherence,
        "automatic_gate_passed": longest >= 3 and median_coherence >= 0.60,
    }


def select_prompt(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one prompt result is required")
    return max(
        candidates,
        key=lambda item: (
            bool(item["automatic_gate_passed"]),
            int(item["max_consecutive_valid_frames"]),
            float(item["median_dominant_component_ratio"]),
        ),
    )
