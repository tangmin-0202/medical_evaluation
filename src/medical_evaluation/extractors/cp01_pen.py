"""Automatic CP01 marking-pen presence gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from medical_evaluation.domain import TimeRange
from medical_evaluation.features.cp01_pen import (
    measure_pen_candidate,
    segment_pen_candidate,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.segmentation.base import (
    FrameMasks,
)
from medical_evaluation.storage import atomic_write_json, safe_child
from medical_evaluation.video import read_frame, sample_frames

PEN_OBJECT_ID = "cp01_marking_pen"


class Cp01PenGateExtractor:
    sparse_fps = 2.0

    def __init__(
        self,
        *,
        evidence_root: Path,
        minimum_consecutive_frames: int = 2,
    ) -> None:
        if minimum_consecutive_frames <= 0:
            raise ValueError("minimum_consecutive_frames must be positive")
        self.evidence_root = evidence_root
        self.minimum_consecutive_frames = minimum_consecutive_frames

    @property
    def model_version(self) -> str:
        return "opencv-cp01-pen-gate-v2"

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id != "cp_01":
            raise ValueError("Cp01PenGateExtractor is CP01-only")
        if dense_fps <= 0 or analysis_width <= 0:
            raise ValueError("sampling and analysis dimensions must be positive")

        decoded: dict[int, np.ndarray] = {}
        tracked: list[FrameMasks] = []
        for sampled in sample_frames(
            video_path,
            start_sec=time_range.start_sec,
            end_sec=time_range.end_sec,
            sample_fps=self.sparse_fps,
        ):
            mask = segment_pen_candidate(sampled.image_bgr)
            tracked.append(
                FrameMasks(
                    frame_index=sampled.frame_index,
                    frame_time_sec=sampled.time_sec,
                    masks={PEN_OBJECT_ID: mask},
                )
            )
            decoded[sampled.frame_index] = sampled.image_bgr

        rows: list[dict[str, Any]] = []
        overlays: list[tuple[FrameMasks, str, bool]] = []
        accepted_times: list[float] = []
        current_run = 0
        maximum_run = 0
        for item in tracked:
            frame = decoded.get(item.frame_index)
            if frame is None:
                frame = read_frame(video_path, item.frame_index)
            raw_mask = np.asarray(
                item.masks.get(PEN_OBJECT_ID, np.zeros(frame.shape[:2], dtype=bool)),
                dtype=bool,
            )
            measurement = measure_pen_candidate(frame, raw_mask)
            if measurement.accepted:
                current_run += 1
                accepted_times.append(item.frame_time_sec)
            else:
                current_run = 0
            maximum_run = max(maximum_run, current_run)
            relative = self._write_bundle(
                item,
                frame,
                raw_mask,
                measurement.selected_mask,
                accepted=measurement.accepted,
                reason=measurement.reason,
            )
            overlays.append((item, relative, measurement.accepted))
            rows.append(
                {
                    "frame_index": item.frame_index,
                    "time_sec": item.frame_time_sec,
                    "mask_area_px": measurement.area_px,
                    "dark_pixel_ratio": measurement.dark_pixel_ratio,
                    "dominant_component_ratio": measurement.dominant_component_ratio,
                    "elongation": measurement.elongation,
                    "accepted": measurement.accepted,
                    "rejection_reason": (
                        None if measurement.accepted else measurement.reason
                    ),
                }
            )

        stage_scan_reliable = len(rows) >= self.minimum_consecutive_frames
        presence = stage_scan_reliable and maximum_run >= self.minimum_consecutive_frames
        features: dict[str, float | bool | None] = {
            "stage_scan_reliable": stage_scan_reliable,
            "pen_presence_detected": presence,
            "pen_valid_frame_count": float(len(accepted_times)),
            "pen_max_consecutive_valid_frames": float(maximum_run),
            "pen_first_seen_sec": min(accepted_times) if accepted_times else None,
            "pen_last_seen_sec": max(accepted_times) if accepted_times else None,
        }
        failure_reason = None
        if not stage_scan_reliable:
            failure_reason = "stage_scan_unreliable"
        elif not presence:
            failure_reason = "pen_not_observed_consecutively"
        atomic_write_json(
            safe_child(self.evidence_root, "cp_01/pen_gate/metrics.json"),
            {
                "detector": "opencv_dark_elongated_near_green_v2",
                "minimum_consecutive_frames": self.minimum_consecutive_frames,
                "features": features,
                "failure_reason": failure_reason,
                "frames": rows,
            },
        )
        selected = [item for item in overlays if item[2]] if presence else overlays
        representatives = _representatives(selected, limit=3)
        evidence = [
            EvidenceItem(
                time_sec=item.frame_time_sec,
                overlay_path=relative,
                rule=(
                    "cp01_marking_pen_presence"
                    if accepted
                    else "cp01_marking_pen_negative_scan"
                ),
            )
            for item, relative, accepted in representatives
        ]
        return ExtractedEvidence(features=features, evidence=evidence)

    def _write_bundle(
        self,
        item: FrameMasks,
        frame: np.ndarray,
        raw_mask: np.ndarray,
        selected_mask: np.ndarray,
        *,
        accepted: bool,
        reason: str,
    ) -> str:
        stem = f"{item.frame_index:08d}"
        base = "cp_01/pen_gate"
        raw_path = safe_child(self.evidence_root, f"{base}/raw/{stem}.jpg")
        mask_path = safe_child(self.evidence_root, f"{base}/masks/raw/{stem}.png")
        selected_path = safe_child(
            self.evidence_root, f"{base}/masks/selected/{stem}.png"
        )
        overlay_relative = f"{base}/overlays/{stem}.jpg"
        overlay_path = safe_child(self.evidence_root, overlay_relative)
        for path in (raw_path, mask_path, selected_path, overlay_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        canvas = frame.copy()
        color = (0, 255, 0) if accepted else (0, 0, 255)
        contours, _ = cv2.findContours(
            selected_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, color, 2)
        cv2.putText(
            canvas,
            reason,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        if not cv2.imwrite(str(raw_path), frame):
            raise OSError(f"could not write CP01 pen raw frame: {raw_path}")
        if not cv2.imwrite(str(mask_path), raw_mask.astype(np.uint8) * 255):
            raise OSError(f"could not write CP01 pen raw mask: {mask_path}")
        if not cv2.imwrite(str(selected_path), selected_mask.astype(np.uint8) * 255):
            raise OSError(f"could not write CP01 pen selected mask: {selected_path}")
        if not cv2.imwrite(str(overlay_path), canvas):
            raise OSError(f"could not write CP01 pen overlay: {overlay_path}")
        return overlay_relative


def _representatives(
    values: list[tuple[FrameMasks, str, bool]], *, limit: int
) -> list[tuple[FrameMasks, str, bool]]:
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, num=limit, dtype=int)
    return [values[int(index)] for index in indices]
