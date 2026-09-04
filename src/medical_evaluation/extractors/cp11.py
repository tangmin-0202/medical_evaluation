from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.annotations import BoxPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.features.appearance import (
    box_overlap_ratio,
    green_dam_area_ratio,
    green_dam_mask,
    visible_reference_mask,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.segmentation.base import FrameMasks, VideoSegmenter
from medical_evaluation.segmentation.prompt_policy import (
    AnnotationPromptPolicy,
    Cp09Cp11PromptPolicy,
)
from medical_evaluation.storage import safe_child
from medical_evaluation.video import read_frame, sample_frames


class Cp11FeatureExtractor:
    stage_scan_fps = 1.0
    final_window_sec = 3.0
    minimum_final_frames = 3
    minimum_green_area_per_present_frame = 0.02
    frame_appearance_max_lab_distance = 20.0

    def __init__(
        self,
        *,
        segmenter: VideoSegmenter,
        annotations: VideoAnnotations,
        evidence_root: Path,
        frame_reference_time_range: TimeRange,
        min_stage_dam_presence_ratio: float,
        min_final_dam_presence_ratio: float = 0.5,
        prompt_policy: Cp09Cp11PromptPolicy | None = None,
    ) -> None:
        if not 0 <= min_stage_dam_presence_ratio <= 1:
            raise ValueError("min_stage_dam_presence_ratio must be normalized")
        if not 0 <= min_final_dam_presence_ratio <= 1:
            raise ValueError("min_final_dam_presence_ratio must be normalized")
        self.segmenter = segmenter
        self.annotations = annotations
        self.evidence_root = evidence_root
        self.frame_reference_time_range = frame_reference_time_range
        self.min_stage_dam_presence_ratio = min_stage_dam_presence_ratio
        self.min_final_dam_presence_ratio = min_final_dam_presence_ratio
        self.prompt_policy = prompt_policy or AnnotationPromptPolicy()

    @property
    def model_version(self) -> str:
        return f"{self.segmenter.model_version}+opencv-appearance-v2"

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id != "cp_11":
            raise ValueError("Cp11FeatureExtractor is CP11-only")
        if analysis_width <= 0:
            raise ValueError("analysis_width must be positive")

        stage_samples = [
            (item.time_sec, green_dam_area_ratio(item.image_bgr))
            for item in sample_frames(
                video_path,
                start_sec=time_range.start_sec,
                end_sec=time_range.end_sec,
                sample_fps=self.stage_scan_fps,
            )
        ]
        presence = (
            float(
                np.mean(
                    [
                        ratio >= self.minimum_green_area_per_present_frame
                        for _time_sec, ratio in stage_samples
                    ]
                )
            )
            if stage_samples
            else None
        )
        final_start_sec = max(
            time_range.start_sec,
            time_range.end_sec - self.final_window_sec,
        )
        final_ratios = [
            ratio for time_sec, ratio in stage_samples if time_sec >= final_start_sec
        ]
        final_presence = (
            float(
                np.mean(
                    [
                        ratio >= self.minimum_green_area_per_present_frame
                        for ratio in final_ratios
                    ]
                )
            )
            if final_ratios
            else None
        )
        empty_features: dict[str, float | bool | None] = {
            "dam_stage_presence_ratio": presence,
            "dam_final_presence_ratio": final_presence,
            "dam_area_ratio": None,
            "nose_overlap": None,
            "visible_frame_area_ratio": None,
            "final_valid_frame_count": 0.0,
        }
        if presence is None or presence < self.min_stage_dam_presence_ratio:
            return ExtractedEvidence(features=empty_features, evidence=[])
        if (
            final_presence is None
            or final_presence < self.min_final_dam_presence_ratio
        ):
            return ExtractedEvidence(features=empty_features, evidence=[])

        final_range = TimeRange(
            start_sec=final_start_sec,
            end_sec=time_range.end_sec,
        )
        dam_prompts = self.prompt_policy.dam_prompts(
            self.annotations,
            final_range,
            checkpoint_id=checkpoint_id,
        )
        nose_box = self._nose_box(final_range)
        frame_prompts = self.prompt_policy.frame_prompts(
            self.annotations,
            self.frame_reference_time_range,
            checkpoint_id=checkpoint_id,
        )

        dam_frames = list(
            self.segmenter.track(
                video_path,
                final_range,
                dam_prompts,
                sample_fps=dense_fps,
            )
        )
        frame_tracking_range = TimeRange(
            start_sec=self.frame_reference_time_range.start_sec,
            end_sec=time_range.end_sec,
        )
        frame_frames = list(
            self.segmenter.track(
                video_path,
                frame_tracking_range,
                frame_prompts,
                sample_fps=1.0,
            )
        )
        reference_lab = self._frame_reference_lab(video_path, frame_frames)
        frame_by_index = {item.frame_index: item for item in frame_frames}

        measurements: list[tuple[FrameMasks, np.ndarray, np.ndarray, float, float, float]] = []
        for item in dam_frames:
            if not final_range.start_sec <= item.frame_time_sec <= final_range.end_sec:
                continue
            dam_mask = item.masks.get("rubber_dam")
            frame_item = frame_by_index.get(item.frame_index)
            frame_mask = (
                frame_item.masks.get("rubber_dam_frame")
                if frame_item is not None
                else None
            )
            if dam_mask is None or frame_mask is None:
                continue
            raw_dam = np.asarray(dam_mask, dtype=bool)
            candidate = np.asarray(frame_mask, dtype=bool)
            if raw_dam.shape != candidate.shape or not raw_dam.any():
                continue
            frame = read_frame(video_path, item.frame_index)
            dam = raw_dam & green_dam_mask(frame)
            visible_frame = visible_reference_mask(
                frame,
                candidate,
                reference_lab,
                max_lab_distance=self.frame_appearance_max_lab_distance,
            )
            measurements.append(
                (
                    item,
                    dam,
                    visible_frame,
                    float(dam.mean()),
                    box_overlap_ratio(dam, nose_box),
                    float(visible_frame.mean()),
                )
            )

        if len(measurements) < self.minimum_final_frames:
            return ExtractedEvidence(
                features={
                    **empty_features,
                    "final_valid_frame_count": float(len(measurements)),
                },
                evidence=[],
            )
        evidence = self._write_evidence(video_path, measurements, nose_box)
        return ExtractedEvidence(
            features={
                "dam_stage_presence_ratio": presence,
                "dam_final_presence_ratio": final_presence,
                "dam_area_ratio": float(np.median([item[3] for item in measurements])),
                "nose_overlap": float(np.median([item[4] for item in measurements])),
                "visible_frame_area_ratio": float(
                    np.median([item[5] for item in measurements])
                ),
                "final_valid_frame_count": float(len(measurements)),
            },
            evidence=evidence,
        )

    def _nose_box(self, final_range: TimeRange) -> tuple[float, float, float, float]:
        candidates = [
            item
            for item in self.annotations.prompts
            if item.object_id == "nose_region"
            and final_range.start_sec <= item.frame_time_sec <= final_range.end_sec
        ]
        if len(candidates) != 1 or not isinstance(candidates[0], BoxPrompt):
            raise ValueError("cp_11 requires exactly one nose_region box")
        item = candidates[0]
        return item.x1, item.y1, item.x2, item.y2

    def _frame_reference_lab(
        self,
        video_path: Path,
        frames: list[FrameMasks],
    ) -> np.ndarray:
        for item in frames:
            if not (
                self.frame_reference_time_range.start_sec
                <= item.frame_time_sec
                <= self.frame_reference_time_range.end_sec
            ):
                continue
            mask = item.masks.get("rubber_dam_frame")
            if mask is None or not np.asarray(mask, dtype=bool).any():
                continue
            frame = read_frame(video_path, item.frame_index)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            return np.median(lab[np.asarray(mask, dtype=bool)], axis=0).astype(float)
        raise ValueError("cp_11 has no readable CP09 frame appearance reference")

    def _write_evidence(
        self,
        video_path: Path,
        measurements: list[tuple[FrameMasks, np.ndarray, np.ndarray, float, float, float]],
        nose_box: tuple[float, float, float, float],
    ) -> list[EvidenceItem]:
        selected = sorted({0, len(measurements) // 2, len(measurements) - 1})
        evidence: list[EvidenceItem] = []
        for position in selected:
            item, dam, candidate, _dam_ratio, _nose, _visible = measurements[position]
            frame = read_frame(video_path, item.frame_index)
            overlay = frame.copy()
            dam_contours, _ = cv2.findContours(
                dam.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            frame_contours, _ = cv2.findContours(
                candidate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, dam_contours, -1, (0, 255, 0), 2)
            cv2.drawContours(overlay, frame_contours, -1, (255, 255, 255), 1)
            height, width = dam.shape
            x1, y1, x2, y2 = nose_box
            cv2.rectangle(
                overlay,
                (round(x1 * width), round(y1 * height)),
                (round(x2 * width), round(y2 * height)),
                (0, 215, 255),
                2,
            )
            output = safe_child(
                self.evidence_root,
                f"cp_11/overlays/{item.frame_index:08d}.jpg",
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), overlay):
                raise OSError(f"could not write evidence overlay: {output}")
            evidence.append(
                EvidenceItem(
                    time_sec=item.frame_time_sec,
                    overlay_path=output.relative_to(self.evidence_root).as_posix(),
                    rule="dam_spread_frame_covered_nose_clear",
                )
            )
        return evidence
