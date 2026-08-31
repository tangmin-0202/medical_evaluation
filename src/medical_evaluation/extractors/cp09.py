from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.annotations import BoxPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.features.geometry import (
    relative_bbox_center_offset_to_box,
    write_reference_overlay,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.segmentation.base import FrameMasks, VideoSegmenter
from medical_evaluation.segmentation.prompts import prompts_for_object
from medical_evaluation.storage import safe_child
from medical_evaluation.video import read_frame


class Cp09FeatureExtractor:
    minimum_valid_frames = 3
    prompt_boundary_tolerance_sec = 0.5

    def __init__(
        self,
        *,
        segmenter: VideoSegmenter,
        annotations: VideoAnnotations,
        evidence_root: Path,
    ) -> None:
        self.segmenter = segmenter
        self.annotations = annotations
        self.evidence_root = evidence_root

    @property
    def model_version(self) -> str:
        return self.segmenter.model_version

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id != "cp_09":
            raise ValueError("Cp09FeatureExtractor is CP09-only")
        if analysis_width <= 0:
            raise ValueError("analysis_width must be positive")

        frame_prompts = prompts_for_object(
            self.annotations,
            "rubber_dam_frame",
            time_range,
            checkpoint_id=checkpoint_id,
            boundary_tolerance_sec=self.prompt_boundary_tolerance_sec,
        )
        oral_reference = self._oral_reference_box(time_range, checkpoint_id)
        tracking_range = TimeRange(
            start_sec=min(time_range.start_sec, *(item.frame_time_sec for item in frame_prompts)),
            end_sec=max(time_range.end_sec, *(item.frame_time_sec for item in frame_prompts)),
        )
        tracked = self.segmenter.track(
            video_path,
            tracking_range,
            frame_prompts,
            sample_fps=dense_fps,
        )

        frame_valid_count = 0
        valid_offsets: list[tuple[FrameMasks, np.ndarray, float]] = []
        for frame_masks in tracked:
            if not time_range.start_sec <= frame_masks.frame_time_sec <= time_range.end_sec:
                continue

            frame_mask = frame_masks.masks.get("rubber_dam_frame")
            frame_is_valid = frame_mask is not None and bool(np.asarray(frame_mask).any())
            frame_valid_count += int(frame_is_valid)

            if not frame_is_valid:
                continue
            assert frame_mask is not None
            offset = relative_bbox_center_offset_to_box(frame_mask, oral_reference)
            if offset is not None:
                valid_offsets.append((frame_masks, frame_mask, offset))

        evidence: list[EvidenceItem] = []
        selected = (
            sorted({0, len(valid_offsets) // 2, len(valid_offsets) - 1})
            if valid_offsets
            else []
        )
        for position in selected:
            frame_masks, frame_mask, _offset = valid_offsets[position]
            frame = read_frame(video_path, frame_masks.frame_index)
            mask_path = safe_child(
                self.evidence_root,
                f"cp_09/masks/rubber_dam_frame/{frame_masks.frame_index:08d}.png",
            )
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), frame_mask.astype(np.uint8) * 255):
                raise OSError(f"could not write mask: {mask_path}")

            overlay = write_reference_overlay(
                frame,
                frame_mask,
                oral_reference,
                self.evidence_root,
                f"cp_09/overlays/{frame_masks.frame_index:08d}.jpg",
            )
            evidence.append(
                EvidenceItem(
                    time_sec=frame_masks.frame_time_sec,
                    overlay_path=overlay.relative_to(self.evidence_root).as_posix(),
                    rule="rubber_dam_frame_relative_to_oral_reference",
                )
            )

        feature = (
            float(np.median([item[2] for item in valid_offsets]))
            if len(valid_offsets) >= self.minimum_valid_frames
            else None
        )
        return ExtractedEvidence(
            features={
                "frame_oral_center_offset": feature,
                "frame_valid_count": float(frame_valid_count),
                "oral_reference_count": 1.0,
                "relative_offset_valid_count": float(len(valid_offsets)),
            },
            evidence=evidence,
        )

    def _oral_reference_box(
        self,
        time_range: TimeRange,
        checkpoint_id: str,
    ) -> tuple[float, float, float, float]:
        allowed_start = time_range.start_sec - self.prompt_boundary_tolerance_sec
        allowed_end = time_range.end_sec + self.prompt_boundary_tolerance_sec
        candidates = [
            prompt
            for prompt in self.annotations.prompts
            if prompt.object_id == "oral_region"
            and allowed_start <= prompt.frame_time_sec <= allowed_end
        ]
        if any(not isinstance(prompt, BoxPrompt) for prompt in candidates):
            raise ValueError(f"{checkpoint_id} oral_region must be a box")
        if len(candidates) != 1:
            raise ValueError(f"{checkpoint_id} requires exactly one oral_region box")
        reference = candidates[0]
        assert isinstance(reference, BoxPrompt)
        return reference.x1, reference.y1, reference.x2, reference.y2
