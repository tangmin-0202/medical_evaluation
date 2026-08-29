from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.features.geometry import frame_center_offset, write_overlay
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.segmentation.base import FrameMasks, VideoSegmenter
from medical_evaluation.segmentation.prompts import prompts_for_object
from medical_evaluation.storage import safe_child
from medical_evaluation.video import read_frame


class Cp09FeatureExtractor:
    minimum_valid_frames = 3

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
        prompts = prompts_for_object(
            self.annotations,
            "rubber_dam_frame",
            time_range,
            checkpoint_id=checkpoint_id,
        )
        tracked = list(
            self.segmenter.track(video_path, time_range, prompts, sample_fps=dense_fps)
        )
        valid: list[tuple[FrameMasks, np.ndarray, float]] = []
        for frame_masks in tracked:
            mask = frame_masks.masks.get("rubber_dam_frame")
            if mask is None:
                continue
            offset = frame_center_offset(mask)
            if offset is not None:
                valid.append((frame_masks, mask, offset))

        evidence: list[EvidenceItem] = []
        selected = sorted({0, len(valid) // 2, len(valid) - 1}) if valid else []
        for position in selected:
            frame_masks, mask, _offset = valid[position]
            frame = read_frame(video_path, frame_masks.frame_index)
            mask_path = safe_child(
                self.evidence_root,
                f"cp_09/masks/{frame_masks.frame_index:08d}.png",
            )
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
                raise OSError(f"could not write mask: {mask_path}")
            overlay = write_overlay(
                frame,
                {"rubber_dam_frame": mask},
                self.evidence_root,
                f"cp_09/overlays/{frame_masks.frame_index:08d}.jpg",
            )
            evidence.append(
                EvidenceItem(
                    time_sec=frame_masks.frame_time_sec,
                    overlay_path=overlay.relative_to(self.evidence_root).as_posix(),
                    rule="rubber_dam_frame_center",
                )
            )

        feature = (
            float(np.median([item[2] for item in valid]))
            if len(valid) >= self.minimum_valid_frames
            else None
        )
        return ExtractedEvidence(
            features={
                "frame_center_offset": feature,
                "frame_valid_count": float(len(valid)),
            },
            evidence=evidence,
        )
