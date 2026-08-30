from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.features.geometry import relative_bbox_center_offset, write_overlay
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
        oral_prompts = prompts_for_object(
            self.annotations,
            "oral_region",
            time_range,
            checkpoint_id=checkpoint_id,
            boundary_tolerance_sec=self.prompt_boundary_tolerance_sec,
        )
        prompts = sorted(frame_prompts + oral_prompts, key=lambda item: item.frame_time_sec)
        tracking_range = TimeRange(
            start_sec=min(time_range.start_sec, *(item.frame_time_sec for item in prompts)),
            end_sec=max(time_range.end_sec, *(item.frame_time_sec for item in prompts)),
        )
        tracked = self.segmenter.track(
            video_path,
            tracking_range,
            prompts,
            sample_fps=dense_fps,
        )

        frame_valid_count = 0
        oral_region_valid_count = 0
        paired: list[tuple[FrameMasks, np.ndarray, np.ndarray, float]] = []
        for frame_masks in tracked:
            if not time_range.start_sec <= frame_masks.frame_time_sec <= time_range.end_sec:
                continue

            frame_mask = frame_masks.masks.get("rubber_dam_frame")
            oral_mask = frame_masks.masks.get("oral_region")
            frame_is_valid = frame_mask is not None and bool(np.asarray(frame_mask).any())
            oral_is_valid = oral_mask is not None and bool(np.asarray(oral_mask).any())
            frame_valid_count += int(frame_is_valid)
            oral_region_valid_count += int(oral_is_valid)

            if not frame_is_valid or not oral_is_valid:
                continue
            assert frame_mask is not None
            assert oral_mask is not None
            offset = relative_bbox_center_offset(frame_mask, oral_mask)
            if offset is not None:
                paired.append((frame_masks, frame_mask, oral_mask, offset))

        evidence: list[EvidenceItem] = []
        selected = sorted({0, len(paired) // 2, len(paired) - 1}) if paired else []
        for position in selected:
            frame_masks, frame_mask, oral_mask, _offset = paired[position]
            frame = read_frame(video_path, frame_masks.frame_index)
            masks = {
                "rubber_dam_frame": frame_mask,
                "oral_region": oral_mask,
            }
            for object_id, mask in masks.items():
                mask_path = safe_child(
                    self.evidence_root,
                    f"cp_09/masks/{object_id}/{frame_masks.frame_index:08d}.png",
                )
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
                    raise OSError(f"could not write mask: {mask_path}")

            overlay = write_overlay(
                frame,
                masks,
                self.evidence_root,
                f"cp_09/overlays/{frame_masks.frame_index:08d}.jpg",
            )
            evidence.append(
                EvidenceItem(
                    time_sec=frame_masks.frame_time_sec,
                    overlay_path=overlay.relative_to(self.evidence_root).as_posix(),
                    rule="rubber_dam_frame_relative_to_oral_region",
                )
            )

        feature = (
            float(np.median([item[3] for item in paired]))
            if len(paired) >= self.minimum_valid_frames
            else None
        )
        return ExtractedEvidence(
            features={
                "frame_oral_center_offset": feature,
                "frame_valid_count": float(frame_valid_count),
                "oral_region_valid_count": float(oral_region_valid_count),
                "paired_valid_count": float(len(paired)),
            },
            evidence=evidence,
        )
