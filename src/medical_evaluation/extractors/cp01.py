from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.features.marks import (
    MarkCandidate,
    MarkObservation,
    cluster_stable_marks,
    detect_dark_mark_observations,
    select_punch_candidate,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.segmentation.base import FrameMasks, VideoSegmenter
from medical_evaluation.segmentation.prompts import prompts_for_object
from medical_evaluation.storage import safe_child
from medical_evaluation.video import read_frame


class Cp01FeatureExtractor:
    corner_margin = 0.15
    max_local_cluster_distance = 0.04
    min_mark_area_ratio = 0.0001
    max_mark_area_ratio = 0.01
    maximum_black_value = 55
    maximum_faint_value = 160
    maximum_faint_saturation = 120

    def __init__(
        self,
        *,
        segmenter: VideoSegmenter,
        annotations: VideoAnnotations,
        evidence_root: Path,
        reference_u: float,
        reference_v: float,
        min_observed_frames: int = 3,
    ) -> None:
        if not all(
            math.isfinite(value) and 0 <= value <= 1
            for value in (reference_u, reference_v)
        ):
            raise ValueError("CP01 reference coordinates must be normalized")
        if min_observed_frames <= 0:
            raise ValueError("min_observed_frames must be positive")
        self.segmenter = segmenter
        self.annotations = annotations
        self.evidence_root = evidence_root
        self.reference_u = reference_u
        self.reference_v = reference_v
        self.min_observed_frames = min_observed_frames

    @property
    def model_version(self) -> str:
        return f"{self.segmenter.model_version}+opencv-marks-v3"

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
            raise ValueError("Cp01FeatureExtractor is CP01-only")
        if analysis_width <= 0:
            raise ValueError("analysis_width must be positive")

        prompts = prompts_for_object(
            self.annotations,
            "rubber_dam",
            time_range,
            checkpoint_id=checkpoint_id,
        )
        if len(prompts) != 1 or prompts[0].kind != "box":
            raise ValueError("cp_01 requires exactly one rubber_dam box")

        observations: list[MarkObservation] = []
        valid_frames: dict[int, FrameMasks] = {}
        tracked = self.segmenter.track(
            video_path,
            time_range,
            prompts,
            sample_fps=dense_fps,
        )
        for frame_masks in tracked:
            if not time_range.start_sec <= frame_masks.frame_time_sec <= time_range.end_sec:
                continue
            dam_mask = frame_masks.masks.get("rubber_dam")
            if dam_mask is None or not bool(np.asarray(dam_mask).any()):
                continue
            valid_frames[frame_masks.frame_index] = frame_masks
            frame = read_frame(video_path, frame_masks.frame_index)
            observations.extend(
                detect_dark_mark_observations(
                    frame,
                    dam_mask,
                    frame_index=frame_masks.frame_index,
                    time_sec=frame_masks.frame_time_sec,
                    min_area_ratio=self.min_mark_area_ratio,
                    max_area_ratio=self.max_mark_area_ratio,
                    maximum_black_value=self.maximum_black_value,
                    maximum_faint_value=self.maximum_faint_value,
                    maximum_faint_saturation=self.maximum_faint_saturation,
                )
            )

        candidates = cluster_stable_marks(
            observations,
            min_observed_frames=self.min_observed_frames,
            max_local_distance=self.max_local_cluster_distance,
        )
        selection = select_punch_candidate(
            candidates,
            corner_margin=self.corner_margin,
        )
        punch = selection.punch
        distance = (
            math.hypot(punch.u - self.reference_u, punch.v - self.reference_v)
            if punch is not None
            else None
        )
        evidence = self._write_evidence(video_path, valid_frames, observations, punch)
        return ExtractedEvidence(
            features={
                "mark_u": punch.u if punch is not None else None,
                "mark_v": punch.v if punch is not None else None,
                "reference_u": self.reference_u,
                "reference_v": self.reference_v,
                "mark_reference_distance": distance,
                "mark_candidate_count": float(len(candidates)),
                "mark_selection_ambiguous": selection.status == "ambiguous",
                "mark_missing": selection.status == "missing",
                "dam_valid_frame_count": float(len(valid_frames)),
            },
            evidence=evidence,
        )

    def _write_evidence(
        self,
        video_path: Path,
        valid_frames: dict[int, FrameMasks],
        observations: list[MarkObservation],
        punch: MarkCandidate | None,
    ) -> list[EvidenceItem]:
        if punch is None:
            return []
        matching = [
            item
            for item in observations
            if math.hypot(item.u - punch.u, item.v - punch.v)
            <= self.max_local_cluster_distance
        ]
        if not matching:
            return []
        chosen = max(matching, key=lambda item: item.time_sec)
        frame_masks = valid_frames[chosen.frame_index]
        dam_mask = np.asarray(frame_masks.masks["rubber_dam"], dtype=bool)
        frame = read_frame(video_path, chosen.frame_index)
        overlay = _write_overlay(
            frame,
            dam_mask,
            punch,
            self.reference_u,
            self.reference_v,
            self.evidence_root,
            f"cp_01/overlays/{chosen.frame_index:08d}.jpg",
        )
        return [
            EvidenceItem(
                time_sec=chosen.time_sec,
                overlay_path=overlay.relative_to(self.evidence_root).as_posix(),
                rule="auto_punch_relative_to_saved_reference",
            )
        ]


def _write_overlay(
    frame_bgr: np.ndarray,
    dam_mask: np.ndarray,
    punch: MarkCandidate,
    reference_u: float,
    reference_v: float,
    evidence_root: Path,
    relative_output: str,
) -> Path:
    y_values, x_values = np.where(dam_mask)
    x_min, x_max = int(x_values.min()), int(x_values.max()) + 1
    y_min, y_max = int(y_values.min()), int(y_values.max()) + 1
    width = x_max - x_min
    height = y_max - y_min
    actual = (round(x_min + punch.u * width), round(y_min + punch.v * height))
    reference = (
        round(x_min + reference_u * width),
        round(y_min + reference_v * height),
    )
    canvas = frame_bgr.copy()
    contours, _ = cv2.findContours(
        dam_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(canvas, contours, -1, (0, 215, 255), 2)
    cv2.circle(canvas, actual, 6, (0, 0, 255), -1)
    cv2.circle(canvas, reference, 6, (0, 255, 0), -1)
    cv2.line(canvas, actual, reference, (0, 215, 255), 2)
    output_path = safe_child(evidence_root, relative_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"could not write evidence overlay: {output_path}")
    return output_path
