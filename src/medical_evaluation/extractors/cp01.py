from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.features.marks import (
    MarkCandidate,
    MarkObservation,
    PunchSelection,
    cluster_stable_marks,
    detect_dark_mark_observations,
    select_darkest_nearest_reference,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.segmentation.base import FrameMasks, VideoSegmenter
from medical_evaluation.segmentation.prompts import prompts_for_object
from medical_evaluation.storage import atomic_write_json, safe_child
from medical_evaluation.video import read_frame


class Cp01FeatureExtractor:
    def __init__(
        self,
        *,
        segmenter: VideoSegmenter,
        annotations: VideoAnnotations,
        evidence_root: Path,
        reference_u: float,
        reference_v: float,
        min_pen_presence_frames: int = 2,
        min_mark_observed_frames: int = 3,
        min_mark_area_ratio: float = 0.0001,
        max_mark_area_ratio: float = 0.01,
        maximum_black_value: int = 55,
        maximum_faint_value: int = 160,
        maximum_faint_saturation: int = 120,
        max_mark_aspect_ratio: float = 2.0,
        min_mark_circularity: float = 0.35,
        max_local_cluster_distance: float = 0.04,
        local_darkness_ring_radius: int = 5,
    ) -> None:
        if not all(
            math.isfinite(value) and 0 <= value <= 1
            for value in (reference_u, reference_v)
        ):
            raise ValueError("CP01 reference coordinates must be normalized")
        if min_pen_presence_frames <= 0 or min_mark_observed_frames <= 0:
            raise ValueError("CP01 frame thresholds must be positive")
        self.segmenter = segmenter
        self.annotations = annotations
        self.evidence_root = evidence_root
        self.reference_u = reference_u
        self.reference_v = reference_v
        self.min_pen_presence_frames = min_pen_presence_frames
        self.min_mark_observed_frames = min_mark_observed_frames
        self.min_mark_area_ratio = min_mark_area_ratio
        self.max_mark_area_ratio = max_mark_area_ratio
        self.maximum_black_value = maximum_black_value
        self.maximum_faint_value = maximum_faint_value
        self.maximum_faint_saturation = maximum_faint_saturation
        self.max_mark_aspect_ratio = max_mark_aspect_ratio
        self.min_mark_circularity = min_mark_circularity
        self.max_local_cluster_distance = max_local_cluster_distance
        self.local_darkness_ring_radius = local_darkness_ring_radius

    @property
    def model_version(self) -> str:
        return f"{self.segmenter.model_version}+opencv-pen-presence-marks-v2"

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

        dam_prompts = prompts_for_object(
            self.annotations,
            "rubber_dam",
            time_range,
            checkpoint_id=checkpoint_id,
        )
        if len(dam_prompts) != 1 or dam_prompts[0].kind != "box":
            raise ValueError("cp_01 requires exactly one rubber_dam box")
        pen_prompt_count = sum(
            prompt.object_id == "marking_pen"
            and time_range.start_sec <= prompt.frame_time_sec <= time_range.end_sec
            for prompt in self.annotations.prompts
        )
        pen_prompts = (
            prompts_for_object(
                self.annotations,
                "marking_pen",
                time_range,
                checkpoint_id=checkpoint_id,
            )
            if pen_prompt_count
            else []
        )
        if pen_prompts:
            has_box = any(prompt.kind == "box" for prompt in pen_prompts)
            if has_box:
                valid_pen_prompts = (
                    len(pen_prompts) == 1 and pen_prompts[0].kind == "box"
                )
            else:
                frame_times = {prompt.frame_time_sec for prompt in pen_prompts}
                valid_pen_prompts = (
                    1 <= len(pen_prompts) <= 2
                    and all(
                        prompt.kind == "point" and prompt.positive is True
                        for prompt in pen_prompts
                    )
                    and len(frame_times) == 1
                )
            if not valid_pen_prompts:
                raise ValueError(
                    "cp_01 marking_pen requires one box or 1-2 positive points "
                    "on one frame"
                )

        observations: list[MarkObservation] = []
        valid_frames: dict[int, FrameMasks] = {}
        pen_valid_frame_count = 0
        first_pen_frame_index: int | None = None
        tracked = self.segmenter.track(
            video_path,
            time_range,
            [*dam_prompts, *pen_prompts],
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
                    max_mark_aspect_ratio=self.max_mark_aspect_ratio,
                    min_mark_circularity=self.min_mark_circularity,
                    local_darkness_ring_radius=self.local_darkness_ring_radius,
                )
            )
            pen_mask = frame_masks.masks.get("marking_pen")
            pen_nonempty = pen_mask is not None and bool(np.asarray(pen_mask).any())
            if pen_nonempty:
                pen_valid_frame_count += 1
                if first_pen_frame_index is None:
                    first_pen_frame_index = frame_masks.frame_index

        pen_presence_detected = pen_valid_frame_count >= self.min_pen_presence_frames
        candidates = cluster_stable_marks(
            observations,
            min_observed_frames=self.min_mark_observed_frames,
            max_local_distance=self.max_local_cluster_distance,
        )
        eligible_candidates = candidates if pen_presence_detected else []
        selection = select_darkest_nearest_reference(
            eligible_candidates,
            reference_u=self.reference_u,
            reference_v=self.reference_v,
        )
        punch = selection.punch
        distance = (
            math.hypot(punch.u - self.reference_u, punch.v - self.reference_v)
            if punch is not None
            else None
        )
        evidence = self._write_evidence(
            video_path,
            valid_frames,
            first_pen_frame_index,
            selection,
        )
        self._write_evidence_json(
            pen_valid_frame_count,
            pen_presence_detected,
            eligible_candidates,
            selection,
        )
        return ExtractedEvidence(
            features={
                "pen_valid_frame_count": float(pen_valid_frame_count),
                "pen_presence_detected": pen_presence_detected,
                "mark_candidate_count": float(len(eligible_candidates)),
                "selected_mark_u": punch.u if punch else None,
                "selected_mark_v": punch.v if punch else None,
                "selected_mark_darkness": punch.median_darkness if punch else None,
                "mark_reference_distance": distance,
                "dam_valid_frame_count": float(len(valid_frames)),
            },
            evidence=evidence,
        )

    def _write_evidence(
        self,
        video_path: Path,
        valid_frames: dict[int, FrameMasks],
        first_pen_frame_index: int | None,
        selection: PunchSelection,
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        if first_pen_frame_index is not None:
            frame_masks = valid_frames[first_pen_frame_index]
            evidence.append(
                self._write_pen_presence_overlay(video_path, frame_masks)
            )
        if selection.punch is not None:
            evidence.append(
                self._write_mark_overlay(
                    video_path,
                    valid_frames[selection.punch.last_frame_index],
                    selection.ranked_candidates,
                    "selection",
                    "darkest_marks_nearest_reference",
                    selected=selection.punch,
                )
            )
        return evidence[:3]

    def _write_pen_presence_overlay(
        self,
        video_path: Path,
        frame_masks: FrameMasks,
    ) -> EvidenceItem:
        frame = read_frame(video_path, frame_masks.frame_index)
        dam = np.asarray(frame_masks.masks["rubber_dam"], dtype=bool)
        pen = np.asarray(frame_masks.masks["marking_pen"], dtype=bool)
        canvas = frame.copy()
        _draw_mask_contour(canvas, dam, (0, 215, 255))
        _draw_mask_contour(canvas, pen, (255, 255, 0))
        cv2.putText(
            canvas,
            "marking pen detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        relative = f"cp_01/overlays/pen-{frame_masks.frame_index:08d}.jpg"
        _save_overlay(canvas, self.evidence_root, relative)
        return EvidenceItem(
            time_sec=frame_masks.frame_time_sec,
            overlay_path=relative,
            rule="stable_marking_pen_presence",
        )

    def _write_mark_overlay(
        self,
        video_path: Path,
        frame_masks: FrameMasks,
        candidates: Sequence[MarkCandidate],
        name: str,
        rule: str,
        *,
        selected: MarkCandidate | None = None,
    ) -> EvidenceItem:
        frame = read_frame(video_path, frame_masks.frame_index)
        dam = np.asarray(frame_masks.masks["rubber_dam"], dtype=bool)
        canvas = frame.copy()
        _draw_mask_contour(canvas, dam, (0, 215, 255))
        for candidate in candidates:
            color = (128, 128, 128) if selected is None else (0, 0, 255)
            cv2.circle(canvas, _local_to_pixel(dam, candidate.u, candidate.v), 6, color, -1)
        if selected is not None:
            actual = _local_to_pixel(dam, selected.u, selected.v)
            reference = _local_to_pixel(dam, self.reference_u, self.reference_v)
            cv2.circle(canvas, reference, 6, (0, 255, 0), -1)
            cv2.line(canvas, actual, reference, (0, 215, 255), 2)
        relative = f"cp_01/overlays/{name}-{frame_masks.frame_index:08d}.jpg"
        _save_overlay(canvas, self.evidence_root, relative)
        return EvidenceItem(
            time_sec=frame_masks.frame_time_sec,
            overlay_path=relative,
            rule=rule,
        )

    def _write_evidence_json(
        self,
        pen_valid_frame_count: int,
        pen_presence_detected: bool,
        candidates: Sequence[MarkCandidate],
        selection: PunchSelection,
    ) -> None:
        atomic_write_json(
            safe_child(self.evidence_root, "cp_01/evidence.json"),
            {
                "thresholds": {
                    "min_pen_presence_frames": self.min_pen_presence_frames,
                    "min_mark_observed_frames": self.min_mark_observed_frames,
                    "min_mark_area_ratio": self.min_mark_area_ratio,
                    "max_mark_area_ratio": self.max_mark_area_ratio,
                    "maximum_black_value": self.maximum_black_value,
                    "maximum_faint_value": self.maximum_faint_value,
                    "maximum_faint_saturation": self.maximum_faint_saturation,
                    "max_mark_aspect_ratio": self.max_mark_aspect_ratio,
                    "min_mark_circularity": self.min_mark_circularity,
                    "max_local_cluster_distance": self.max_local_cluster_distance,
                    "local_darkness_ring_radius": self.local_darkness_ring_radius,
                },
                "pen_presence": {
                    "detected": pen_presence_detected,
                    "valid_frame_count": pen_valid_frame_count,
                },
                "candidate_tracks": [_track_payload(item) for item in candidates],
                "selection_reason": selection.reason,
            },
        )


def _track_payload(candidate: MarkCandidate) -> dict[str, float | int]:
    return {
        "u": candidate.u,
        "v": candidate.v,
        "first_sec": candidate.first_sec,
        "last_sec": candidate.last_sec,
        "observed_frame_count": candidate.observed_frame_count,
        "median_darkness": candidate.median_darkness,
        "darkness_delta": candidate.darkness_delta,
    }


def _local_to_pixel(mask: np.ndarray, u: float, v: float) -> tuple[int, int]:
    y_values, x_values = np.where(mask)
    x_min, x_max = int(x_values.min()), int(x_values.max()) + 1
    y_min, y_max = int(y_values.min()), int(y_values.max()) + 1
    return (
        round(x_min + u * (x_max - x_min)),
        round(y_min + v * (y_max - y_min)),
    )


def _draw_mask_contour(
    canvas: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(canvas, contours, -1, color, 2)


def _save_overlay(canvas: np.ndarray, evidence_root: Path, relative: str) -> None:
    output = safe_child(evidence_root, relative)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"could not write evidence overlay: {output}")
