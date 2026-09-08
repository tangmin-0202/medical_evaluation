from __future__ import annotations

from collections.abc import Mapping
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
from medical_evaluation.features.frame_reference import FrameReferenceStore
from medical_evaluation.features.mannequin_registration import (
    estimate_similarity_registration,
    estimate_static_scene_registration,
    invert_similarity,
    transform_points,
    warp_mask,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.segmentation.base import FrameMasks, VideoSegmenter
from medical_evaluation.segmentation.prompt_policy import (
    AnnotationPromptPolicy,
    Cp09Cp11PromptPolicy,
    TextPromptPolicy,
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
        template: Mapping[str, object] | None = None,
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
        self.template = dict(template or {})
        self.reference_store = FrameReferenceStore(evidence_root)

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

        if isinstance(self.prompt_policy, TextPromptPolicy):
            return self._extract_head_relative_final(
                video_path,
                checkpoint_id,
                time_range,
                final_start_sec=final_start_sec,
                dense_fps=dense_fps,
                presence=presence,
                final_presence=final_presence,
            )

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

    def _extract_head_relative_final(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        final_start_sec: float,
        dense_fps: float,
        presence: float,
        final_presence: float,
    ) -> ExtractedEvidence:
        empty: dict[str, float | bool | None] = {
            "dam_stage_presence_ratio": presence,
            "dam_final_presence_ratio": final_presence,
            "head_registration_reliable": False,
            "head_valid_count": 0.0,
            "frame_reference_available": False,
            "expected_frame_dam_coverage_ratio": None,
            "visible_frame_ratio": None,
            "nose_overlap": None,
            "visible_frame_area_ratio": None,
            "final_valid_frame_count": 0.0,
            "head_registration_candidate_count": 0.0,
            "head_registration_rejected_count": 0.0,
            "head_registration_low_mask_overlap_count": 0.0,
            "head_registration_low_match_count": 0.0,
        }
        reference = self.reference_store.load()
        if reference is None:
            return ExtractedEvidence(features=empty, evidence=[])
        empty["frame_reference_available"] = True
        try:
            anchor_image, anchor_head, anchor_frame = self.reference_store.read_artifacts(
                reference
            )
        except ValueError:
            return ExtractedEvidence(features=empty, evidence=[])
        final_range = TimeRange(start_sec=final_start_sec, end_sec=time_range.end_sec)
        head_frames = list(
            self.segmenter.track(
                video_path,
                final_range,
                self.prompt_policy.head_prompts(
                    self.annotations, final_range, checkpoint_id=checkpoint_id
                ),
                sample_fps=dense_fps,
            )
        )
        dam_frames = list(
            self.segmenter.track(
                video_path,
                final_range,
                self.prompt_policy.dam_prompts(
                    self.annotations, final_range, checkpoint_id=checkpoint_id
                ),
                sample_fps=dense_fps,
            )
        )
        dam_by_index = {item.frame_index: item for item in dam_frames}
        polygon = self._template_nose_polygon(anchor_head.shape)
        anchor_lab = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2LAB)
        reference_lab = np.median(anchor_lab[anchor_frame], axis=0).astype(float)
        measurements: list[
            tuple[FrameMasks, np.ndarray, np.ndarray, np.ndarray, float, float, float]
        ] = []
        reliable_count = 0
        rejected_count = 0
        low_mask_overlap_count = 0
        low_match_count = 0
        for head_item in head_frames:
            head_mask = head_item.masks.get("mannequin_head")
            dam_item = dam_by_index.get(head_item.frame_index)
            dam_mask = dam_item.masks.get("rubber_dam") if dam_item is not None else None
            if head_mask is None or dam_mask is None:
                continue
            image = read_frame(video_path, head_item.frame_index)
            debug_head = safe_child(
                self.evidence_root,
                f"cp_11/debug/head/{head_item.frame_index:08d}.png",
            )
            debug_head.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(debug_head), np.asarray(head_mask, dtype=np.uint8) * 255
            ):
                raise OSError(f"could not write CP11 debug head mask: {debug_head}")
            registration = estimate_similarity_registration(
                image,
                np.asarray(head_mask, dtype=bool),
                anchor_image,
                anchor_head,
            )
            if not registration.accepted:
                registration = estimate_static_scene_registration(
                    image,
                    np.asarray(head_mask, dtype=bool),
                    anchor_image,
                    anchor_head,
                )
            if not registration.accepted:
                rejected_count += 1
                low_mask_overlap_count += int(registration.reason == "low_mask_overlap")
                low_match_count += int(
                    registration.reason == "insufficient_feature_matches"
                )
                continue
            anchor_to_current = invert_similarity(registration.matrix)
            reliable_count += 1
            projected_frame = warp_mask(
                anchor_frame,
                anchor_to_current,
                output_shape=np.asarray(dam_mask).shape,
            )
            projected_nose_points = transform_points(polygon, anchor_to_current)
            nose_mask = np.zeros_like(projected_frame, dtype=np.uint8)
            cv2.fillPoly(
                nose_mask,
                [np.rint(projected_nose_points).astype(np.int32)],
                1,
            )
            dam = np.asarray(dam_mask, dtype=bool) & green_dam_mask(image)
            frame_area = max(1, int(projected_frame.sum()))
            nose_area = max(1, int(nose_mask.sum()))
            coverage = float(np.count_nonzero(dam & projected_frame) / frame_area)
            nose_overlap = float(np.count_nonzero(dam & nose_mask.astype(bool)) / nose_area)
            visible_frame = visible_reference_mask(
                image,
                projected_frame,
                reference_lab,
                max_lab_distance=self.frame_appearance_max_lab_distance,
            )
            visible_ratio = float(visible_frame.sum() / frame_area)
            measurements.append(
                (
                    head_item,
                    dam,
                    projected_frame,
                    nose_mask.astype(bool),
                    coverage,
                    nose_overlap,
                    visible_ratio,
                )
            )
        empty["head_valid_count"] = float(reliable_count)
        empty["head_registration_candidate_count"] = float(
            reliable_count + rejected_count
        )
        empty["head_registration_rejected_count"] = float(rejected_count)
        empty["head_registration_low_mask_overlap_count"] = float(
            low_mask_overlap_count
        )
        empty["head_registration_low_match_count"] = float(low_match_count)
        empty["head_registration_reliable"] = reliable_count >= self.minimum_final_frames
        empty["final_valid_frame_count"] = float(len(measurements))
        if len(measurements) < self.minimum_final_frames:
            return ExtractedEvidence(features=empty, evidence=[])
        coverage = float(np.median([item[4] for item in measurements]))
        nose_overlap = float(np.median([item[5] for item in measurements]))
        visible_ratio = float(np.median([item[6] for item in measurements]))
        evidence = self._write_head_relative_evidence(video_path, measurements)
        return ExtractedEvidence(
            features={
                **empty,
                "head_registration_reliable": True,
                "expected_frame_dam_coverage_ratio": coverage,
                "visible_frame_ratio": visible_ratio,
                "nose_overlap": nose_overlap,
                "visible_frame_area_ratio": 1.0 - coverage,
            },
            evidence=evidence,
        )

    def _template_nose_polygon(self, shape: tuple[int, int]) -> np.ndarray:
        polygon = self.template.get("nose_polygon_normalized")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError("mannequin template requires nose_polygon_normalized")
        height, width = shape
        return np.asarray(
            [[float(point[0]) * width, float(point[1]) * height] for point in polygon],
            dtype=np.float64,
        )

    def _write_head_relative_evidence(
        self,
        video_path: Path,
        measurements: list[
            tuple[FrameMasks, np.ndarray, np.ndarray, np.ndarray, float, float, float]
        ],
    ) -> list[EvidenceItem]:
        selected = sorted({0, len(measurements) // 2, len(measurements) - 1})
        evidence: list[EvidenceItem] = []
        for position in selected:
            (
                item,
                dam,
                projected_frame,
                nose,
                _coverage,
                _nose_overlap,
                _visible_ratio,
            ) = measurements[position]
            canvas = read_frame(video_path, item.frame_index).copy()
            for mask, color, thickness in (
                (dam, (0, 255, 0), 2),
                (projected_frame, (255, 255, 255), 3),
                (nose, (0, 220, 255), 3),
            ):
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(canvas, contours, -1, color, thickness)
            output = safe_child(
                self.evidence_root, f"cp_11/overlays/{item.frame_index:08d}.jpg"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), canvas):
                raise OSError(f"could not write evidence overlay: {output}")
            evidence.append(
                EvidenceItem(
                    time_sec=item.frame_time_sec,
                    overlay_path=output.relative_to(self.evidence_root).as_posix(),
                    rule="cp09_frame_covered_and_template_nose_clear",
                )
            )
        return evidence

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
