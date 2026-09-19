"""Cross-stage CP08 evidence extraction.

The extractor uses only stage time ranges from annotations.  Every model
prompt is text-only and every object is tracked in an independent session.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.features.cp08_positioning import (
    ClampWingSplit,
    InstrumentShapeMeasurement,
    WingHoleColorMeasurement,
    measure_instrument_shape,
    measure_wing_hole_color,
    split_clamp_wings,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.segmentation.base import (
    FrameMasks,
    SegmentationPrompt,
    VideoSegmenter,
)
from medical_evaluation.segmentation.sam3_backend import Sam3AmbiguousTextResult
from medical_evaluation.storage import atomic_write_json, safe_child
from medical_evaluation.video import read_frame, sample_frames

INSTRUMENT_PROMPT = "metal dental instrument with a long handle and curved working shaft"
TOOTH_PROMPT = "target tooth enclosed by the metal rubber dam clamp"
CLAMP_PROMPT = "metal rubber dam clamp around the tooth"
DAM_PROMPT = "large green sheet covering the mouth area"


@dataclass(frozen=True)
class _FinalMeasurement:
    item: FrameMasks
    tooth: np.ndarray
    clamp: np.ndarray
    dam: np.ndarray
    split: ClampWingSplit
    left_hole: WingHoleColorMeasurement | None
    right_hole: WingHoleColorMeasurement | None


@dataclass(frozen=True)
class _DamCrossCheck:
    sam_present: bool
    local_green_ratio: float | None
    local_green_state: bool | None
    dam_green_ratio: float | None
    dam_green_state: bool | None
    conflict: bool
    agreed_state: bool | None


@dataclass(frozen=True)
class _DamGateFrame:
    item: FrameMasks
    frame: np.ndarray
    mask: np.ndarray | None
    frame_green_ratio: float
    mask_green_ratio: float | None
    state: bool | None


class Cp08FeatureExtractor:
    sparse_fps = 1.0
    minimum_dense_fps = 5.0
    dense_window_sec = 1.0
    final_window_sec = 3.0
    minimum_valid_frames = 3
    minimum_candidate_elongation = 3.0
    minimum_candidate_area_px = 25
    local_green_present_ratio = 0.12
    local_green_absent_ratio = 0.03
    dam_green_present_ratio = 0.50
    dam_green_absent_ratio = 0.15
    minimum_green_sample_px = 64

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
        return f"{self.segmenter.model_version}+cp08-opencv-v1"

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id != "cp_08":
            raise ValueError("Cp08FeatureExtractor is CP08-only")
        if analysis_width <= 0:
            raise ValueError("analysis_width must be positive")
        if dense_fps <= 0:
            raise ValueError("dense_fps must be positive")
        effective_dense_fps = max(dense_fps, self.minimum_dense_fps)

        sparse = list(
            self.segmenter.track(
                video_path,
                time_range,
                [
                    self._text_prompt(
                        "cp08_instrument", INSTRUMENT_PROMPT, time_range.end_sec
                    )
                ],
                sample_fps=self.sparse_fps,
            )
        )
        if not sparse:
            sparse = [
                FrameMasks(
                    frame_index=item.frame_index,
                    frame_time_sec=item.time_sec,
                    masks={},
                )
                for item in sample_frames(
                    video_path,
                    start_sec=time_range.start_sec,
                    end_sec=time_range.end_sec,
                    sample_fps=self.sparse_fps,
                )
            ]
        sparse_masks = [
            (item, self._mask(item, "cp08_instrument")) for item in sparse
        ]
        observed = any(mask is not None for _item, mask in sparse_masks)
        candidates: list[FrameMasks] = []
        for item, mask in sparse_masks:
            if mask is None:
                continue
            measurement = measure_instrument_shape(mask)
            if not self._plausible_instrument_candidate(measurement):
                continue
            candidates.append(item)
        dense_sessions: list[list[FrameMasks]] = []
        attempted_ranges: list[TimeRange] = []
        for candidate in candidates:
            if any(
                attempted.start_sec <= candidate.frame_time_sec < attempted.end_sec
                for attempted in attempted_ranges
            ):
                continue
            dense_range = TimeRange(
                start_sec=max(time_range.start_sec, candidate.frame_time_sec - self.dense_window_sec),
                end_sec=min(time_range.end_sec, candidate.frame_time_sec + self.dense_window_sec),
            )
            attempted_ranges.append(dense_range)
            dense = list(
                self.segmenter.track(
                    video_path,
                    dense_range,
                    [
                        self._text_prompt(
                            "cp08_instrument",
                            INSTRUMENT_PROMPT,
                            dense_range.end_sec,
                        )
                    ],
                    sample_fps=effective_dense_fps,
                )
            )
            dense_sessions.append(dense)
            measurements = [
                measure_instrument_shape(mask)
                for item in dense
                if (mask := self._mask(item, "cp08_instrument")) is not None
            ]
            if self._shape_consensus(measurements) is True:
                break
        action_features, action_evidence = self._extract_action(
            video_path, sparse=sparse, dense_sessions=dense_sessions, observed=observed
        )
        if not (
            action_features["instrument_observed"] is True
            and action_features["instrument_shape_reliable"] is True
            and action_features["instrument_shape_match"] is True
        ):
            return ExtractedEvidence(
                features=action_features,
                evidence=action_evidence,
            )

        final_range = self._cp09_final_range()
        final_frames: dict[str, list[FrameMasks]] = {}
        final_errors: dict[str, str] = {}
        try:
            final_frames["dam"] = list(
                self.segmenter.track(
                    video_path,
                    final_range,
                    [
                        self._text_prompt(
                            "cp08_rubber_dam", DAM_PROMPT, final_range.end_sec
                        )
                    ],
                    sample_fps=effective_dense_fps,
                )
            )
        except Sam3AmbiguousTextResult:
            final_frames["dam"] = []
            final_errors["dam"] = "ambiguous_unscored_candidates"
        dam_state, dam_valid_count, dam_gate_frames = self._precheck_dam(
            video_path,
            final_range,
            final_frames["dam"],
            sample_fps=effective_dense_fps,
        )
        if dam_state is False:
            dam_evidence = self._write_dam_gate_evidence(
                dam_gate_frames, failure_reasons=["rubber_dam_not_positioned"]
            )
            return ExtractedEvidence(
                features={
                    **action_features,
                    "final_state_observable": True,
                    "rubber_dam_positioned": False,
                    "final_state_valid_frame_count": float(dam_valid_count),
                    "rubber_dam_segmentation_conflict": False,
                    "dam_color_conflict_frame_count": 0.0,
                    "dam_color_conflict_frame_ratio": 0.0,
                },
                evidence=[*action_evidence, *dam_evidence],
            )
        final_objects = {
            "tooth": ("cp08_target_tooth", TOOTH_PROMPT),
            "clamp": ("cp08_full_clamp", CLAMP_PROMPT),
        }
        for name, (object_id, text) in final_objects.items():
            try:
                final_frames[name] = list(
                    self.segmenter.track(
                        video_path,
                        final_range,
                        [self._text_prompt(object_id, text, final_range.end_sec)],
                        sample_fps=effective_dense_fps,
                    )
                )
            except Sam3AmbiguousTextResult:
                final_frames[name] = []
                final_errors[name] = "ambiguous_unscored_candidates"
        final_features, final_evidence = self._extract_final(
            video_path, final_frames, segmentation_errors=final_errors
        )
        return ExtractedEvidence(
            features={**action_features, **final_features},
            evidence=[*action_evidence, *final_evidence],
        )

    def _precheck_dam(
        self,
        video_path: Path,
        time_range: TimeRange,
        sam_frames: list[FrameMasks],
        *,
        sample_fps: float,
    ) -> tuple[bool | None, int, list[_DamGateFrame]]:
        sources = list(sam_frames)
        decoded: dict[int, np.ndarray] = {}
        if not sources:
            for sampled in sample_frames(
                video_path,
                start_sec=time_range.start_sec,
                end_sec=time_range.end_sec,
                sample_fps=sample_fps,
            ):
                sources.append(
                    FrameMasks(
                        frame_index=sampled.frame_index,
                        frame_time_sec=sampled.time_sec,
                        masks={},
                    )
                )
                decoded[sampled.frame_index] = sampled.image_bgr
        measured: list[_DamGateFrame] = []
        for item in sources:
            frame = decoded.get(item.frame_index)
            if frame is None:
                frame = read_frame(video_path, item.frame_index)
            green = self._green_pixels(frame)
            frame_ratio = float(green.mean())
            mask = self._mask(item, "cp08_rubber_dam")
            mask_ratio = self._region_ratio(green, mask) if mask is not None else None
            state: bool | None = None
            if mask is not None and mask_ratio is not None:
                if mask_ratio >= self.dam_green_present_ratio:
                    state = True
            elif frame_ratio <= self.local_green_absent_ratio:
                state = False
            measured.append(
                _DamGateFrame(item, frame, mask, frame_ratio, mask_ratio, state)
            )
        valid = [item.state for item in measured if item.state is not None]
        if len(valid) < self.minimum_valid_frames:
            return None, len(valid), measured
        true_ratio = float(np.mean(valid))
        if true_ratio >= 2 / 3:
            return True, len(valid), measured
        if true_ratio <= 1 / 3:
            return False, len(valid), measured
        return None, len(valid), measured

    def _write_dam_gate_evidence(
        self,
        values: list[_DamGateFrame],
        *,
        failure_reasons: list[str],
    ) -> list[EvidenceItem]:
        rows: list[dict[str, Any]] = []
        evidence: list[EvidenceItem] = []
        for value in values:
            mask = (
                np.zeros(value.frame.shape[:2], dtype=bool)
                if value.mask is None
                else value.mask
            )
            paths = self._write_frame_bundle(
                "final",
                value.item,
                value.frame,
                {"dam": mask},
                ((mask, (0, 255, 0)),),
                artifact_id=f"dam-gate-{value.item.frame_index:08d}",
            )
            rows.append(
                {
                    "frame_index": value.item.frame_index,
                    "time_sec": value.item.frame_time_sec,
                    "sam_dam_present": value.mask is not None,
                    "frame_green_ratio": value.frame_green_ratio,
                    "local_green_ratio": value.frame_green_ratio,
                    "local_green_state": value.state,
                    "dam_green_ratio": value.mask_green_ratio,
                    "dam_green_state": value.state if value.mask is not None else None,
                    "dam_color_conflict": False,
                    "dam_presence_state": value.state,
                    "rejection_reason": (
                        None if value.state is not None else "dam_presence_unreliable"
                    ),
                }
            )
            evidence.append(
                EvidenceItem(
                    time_sec=value.item.frame_time_sec,
                    overlay_path=paths["overlay"].relative_to(
                        self.evidence_root
                    ).as_posix(),
                    rule="cp08_cp09_tail_rubber_dam_presence_gate",
                )
            )
        self._write_metrics(
            "final",
            selected_prompts={"dam": DAM_PROMPT},
            rows=rows,
            failure_reasons=failure_reasons,
        )
        return evidence

    @staticmethod
    def _text_prompt(
        object_id: str,
        text: str,
        frame_time_sec: float,
    ) -> SegmentationPrompt:
        return SegmentationPrompt(
            object_id=object_id,
            kind="text",
            frame_time_sec=frame_time_sec,
            text=text,
        )

    @staticmethod
    def _mask(item: FrameMasks, object_id: str) -> np.ndarray | None:
        value = item.masks.get(object_id)
        if value is None:
            return None
        mask = np.asarray(value, dtype=bool)
        return mask if mask.ndim == 2 and mask.any() else None

    def _plausible_instrument_candidate(
        self, measurement: InstrumentShapeMeasurement
    ) -> bool:
        return bool(
            measurement.reliable
            and measurement.component_mask is not None
            and measurement.selected_component_area_px >= self.minimum_candidate_area_px
            and measurement.elongation is not None
            and measurement.elongation >= self.minimum_candidate_elongation
        )

    def _cp09_final_range(self) -> TimeRange:
        cp09 = next(
            (step.time_range for step in self.annotations.steps if step.checkpoint_id == "cp_09"),
            None,
        )
        if cp09 is None:
            raise ValueError("CP08 extraction requires a CP09 stage time range")
        return TimeRange(
            start_sec=max(cp09.start_sec, cp09.end_sec - self.final_window_sec),
            end_sec=cp09.end_sec,
        )

    def _extract_action(
        self,
        video_path: Path,
        *,
        sparse: list[FrameMasks],
        dense_sessions: list[list[FrameMasks]],
        observed: bool,
    ) -> tuple[dict[str, float | bool | None], list[EvidenceItem]]:
        metrics: list[
            tuple[
                str,
                str,
                FrameMasks,
                np.ndarray | None,
                InstrumentShapeMeasurement | None,
            ]
        ] = []
        sessions = [("sparse", "sparse-00", sparse)] + [
            ("dense", f"dense-{index:02d}", values)
            for index, values in enumerate(dense_sessions, start=1)
        ]
        for phase, session_id, values in sessions:
            for item in values:
                mask = self._mask(item, "cp08_instrument")
                measurement = measure_instrument_shape(mask) if mask is not None else None
                metrics.append((phase, session_id, item, mask, measurement))
        reliable = [
            item
            for phase, _session, _frame, _mask, item in metrics
            if phase == "dense" and item is not None and item.reliable
        ]
        session_consensus = []
        for index in range(1, len(dense_sessions) + 1):
            session_id = f"dense-{index:02d}"
            session_consensus.append(
                self._shape_consensus(
                    [
                        item
                        for phase, session, _frame, _mask, item in metrics
                        if phase == "dense"
                        and session == session_id
                        and item is not None
                    ]
                )
            )
        shape_reliable = any(value is not None for value in session_consensus)
        match_count = sum(item.matches_proxy is True for item in reliable)
        shape_match: bool | None = None
        if any(value is True for value in session_consensus):
            shape_match = True
        elif any(value is False for value in session_consensus):
            shape_match = False
        failure_reasons: list[str] = []
        if not observed:
            failure_reasons.append("instrument_not_observed_in_full_sparse_scan")
        elif not dense_sessions:
            failure_reasons.append("no_shape_plausible_sparse_candidate")
        elif not shape_reliable:
            failure_reasons.append("insufficient_reliable_instrument_frames")
        elif shape_match is None:
            failure_reasons.append("instrument_shape_temporal_disagreement")
        evidence = self._write_action_evidence(video_path, metrics, failure_reasons)
        return (
            {
                "instrument_observed": observed,
                "instrument_shape_reliable": shape_reliable,
                "instrument_shape_match": shape_match,
                "instrument_valid_frame_count": float(len(reliable)),
                "instrument_matching_frame_count": float(match_count),
            },
            evidence,
        )

    def _shape_consensus(
        self, measurements: list[InstrumentShapeMeasurement]
    ) -> bool | None:
        reliable = [item for item in measurements if item.reliable]
        if len(reliable) < self.minimum_valid_frames:
            return None
        required = max(self.minimum_valid_frames, int(np.ceil(len(reliable) * 2 / 3)))
        if sum(item.matches_proxy is True for item in reliable) >= required:
            return True
        if sum(item.matches_proxy is False for item in reliable) >= required:
            return False
        return None

    def _extract_final(
        self,
        video_path: Path,
        frames: dict[str, list[FrameMasks]],
        *,
        segmentation_errors: dict[str, str] | None = None,
    ) -> tuple[dict[str, float | bool | None], list[EvidenceItem]]:
        segmentation_errors = segmentation_errors or {}
        object_ids = {
            "tooth": "cp08_target_tooth",
            "clamp": "cp08_full_clamp",
            "dam": "cp08_rubber_dam",
        }
        indexed = {
            name: {item.frame_index: item for item in values}
            for name, values in frames.items()
        }
        all_indices = sorted(set().union(*(set(items) for items in indexed.values())))
        observable_indices = [
            index
            for index in all_indices
            if self._mask_from_index(indexed["tooth"], index, object_ids["tooth"]) is not None
            and self._mask_from_index(indexed["clamp"], index, object_ids["clamp"]) is not None
        ]
        final_observable = len(observable_indices) >= self.minimum_valid_frames
        frame_cache = {
            index: read_frame(video_path, index) for index in observable_indices
        }
        cross_checks: dict[int, _DamCrossCheck] = {}
        for index in observable_indices:
            tooth = self._mask_from_index(indexed["tooth"], index, object_ids["tooth"])
            clamp = self._mask_from_index(indexed["clamp"], index, object_ids["clamp"])
            dam = self._mask_from_index(indexed["dam"], index, object_ids["dam"])
            assert tooth is not None and clamp is not None
            cross_checks[index] = self._cross_check_dam(
                frame_cache[index], tooth, clamp, dam
            )
        rubber_dam_positioned: bool | None = None
        if final_observable:
            checks = [cross_checks[index] for index in observable_indices]
            present_count = sum(item.agreed_state is True for item in checks)
            absent_count = sum(item.agreed_state is False for item in checks)
            if (
                present_count >= self.minimum_valid_frames
                and present_count / len(checks) >= 2 / 3
            ):
                rubber_dam_positioned = True
            elif (
                absent_count >= self.minimum_valid_frames
                and absent_count / len(checks) >= 2 / 3
            ):
                rubber_dam_positioned = False

        measurements: list[_FinalMeasurement] = []
        for index in observable_indices:
            tooth = self._mask_from_index(indexed["tooth"], index, object_ids["tooth"])
            clamp = self._mask_from_index(indexed["clamp"], index, object_ids["clamp"])
            dam = self._mask_from_index(indexed["dam"], index, object_ids["dam"])
            if tooth is None or clamp is None or dam is None:
                continue
            split = split_clamp_wings(tooth, clamp)
            source = indexed["tooth"][index]
            frame = frame_cache[index]
            left = (
                measure_wing_hole_color(frame, split.left_wing, dam)
                if split.left_wing is not None
                else None
            )
            right = (
                measure_wing_hole_color(frame, split.right_wing, dam)
                if split.right_wing is not None
                else None
            )
            measurements.append(_FinalMeasurement(source, tooth, clamp, dam, split, left, right))

        features: dict[str, float | bool | None] = {
            "final_state_observable": final_observable,
            "rubber_dam_positioned": rubber_dam_positioned,
            "final_state_valid_frame_count": float(len(measurements)),
        }
        if segmentation_errors:
            features["final_segmentation_conflict"] = True
            features["final_segmentation_error_count"] = float(
                len(segmentation_errors)
            )
        conflict_count = sum(item.conflict for item in cross_checks.values())
        unresolved_conflict = conflict_count > 0 and rubber_dam_positioned is None
        features.update(
            {
                "rubber_dam_segmentation_conflict": unresolved_conflict,
                "dam_color_conflict_frame_count": float(conflict_count),
                "dam_color_conflict_frame_ratio": (
                    float(conflict_count / len(observable_indices))
                    if observable_indices
                    else None
                ),
            }
        )
        for side in ("left", "right"):
            wing_values = [
                item.split.left_complete if side == "left" else item.split.right_complete
                for item in measurements
                if item.split.reliable
            ]
            complete = self._boolean_consensus(wing_values)
            holes = [
                item.left_hole if side == "left" else item.right_hole
                for item in measurements
            ]
            valid_holes = [item for item in holes if item is not None and item.reliable]
            features.update(
                {
                    f"{side}_wing_complete": complete,
                    f"{side}_wing_valid_frame_count": float(len(wing_values)),
                    f"{side}_wing_complete_ratio": (
                        float(np.mean(wing_values)) if wing_values else None
                    ),
                    f"{side}_wing_hole_detected": (
                        True if len(valid_holes) >= self.minimum_valid_frames else None
                    ),
                    f"{side}_wing_hole_valid_frame_count": float(len(valid_holes)),
                    f"{side}_wing_hole_dam_color_ratio": self._median_ratio(
                        valid_holes, "dam_color_ratio"
                    ),
                    f"{side}_wing_hole_non_dam_color_ratio": self._median_ratio(
                        valid_holes, "non_dam_color_ratio"
                    ),
                }
            )
        failure_reasons: list[str] = []
        failure_reasons.extend(
            f"{name}_segmentation_{reason}"
            for name, reason in sorted(segmentation_errors.items())
        )
        if unresolved_conflict:
            failure_reasons.append("rubber_dam_color_conflict")
        if not final_observable:
            failure_reasons.append("insufficient_synchronized_tooth_and_clamp_frames")
        elif rubber_dam_positioned is None:
            failure_reasons.append("rubber_dam_temporal_evidence_unreliable")
        elif rubber_dam_positioned is False:
            failure_reasons.append("rubber_dam_not_positioned")
        if rubber_dam_positioned is True:
            reliable_splits = sum(item.split.reliable for item in measurements)
            if reliable_splits < self.minimum_valid_frames:
                failure_reasons.append("clamp_wing_split_unreliable")
            for side in ("left", "right"):
                complete = features[f"{side}_wing_complete"]
                if complete is False:
                    failure_reasons.append(f"{side}_wing_not_complete")
                elif complete is None:
                    failure_reasons.append(f"{side}_wing_consensus_unreliable")
                if features[f"{side}_wing_hole_detected"] is not True:
                    failure_reasons.append(f"{side}_wing_hole_unreliable")
        evidence = self._write_final_evidence(
            video_path,
            indexed,
            all_indices,
            measurements,
            cross_checks,
            frame_cache,
            failure_reasons,
        )
        return features, evidence

    def _cross_check_dam(
        self,
        frame: np.ndarray,
        tooth: np.ndarray,
        clamp: np.ndarray,
        dam: np.ndarray | None,
    ) -> _DamCrossCheck:
        reference = np.asarray(tooth, dtype=bool) | np.asarray(clamp, dtype=bool)
        ys, xs = np.nonzero(reference)
        if not len(xs):
            return _DamCrossCheck(dam is not None, None, None, None, None, False, None)
        height, width = reference.shape
        span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        padding = max(8, round(0.35 * span))
        x1 = max(0, int(xs.min()) - padding)
        x2 = min(width, int(xs.max()) + padding + 1)
        y1 = max(0, int(ys.min()) - padding)
        y2 = min(height, int(ys.max()) + padding + 1)
        local = np.zeros(reference.shape, dtype=bool)
        local[y1:y2, x1:x2] = True
        excluded = cv2.dilate(reference.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
        local &= ~excluded.astype(bool)
        green = self._green_pixels(frame)
        local_ratio = self._region_ratio(green, local)
        local_state = self._ratio_state(
            local_ratio, self.local_green_present_ratio, self.local_green_absent_ratio
        )
        sam_present = dam is not None
        dam_ratio = None
        dam_state = None
        if dam is not None:
            dam_region = np.asarray(dam, dtype=bool) & ~excluded.astype(bool)
            dam_ratio = self._region_ratio(green, dam_region)
            dam_state = self._ratio_state(
                dam_ratio, self.dam_green_present_ratio, self.dam_green_absent_ratio
            )
        conflict = bool(
            (not sam_present and local_state is True)
            or (
                sam_present
                and (local_state is False or dam_state is False)
            )
        )
        agreed_state: bool | None = None
        if sam_present and local_state is True and dam_state is True:
            agreed_state = True
        elif not sam_present and local_state is False:
            agreed_state = False
        return _DamCrossCheck(
            sam_present,
            local_ratio,
            local_state,
            dam_ratio,
            dam_state,
            conflict,
            agreed_state,
        )

    @staticmethod
    def _green_pixels(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_BGR2HSV)
        return (
            (hsv[..., 0] >= 30)
            & (hsv[..., 0] <= 100)
            & (hsv[..., 1] >= 50)
            & (hsv[..., 2] >= 25)
        )

    def _region_ratio(self, green: np.ndarray, region: np.ndarray) -> float | None:
        area = int(np.count_nonzero(region))
        if area < self.minimum_green_sample_px:
            return None
        return float(np.count_nonzero(green & region) / area)

    @staticmethod
    def _ratio_state(
        ratio: float | None, present_threshold: float, absent_threshold: float
    ) -> bool | None:
        if ratio is None:
            return None
        if ratio >= present_threshold:
            return True
        if ratio <= absent_threshold:
            return False
        return None

    @staticmethod
    def _mask_from_index(
        values: dict[int, FrameMasks], index: int, object_id: str
    ) -> np.ndarray | None:
        item = values.get(index)
        return None if item is None else Cp08FeatureExtractor._mask(item, object_id)

    def _boolean_consensus(self, values: list[bool]) -> bool | None:
        if len(values) < self.minimum_valid_frames:
            return None
        true_ratio = float(np.mean(values))
        if true_ratio >= 2 / 3:
            return True
        if true_ratio <= 1 / 3:
            return False
        return None

    def _median_ratio(
        self, values: list[WingHoleColorMeasurement], field: str
    ) -> float | None:
        ratios = [getattr(item, field) for item in values if getattr(item, field) is not None]
        if len(ratios) < self.minimum_valid_frames:
            return None
        return float(np.median(ratios))

    def _write_action_evidence(
        self,
        video_path: Path,
        values: list[
            tuple[
                str,
                str,
                FrameMasks,
                np.ndarray | None,
                InstrumentShapeMeasurement | None,
            ]
        ],
        failure_reasons: list[str],
    ) -> list[EvidenceItem]:
        rows: list[dict[str, Any]] = []
        evidence: list[EvidenceItem] = []
        frame_cache: dict[int, np.ndarray] = {}
        for phase, session_id, item, mask, measurement in values:
            if item.frame_index not in frame_cache:
                frame_cache[item.frame_index] = read_frame(video_path, item.frame_index)
            frame = frame_cache[item.frame_index]
            raw = np.zeros(frame.shape[:2], dtype=bool) if mask is None else mask
            selected = (
                np.zeros(frame.shape[:2], dtype=bool)
                if measurement is None or measurement.component_mask is None
                else np.asarray(measurement.component_mask, dtype=bool)
            )
            paths = self._write_frame_bundle(
                "action",
                item,
                frame,
                {"instrument_raw": raw, "instrument_selected": selected},
                ((selected, (0, 0, 255)),),
                artifact_id=f"{phase}-{session_id}-{item.frame_index:08d}",
            )
            rejection_reasons: list[str] = []
            if measurement is None:
                rejection_reasons.append("instrument_mask_missing")
            elif not measurement.reliable or measurement.matches_proxy is not True:
                rejection_reasons.append(measurement.reason)
            rows.append(
                {
                    "frame_index": item.frame_index,
                    "time_sec": item.frame_time_sec,
                    "scan_phase": phase,
                    "session_id": session_id,
                    "shape_observed": measurement.observed if measurement else False,
                    "shape_reliable": measurement.reliable if measurement else False,
                    "shape_match": measurement.matches_proxy if measurement else None,
                    "component_count": (
                        measurement.input_component_count if measurement else 0
                    ),
                    "selected_component_area_px": (
                        measurement.selected_component_area_px if measurement else 0
                    ),
                    "elongation": measurement.elongation if measurement else None,
                    "handle_to_shaft_width_ratio": (
                        measurement.handle_to_shaft_width_ratio if measurement else None
                    ),
                    "curvature_deg": measurement.curvature_deg if measurement else None,
                    "measurement_reason": measurement.reason if measurement else None,
                    "rejection_reason": (
                        rejection_reasons[0] if rejection_reasons else None
                    ),
                    "rejection_reasons": rejection_reasons,
                    "original_path": paths["original"].relative_to(
                        self.evidence_root
                    ).as_posix(),
                    "raw_mask_path": paths["instrument_raw"].relative_to(
                        self.evidence_root
                    ).as_posix(),
                    "selected_mask_path": paths["instrument_selected"].relative_to(
                        self.evidence_root
                    ).as_posix(),
                    "overlay_path": paths["overlay"].relative_to(
                        self.evidence_root
                    ).as_posix(),
                }
            )
            evidence.append(
                EvidenceItem(
                    time_sec=item.frame_time_sec,
                    overlay_path=paths["overlay"].relative_to(
                        self.evidence_root
                    ).as_posix(),
                    rule="cp08_instrument_shape_proxy",
                )
            )
        self._write_metrics(
            "action",
            selected_prompts={"instrument": INSTRUMENT_PROMPT},
            rows=rows,
            failure_reasons=failure_reasons,
        )
        return evidence

    def _write_final_evidence(
        self,
        video_path: Path,
        indexed: dict[str, dict[int, FrameMasks]],
        indices: list[int],
        measurements: list[_FinalMeasurement],
        cross_checks: dict[int, _DamCrossCheck],
        frame_cache: dict[int, np.ndarray],
        failure_reasons: list[str],
    ) -> list[EvidenceItem]:
        measured = {item.item.frame_index: item for item in measurements}
        rows: list[dict[str, Any]] = []
        evidence: list[EvidenceItem] = []
        object_ids = {
            "tooth": "cp08_target_tooth",
            "clamp": "cp08_full_clamp",
            "dam": "cp08_rubber_dam",
        }
        for index in indices:
            source = next((values[index] for values in indexed.values() if index in values), None)
            if source is None:
                continue
            frame = frame_cache.get(index)
            if frame is None:
                frame = read_frame(video_path, index)
                frame_cache[index] = frame
            masks = {
                name: self._mask_from_index(indexed[name], index, object_id)
                for name, object_id in object_ids.items()
            }
            effective = {
                name: np.zeros(frame.shape[:2], dtype=bool) if mask is None else mask
                for name, mask in masks.items()
            }
            item = measured.get(index)
            derived = {
                "left_wing": item.split.left_wing if item else None,
                "right_wing": item.split.right_wing if item else None,
                "left_hole": item.left_hole.hole_mask if item and item.left_hole else None,
                "right_hole": item.right_hole.hole_mask if item and item.right_hole else None,
                "left_sample": item.left_hole.sample_mask if item and item.left_hole else None,
                "right_sample": item.right_hole.sample_mask if item and item.right_hole else None,
            }
            for name, mask in derived.items():
                effective[name] = (
                    np.zeros(frame.shape[:2], dtype=bool)
                    if mask is None
                    else np.asarray(mask, dtype=bool)
                )
            paths = self._write_frame_bundle(
                "final",
                source,
                frame,
                effective,
                (
                    (effective["tooth"], (0, 0, 255)),
                    (effective["clamp"], (255, 255, 255)),
                    (effective["dam"], (0, 255, 0)),
                    (effective["left_wing"], (255, 0, 0)),
                    (effective["right_wing"], (255, 0, 255)),
                    (effective["left_hole"], (0, 255, 255)),
                    (effective["right_hole"], (0, 165, 255)),
                    (effective["left_sample"], (255, 255, 0)),
                    (effective["right_sample"], (128, 255, 0)),
                ),
                artifact_id=f"{index:08d}",
            )
            rejection_reasons: list[str] = []
            if masks["tooth"] is None:
                rejection_reasons.append("tooth_mask_missing")
            if masks["clamp"] is None:
                rejection_reasons.append("clamp_mask_missing")
            if masks["dam"] is None:
                rejection_reasons.append("dam_mask_missing")
            if item is not None:
                if not item.split.reliable:
                    rejection_reasons.append(
                        f"clamp_wing_split_unreliable:{item.split.reason}"
                    )
                else:
                    for side in ("left", "right"):
                        wing = (
                            item.split.left_wing if side == "left" else item.split.right_wing
                        )
                        complete = (
                            item.split.left_complete
                            if side == "left"
                            else item.split.right_complete
                        )
                        if wing is None:
                            rejection_reasons.append(f"{side}_wing_missing")
                        elif not complete:
                            rejection_reasons.append(f"{side}_wing_not_complete")
                        hole = item.left_hole if side == "left" else item.right_hole
                        if hole is None:
                            rejection_reasons.append(f"{side}_hole_not_measured")
                        elif not hole.reliable:
                            rejection_reasons.append(f"{side}_{hole.reason}")
            rows.append(
                {
                    "frame_index": index,
                    "time_sec": source.frame_time_sec,
                    "left_wing_complete": item.split.left_complete if item else None,
                    "right_wing_complete": item.split.right_complete if item else None,
                    "left_wing_valid": bool(
                        item and item.split.reliable and item.split.left_wing is not None
                    ),
                    "right_wing_valid": bool(
                        item and item.split.reliable and item.split.right_wing is not None
                    ),
                    "wing_split_reliable": item.split.reliable if item else False,
                    "wing_split_reason": item.split.reason if item else None,
                    "left_wing_completeness_score": (
                        item.split.left_completeness_score if item else None
                    ),
                    "right_wing_completeness_score": (
                        item.split.right_completeness_score if item else None
                    ),
                    "left_hole_detected": (
                        item.left_hole.hole_detected if item and item.left_hole else False
                    ),
                    "right_hole_detected": (
                        item.right_hole.hole_detected if item and item.right_hole else False
                    ),
                    "left_hole_reliable": (
                        item.left_hole.reliable if item and item.left_hole else False
                    ),
                    "right_hole_reliable": (
                        item.right_hole.reliable if item and item.right_hole else False
                    ),
                    "left_hole_reason": item.left_hole.reason if item and item.left_hole else None,
                    "right_hole_reason": item.right_hole.reason if item and item.right_hole else None,
                    "left_hole_dam_color_ratio": (
                        item.left_hole.dam_color_ratio if item and item.left_hole else None
                    ),
                    "right_hole_dam_color_ratio": (
                        item.right_hole.dam_color_ratio if item and item.right_hole else None
                    ),
                    "left_hole_non_dam_color_ratio": (
                        item.left_hole.non_dam_color_ratio if item and item.left_hole else None
                    ),
                    "right_hole_non_dam_color_ratio": (
                        item.right_hole.non_dam_color_ratio if item and item.right_hole else None
                    ),
                    "left_hole_area_px": (
                        item.left_hole.hole_area_px if item and item.left_hole else 0
                    ),
                    "right_hole_area_px": (
                        item.right_hole.hole_area_px if item and item.right_hole else 0
                    ),
                    "left_sample_area_px": (
                        item.left_hole.sample_area_px if item and item.left_hole else 0
                    ),
                    "right_sample_area_px": (
                        item.right_hole.sample_area_px if item and item.right_hole else 0
                    ),
                    "sam_dam_present": (
                        cross_checks[index].sam_present if index in cross_checks else False
                    ),
                    "local_green_ratio": (
                        cross_checks[index].local_green_ratio
                        if index in cross_checks
                        else None
                    ),
                    "local_green_state": (
                        cross_checks[index].local_green_state
                        if index in cross_checks
                        else None
                    ),
                    "dam_green_ratio": (
                        cross_checks[index].dam_green_ratio
                        if index in cross_checks
                        else None
                    ),
                    "dam_green_state": (
                        cross_checks[index].dam_green_state
                        if index in cross_checks
                        else None
                    ),
                    "dam_color_conflict": (
                        cross_checks[index].conflict if index in cross_checks else False
                    ),
                    "rejection_reason": (
                        rejection_reasons[0] if rejection_reasons else None
                    ),
                    "rejection_reasons": rejection_reasons,
                }
            )
            evidence.append(
                EvidenceItem(
                    time_sec=source.frame_time_sec,
                    overlay_path=paths["overlay"].relative_to(
                        self.evidence_root
                    ).as_posix(),
                    rule="cp08_cp09_tail_clamp_wings_and_hole_colour",
                )
            )
        self._write_metrics(
            "final",
            selected_prompts={
                "tooth": TOOTH_PROMPT,
                "clamp": CLAMP_PROMPT,
                "dam": DAM_PROMPT,
            },
            rows=rows,
            failure_reasons=failure_reasons,
        )
        return evidence

    def _write_frame_bundle(
        self,
        group: str,
        item: FrameMasks,
        frame: np.ndarray,
        masks: dict[str, np.ndarray],
        contours: tuple[tuple[np.ndarray, tuple[int, int, int]], ...],
        *,
        artifact_id: str,
    ) -> dict[str, Path]:
        original = safe_child(
            self.evidence_root, f"cp_08/{group}/originals/{artifact_id}.jpg"
        )
        original.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(original), frame):
            raise OSError(f"could not write CP08 original: {original}")
        outputs: dict[str, Path] = {"original": original}
        for name, mask in masks.items():
            output = safe_child(
                self.evidence_root,
                f"cp_08/{group}/masks/{name}/{artifact_id}.png",
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), np.asarray(mask, dtype=np.uint8) * 255):
                raise OSError(f"could not write CP08 mask: {output}")
            outputs[name] = output
        canvas = frame.copy()
        for mask, color in contours:
            found, _ = cv2.findContours(
                np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, found, -1, color, 2)
        overlay = safe_child(
            self.evidence_root, f"cp_08/{group}/overlays/{artifact_id}.jpg"
        )
        overlay.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(overlay), canvas):
            raise OSError(f"could not write CP08 overlay: {overlay}")
        outputs["overlay"] = overlay
        return outputs

    def _write_metrics(
        self,
        group: str,
        *,
        selected_prompts: dict[str, str],
        rows: list[dict[str, Any]],
        failure_reasons: list[str],
    ) -> None:
        output = safe_child(self.evidence_root, f"cp_08/{group}/per_frame_metrics.json")
        atomic_write_json(
            output,
            {
                "selected_prompts": selected_prompts,
                "failure_reason": failure_reasons[0] if failure_reasons else None,
                "failure_reasons": failure_reasons,
                "frames": rows,
            },
        )
