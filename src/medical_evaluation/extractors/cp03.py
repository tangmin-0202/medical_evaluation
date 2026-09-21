"""CP03 final-hole adhesion extraction using SAM dam masks and source pixels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.domain import TimeRange
from medical_evaluation.features.cp03_hole import (
    HoleAdhesionAnalysis,
    analyze_hole_adhesion,
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
from medical_evaluation.video import read_frame

DAM_PROMPT = "green dental rubber dam sheet"


@dataclass(frozen=True)
class _FrameObservation:
    item: FrameMasks
    frame: np.ndarray
    dam_mask: np.ndarray | None
    analysis: HoleAdhesionAnalysis | None


class Cp03FeatureExtractor:
    sample_fps = 2.0
    initial_window_sec = 4.0
    fallback_window_sec = 8.0
    minimum_clear_frames = 1

    def __init__(self, *, segmenter: VideoSegmenter, evidence_root: Path) -> None:
        self.segmenter = segmenter
        self.evidence_root = evidence_root

    @property
    def model_version(self) -> str:
        return f"{self.segmenter.model_version}+cp03-adhesion-v1"

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id != "cp_03":
            raise ValueError("Cp03FeatureExtractor is CP03-only")
        if dense_fps <= 0 or analysis_width <= 0:
            raise ValueError("sampling and analysis dimensions must be positive")

        observations = self._scan_window(
            video_path,
            self._tail_range(time_range, self.initial_window_sec),
        )
        selected = self._latest_clear_hole(observations)
        if selected is None and (
            time_range.end_sec - time_range.start_sec > self.initial_window_sec
        ):
            observations = self._scan_window(
                video_path,
                self._tail_range(time_range, self.fallback_window_sec),
            )
            selected = self._latest_clear_hole(observations)

        reliable_count = sum(
            item.analysis is not None and item.analysis.reliable
            for item in observations
        )
        final_scan_reliable = reliable_count >= self.minimum_clear_frames
        any_hole = selected is not None
        hole_observed: bool | None
        if not final_scan_reliable:
            hole_observed = None
        else:
            hole_observed = any_hole

        adhesion_count = int(
            selected is not None
            and selected.analysis is not None
            and selected.analysis.adhesion_detected is True
        )
        adhesion_free: bool | None = None
        if selected is not None and selected.analysis is not None:
            adhesion_free = not bool(selected.analysis.adhesion_detected)

        features: dict[str, float | bool | None] = {
            "final_scan_reliable": final_scan_reliable,
            "hole_observed": hole_observed,
            "hole_clear_consecutive_frames": float(selected is not None),
            "hole_adhesion_free": adhesion_free,
            "adhesion_observed_frame_count": float(adhesion_count),
        }
        evidence = self._write_evidence(
            observations, [] if selected is None else [selected]
        )
        return ExtractedEvidence(features=features, evidence=evidence)

    def _scan_window(
        self, video_path: Path, time_range: TimeRange
    ) -> list[_FrameObservation]:
        prompt = SegmentationPrompt(
            object_id="rubber_dam",
            kind="text",
            frame_time_sec=time_range.start_sec,
            text=DAM_PROMPT,
        )
        observations: list[_FrameObservation] = []
        try:
            for item in self.segmenter.track(
                video_path,
                time_range,
                [prompt],
                sample_fps=self.sample_fps,
            ):
                frame = read_frame(video_path, item.frame_index)
                mask = self._dam_mask(item, frame.shape[:2])
                analysis = None if mask is None else analyze_hole_adhesion(frame, mask)
                observations.append(
                    _FrameObservation(
                        item=item,
                        frame=frame,
                        dam_mask=mask,
                        analysis=analysis,
                    )
                )
        except Sam3AmbiguousTextResult:
            # Multiple unranked SAM candidates are not safe to guess between.
            return []
        return observations

    @staticmethod
    def _tail_range(stage: TimeRange, seconds: float) -> TimeRange:
        return TimeRange(
            start_sec=max(stage.start_sec, stage.end_sec - seconds),
            end_sec=stage.end_sec,
        )

    @staticmethod
    def _dam_mask(item: FrameMasks, shape: tuple[int, int]) -> np.ndarray | None:
        raw = item.masks.get("rubber_dam")
        if raw is None or not np.asarray(raw, dtype=bool).any():
            return None
        mask = np.asarray(raw, dtype=bool)
        if mask.shape != shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return mask

    @classmethod
    def _latest_clear_hole(
        cls, observations: list[_FrameObservation]
    ) -> _FrameObservation | None:
        return next(
            (
                observation
                for observation in reversed(observations)
                if observation.analysis is not None
                and observation.analysis.reliable
                and observation.analysis.hole_observed
            ),
            None,
        )

    def _write_evidence(
        self,
        observations: list[_FrameObservation],
        run: list[_FrameObservation],
    ) -> list[EvidenceItem]:
        selected_ids = {id(item) for item in run[-3:]}
        evidence: list[EvidenceItem] = []
        frame_rows: list[dict[str, float | bool | None]] = []
        for observation in observations:
            item = observation.item
            stem = f"{item.frame_index:08d}"
            base = "cp_03"
            paths = {
                "original": safe_child(self.evidence_root, f"{base}/originals/{stem}.jpg"),
                "mask": safe_child(self.evidence_root, f"{base}/masks/{stem}.png"),
                "roi": safe_child(self.evidence_root, f"{base}/rois/{stem}.jpg"),
                "overlay": safe_child(self.evidence_root, f"{base}/overlays/{stem}.jpg"),
            }
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            mask_image = (
                np.zeros(observation.frame.shape[:2], np.uint8)
                if observation.dam_mask is None
                else observation.dam_mask.astype(np.uint8) * 255
            )
            overlay = self._overlay(observation)
            roi = self._roi(observation)
            for path, image in (
                (paths["original"], observation.frame),
                (paths["mask"], mask_image),
                (paths["roi"], roi),
                (paths["overlay"], overlay),
            ):
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"could not write CP03 evidence: {path}")
            analysis = observation.analysis
            frame_rows.append(
                {
                    "time_sec": item.frame_time_sec,
                    "frame_index": float(item.frame_index),
                    "dam_reliable": bool(analysis and analysis.reliable),
                    "hole_observed": bool(analysis and analysis.hole_observed),
                    "adhesion_detected": (
                        None if analysis is None else analysis.adhesion_detected
                    ),
                    "adhesion_area_ratio": (
                        None if analysis is None else analysis.adhesion_area_ratio
                    ),
                }
            )
            if id(observation) in selected_ids:
                evidence.append(
                    EvidenceItem(
                        time_sec=item.frame_time_sec,
                        overlay_path=paths["overlay"].relative_to(
                            self.evidence_root
                        ).as_posix(),
                        rule="cp03_hole_adhesion",
                    )
                )
        atomic_write_json(
            safe_child(self.evidence_root, "cp_03/frame_analysis.json"),
            {"frames": frame_rows},
        )
        return evidence

    @staticmethod
    def _overlay(observation: _FrameObservation) -> np.ndarray:
        canvas = observation.frame.copy()
        if observation.dam_mask is not None:
            contours, _ = cv2.findContours(
                observation.dam_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
        analysis = observation.analysis
        if analysis is not None and analysis.hole_observed:
            contours, _ = cv2.findContours(
                analysis.hole_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(canvas, contours, -1, (255, 255, 0), 2)
            canvas[analysis.adhesion_mask] = (0, 0, 255)
        return canvas

    @staticmethod
    def _roi(observation: _FrameObservation) -> np.ndarray:
        analysis = observation.analysis
        if (
            analysis is None
            or analysis.center_xy is None
            or analysis.radius_px is None
        ):
            return observation.frame.copy()
        x, y = (round(value) for value in analysis.center_xy)
        margin = max(16, round(analysis.radius_px * 1.8))
        y1, y2 = max(0, y - margin), min(observation.frame.shape[0], y + margin + 1)
        x1, x2 = max(0, x - margin), min(observation.frame.shape[1], x + margin + 1)
        return observation.frame[y1:y2, x1:x2].copy()
