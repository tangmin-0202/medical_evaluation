from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.domain import TimeRange
from medical_evaluation.features.tooth_floss import (
    FlossLine,
    ToothAnchor,
    detect_floss_line,
    locate_tooth_anchor,
    segment_tooth,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.storage import safe_child
from medical_evaluation.video import SampledFrame, sample_frames


class Cp10FeatureExtractor:
    minimum_analysis_fps = 5.0
    contact_side_margin_ratio = 0.15

    def __init__(self, *, evidence_root: Path) -> None:
        self.evidence_root = evidence_root

    @property
    def model_version(self) -> str:
        return "opencv-tooth-floss-v1"

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id != "cp_10":
            raise ValueError("Cp10FeatureExtractor is CP10-only")
        if dense_fps <= 0 or analysis_width <= 0:
            raise ValueError("analysis settings must be positive")
        frames = list(
            sample_frames(
                video_path,
                start_sec=time_range.start_sec,
                end_sec=time_range.end_sec,
                sample_fps=max(dense_fps, self.minimum_analysis_fps),
            )
        )
        anchor = locate_tooth_anchor(item.image_bgr for item in frames)
        empty: dict[str, float | bool | None] = {
            "tooth_anchor_reliable": False,
            "tooth_anchor_score": None,
            "tooth_valid_frame_count": 0.0,
            "floss_observed_frame_count": 0.0,
            "upper_contact_frame_count": 0.0,
            "lower_contact_frame_count": 0.0,
        }
        if anchor is None:
            return ExtractedEvidence(features=empty, evidence=[])

        measurements: list[tuple[SampledFrame, np.ndarray, FlossLine | None]] = []
        tooth_valid_count = 0
        upper_count = 0
        lower_count = 0
        margin = anchor.height * self.contact_side_margin_ratio
        for frame in frames:
            tooth = segment_tooth(frame.image_bgr, anchor)
            tooth_valid_count += int(
                tooth.sum() >= max(20, anchor.width * anchor.height * 0.08)
            )
            floss = detect_floss_line(frame.image_bgr, anchor)
            if floss is not None:
                upper_count += int(floss.contact_y_offset <= -margin)
                lower_count += int(floss.contact_y_offset >= margin)
            measurements.append((frame, tooth, floss))

        observed = [item for item in measurements if item[2] is not None]
        evidence = self._write_evidence(measurements, anchor, margin)
        return ExtractedEvidence(
            features={
                "tooth_anchor_reliable": True,
                "tooth_anchor_score": float(anchor.score),
                "tooth_valid_frame_count": float(tooth_valid_count),
                "floss_observed_frame_count": float(len(observed)),
                "upper_contact_frame_count": float(upper_count),
                "lower_contact_frame_count": float(lower_count),
            },
            evidence=evidence,
        )

    def _write_evidence(
        self,
        measurements: list[tuple[SampledFrame, np.ndarray, FlossLine | None]],
        anchor: ToothAnchor,
        margin: float,
    ) -> list[EvidenceItem]:
        observed = [item for item in measurements if item[2] is not None]
        if not observed:
            return []
        upper = [item for item in observed if item[2].contact_y_offset <= -margin]  # type: ignore[union-attr]
        lower = [item for item in observed if item[2].contact_y_offset >= margin]  # type: ignore[union-attr]
        selected: list[tuple[SampledFrame, np.ndarray, FlossLine | None]] = []
        for group in (upper, lower, observed):
            if group:
                item = max(group, key=lambda value: value[2].score)  # type: ignore[union-attr]
                if item not in selected:
                    selected.append(item)
            if len(selected) == 3:
                break
        evidence: list[EvidenceItem] = []
        for frame, tooth, candidate in selected:
            assert candidate is not None
            overlay = frame.image_bgr.copy()
            contours, _ = cv2.findContours(
                tooth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3)
            cv2.line(
                overlay,
                (candidate.x1, candidate.y1),
                (candidate.x2, candidate.y2),
                (255, 0, 255),
                4,
            )
            cv2.circle(
                overlay,
                (round(anchor.center_x), round(anchor.center_y)),
                5,
                (0, 255, 255),
                -1,
            )
            output = safe_child(
                self.evidence_root, f"cp_10/overlays/{frame.frame_index:08d}.jpg"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), overlay):
                raise OSError(f"could not write CP10 evidence: {output}")
            mask = np.zeros(tooth.shape, dtype=np.uint8)
            cv2.line(
                mask,
                (candidate.x1, candidate.y1),
                (candidate.x2, candidate.y2),
                255,
                3,
            )
            mask_path = safe_child(
                self.evidence_root, f"cp_10/masks/dental_floss/{frame.frame_index:08d}.png"
            )
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"could not write CP10 floss mask: {mask_path}")
            evidence.append(
                EvidenceItem(
                    time_sec=frame.time_sec,
                    overlay_path=output.relative_to(self.evidence_root).as_posix(),
                    rule="floss_contacts_both_sides_of_target_tooth",
                )
            )
        return evidence
