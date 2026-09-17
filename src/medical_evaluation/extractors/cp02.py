from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.features.cp02_punch import (
    DiskHoleLayout,
    Hole,
    MovingDisk,
    aligned_hole_opposite_handle,
    locate_disk_layout_near,
    locate_last_moving_multihole_disk,
    prepunch_scan_range,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.storage import atomic_write_json, safe_child
from medical_evaluation.video import SampledFrame, sample_frames


@dataclass(frozen=True)
class _HoleFrame:
    frame: SampledFrame
    disk: MovingDisk
    layout: DiskHoleLayout
    aligned_index: int
    size_rank: int
    green_ratio: float | None


class Cp02FeatureExtractor:
    coarse_fps = 2.0
    minimum_dense_fps = 10.0
    dense_radius_sec = 1.2
    maximum_stable_gap_sec = 0.35
    maximum_exit_gap_sec = 0.4
    residue_absent_ratio = 0.10
    residue_present_ratio = 0.35

    def __init__(self, *, annotations: VideoAnnotations, evidence_root: Path) -> None:
        self.annotations = annotations
        self.evidence_root = evidence_root

    @property
    def model_version(self) -> str:
        return "opencv-cp02-five-hole-v1"

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id != "cp_02":
            raise ValueError("Cp02FeatureExtractor is CP02-only")
        if dense_fps <= 0 or analysis_width <= 0:
            raise ValueError("analysis settings must be positive")
        scan_range = self._scan_range(time_range)
        coarse = list(sample_frames(
            video_path,
            start_sec=scan_range.start_sec,
            end_sec=scan_range.end_sec,
            sample_fps=self.coarse_fps,
        ))
        empty = self._empty_features(stage_scan_reliable=bool(coarse))
        if len(coarse) < 3:
            return ExtractedEvidence(features=empty, evidence=[])
        anchor = locate_last_moving_multihole_disk(
            [item.image_bgr for item in coarse],
        )
        if anchor is None:
            empty["punch_action_observed"] = False
            return ExtractedEvidence(features=empty, evidence=[])

        anchor_time = coarse[anchor.frame_position].time_sec
        dense_start = max(scan_range.start_sec, anchor_time - self.dense_radius_sec)
        dense_end = min(scan_range.end_sec, anchor_time + self.dense_radius_sec)
        dense = list(sample_frames(
            video_path,
            start_sec=dense_start,
            end_sec=dense_end,
            sample_fps=max(dense_fps, self.minimum_dense_fps),
        ))
        observations = self._measure_dense(dense, anchor)
        stable = self._last_stable_pair(observations)
        if stable is None:
            features = dict(empty)
            features.update({
                "punch_action_observed": True,
                "hole_valid_frame_count": float(len(observations)),
            })
            self._write_table(observations, reason="final_selection_not_stable")
            return ExtractedEvidence(features=features, evidence=[])

        earlier, last = stable
        ratios = [item.green_ratio for item in stable if item.green_ratio is not None]
        median_green = float(np.median(ratios)) if ratios else None
        if median_green is None:
            residue: bool | None = None
        elif median_green <= self.residue_absent_ratio:
            residue = False
        elif median_green >= self.residue_present_ratio:
            residue = True
        else:
            residue = None
        features: dict[str, float | bool | None] = {
            "stage_scan_reliable": True,
            "punch_action_observed": True,
            "selected_second_largest": last.size_rank == 2,
            "selected_hole_size_rank": float(last.size_rank),
            "hole_ranking_confidence": float(min(
                earlier.layout.ranking_confidence,
                last.layout.ranking_confidence,
            )),
            "hole_valid_frame_count": float(len(observations)),
            "final_stable_start_sec": float(earlier.frame.time_sec),
            "final_stable_end_sec": float(last.frame.time_sec),
            "residue_green_ratio": median_green,
            "residue_before": residue,
            "cleanup_contact_observed": None,
            "residue_after": False if residue is False else None,
        }
        evidence = self._write_evidence(stable)
        self._write_table(observations, reason="criteria_satisfied")
        return ExtractedEvidence(features=features, evidence=evidence)

    def _scan_range(self, cp02_range: TimeRange) -> TimeRange:
        cp03 = next(
            (item.time_range for item in self.annotations.steps if item.checkpoint_id == "cp_03"),
            None,
        )
        return prepunch_scan_range(cp02_range, cp03) if cp03 is not None else cp02_range

    def _measure_dense(
        self, frames: list[SampledFrame], anchor: MovingDisk,
    ) -> list[_HoleFrame]:
        observations: list[_HoleFrame] = []
        for frame in frames:
            located = locate_disk_layout_near(frame.image_bgr, anchor)
            if located is None:
                continue
            disk, layout = located
            aligned = aligned_hole_opposite_handle(frame.image_bgr, disk, layout.holes)
            if aligned is None or layout.second_largest_index is None:
                continue
            order = sorted(
                range(len(layout.holes)),
                key=lambda index: layout.holes[index].radius,
                reverse=True,
            )
            observations.append(_HoleFrame(
                frame=frame,
                disk=disk,
                layout=layout,
                aligned_index=aligned,
                size_rank=order.index(aligned) + 1,
                green_ratio=_hole_green_ratio(frame.image_bgr, layout.holes[aligned]),
            ))
        return observations

    def _last_stable_pair(
        self, observations: list[_HoleFrame],
    ) -> tuple[_HoleFrame, _HoleFrame] | None:
        if len(observations) < 2:
            return None
        earlier, last = observations[-2:]
        if (
            earlier.size_rank != last.size_rank
            or last.frame.time_sec - earlier.frame.time_sec > self.maximum_stable_gap_sec
        ):
            return None
        return earlier, last

    @staticmethod
    def _empty_features(*, stage_scan_reliable: bool) -> dict[str, float | bool | None]:
        return {
            "stage_scan_reliable": stage_scan_reliable,
            "punch_action_observed": None,
            "selected_second_largest": None,
            "selected_hole_size_rank": None,
            "hole_ranking_confidence": None,
            "hole_valid_frame_count": 0.0,
            "final_stable_start_sec": None,
            "final_stable_end_sec": None,
            "residue_green_ratio": None,
            "residue_before": None,
            "cleanup_contact_observed": None,
            "residue_after": None,
        }

    def _write_evidence(
        self, stable: tuple[_HoleFrame, _HoleFrame],
    ) -> list[EvidenceItem]:
        evidence = []
        for item in stable:
            overlay = item.frame.image_bgr.copy()
            cv2.circle(
                overlay,
                (round(item.disk.x), round(item.disk.y)),
                round(item.disk.radius),
                (0, 255, 255),
                3,
            )
            order = sorted(
                range(len(item.layout.holes)),
                key=lambda index: item.layout.holes[index].radius,
                reverse=True,
            )
            hole_mask = np.zeros(overlay.shape[:2], np.uint8)
            for index, hole in enumerate(item.layout.holes):
                rank = order.index(index) + 1
                color = (0, 0, 255) if index == item.aligned_index else (255, 0, 255)
                center = (round(hole.x), round(hole.y))
                cv2.circle(overlay, center, max(2, round(hole.radius)), color, 2)
                cv2.putText(
                    overlay,
                    str(rank),
                    (center[0] + 4, center[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.circle(hole_mask, center, max(1, round(0.65 * hole.radius)), 255, -1)
            relative = f"cp_02/overlays/{item.frame.frame_index:08d}.jpg"
            output = safe_child(self.evidence_root, relative)
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), overlay):
                raise OSError(f"could not write CP02 evidence: {output}")
            raw = safe_child(
                self.evidence_root, f"cp_02/raw/{item.frame.frame_index:08d}.jpg",
            )
            raw.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(raw), item.frame.image_bgr):
                raise OSError(f"could not write CP02 raw evidence: {raw}")
            mask = safe_child(
                self.evidence_root, f"cp_02/masks/holes/{item.frame.frame_index:08d}.png",
            )
            mask.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask), hole_mask):
                raise OSError(f"could not write CP02 hole mask: {mask}")
            evidence.append(EvidenceItem(
                time_sec=item.frame.time_sec,
                overlay_path=relative,
                rule="final_prepunch_second_largest_hole_alignment",
            ))
        return evidence

    def _write_table(self, observations: list[_HoleFrame], *, reason: str) -> None:
        output = safe_child(self.evidence_root, "cp_02/hole_observations.json")
        atomic_write_json(output, {
            "reason": reason,
            "frames": [
                {
                    "frame_index": item.frame.frame_index,
                    "time_sec": item.frame.time_sec,
                    "disk": {
                        "x": item.disk.x,
                        "y": item.disk.y,
                        "radius": item.disk.radius,
                    },
                    "holes": [
                        {"x": hole.x, "y": hole.y, "radius": hole.radius}
                        for hole in item.layout.holes
                    ],
                    "ranking_confidence": item.layout.ranking_confidence,
                    "aligned_index": item.aligned_index,
                    "selected_size_rank": item.size_rank,
                    "green_ratio": item.green_ratio,
                }
                for item in observations
            ],
        })


def _hole_green_ratio(frame_bgr: np.ndarray, hole: Hole) -> float | None:
    hsv = cv2.cvtColor(np.asarray(frame_bgr, dtype=np.uint8), cv2.COLOR_BGR2HSV)
    green = (
        (hsv[..., 0] >= 30)
        & (hsv[..., 0] <= 100)
        & (hsv[..., 1] >= 55)
        & (hsv[..., 2] >= 45)
    )
    if int(green.sum()) < max(50, round(green.size * 0.0001)):
        return None
    yy, xx = np.ogrid[:green.shape[0], :green.shape[1]]
    interior = (
        (xx - hole.x) ** 2 + (yy - hole.y) ** 2 <= (0.65 * hole.radius) ** 2
    )
    if not interior.any():
        return None
    return float(np.mean(green[interior]))
