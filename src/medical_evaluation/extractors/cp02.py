from __future__ import annotations

from collections.abc import Iterable
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
    locate_moving_multihole_disk,
    monotonic_arc_size_order,
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
    probe_contact: bool


class Cp02FeatureExtractor:
    locator_fps = 2.0
    visibility_fps = 5.0
    tail_search_chunk_sec = 5.0
    tail_search_overlap_sec = 1.0
    coarse_locator_width = 960
    coarse_chunk_frames = 8
    # Half-native sampling on the 25 FPS source covers alternating source
    # frames while avoiding repeated Hough work on every frame in each motion
    # episode. Reliability gates still require repeated clear observations.
    minimum_dense_fps = 12.5
    final_refine_fps = 50.0
    final_refine_lookback_sec = 1.0
    dense_radius_sec = 1.2
    maximum_stable_gap_sec = 0.8
    # Compatibility for the old, now-unused stable-pair helper.
    maximum_prepunch_event_gap_sec = 4.0
    minimum_contact_visibility_gap_sec = 0.5
    maximum_exit_gap_sec = 0.4
    residue_absent_ratio = 0.10
    residue_present_ratio = 0.35

    def __init__(self, *, annotations: VideoAnnotations, evidence_root: Path) -> None:
        self.annotations = annotations
        self.evidence_root = evidence_root

    @property
    def model_version(self) -> str:
        return "opencv-cp02-five-hole-v2"

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
        empty = self._empty_features(stage_scan_reliable=False)
        visibility_seed: _HoleFrame | None = None
        latest_frames: list[SampledFrame] = []
        for window in self._tail_search_windows(scan_range):
            locator_chunk = list(sample_frames(
                video_path,
                start_sec=window.start_sec,
                end_sec=window.end_sec,
                sample_fps=self.locator_fps,
            ))
            if not latest_frames:
                latest_frames = locator_chunk
            if len(locator_chunk) < 3:
                continue
            source_width = locator_chunk[0].image_bgr.shape[1]
            locator_scale = min(1.0, self.coarse_locator_width / source_width)
            locator_frames = [
                cv2.resize(
                    item.image_bgr, None, fx=locator_scale, fy=locator_scale,
                    interpolation=cv2.INTER_AREA,
                )
                if locator_scale < 1.0 else item.image_bgr
                for item in locator_chunk
            ]
            locator_anchor = locate_last_moving_multihole_disk(locator_frames)
            if locator_anchor is None:
                continue
            anchor = self._rescale_disk(locator_anchor, locator_scale)
            visibility_chunk = list(sample_frames(
                video_path,
                start_sec=window.start_sec,
                end_sec=window.end_sec,
                sample_fps=self.visibility_fps,
            ))
            visible = self._combine_observations(
                self._measure_dense(visibility_chunk, anchor),
                self._measure_dense(visibility_chunk, anchor, track=True),
            )
            if visible:
                visibility_seed = visible[-1]
                break

        if visibility_seed is None:
            self._write_table([], reason="automatic_disk_not_reliable")
            return ExtractedEvidence(
                features=empty,
                evidence=self._write_diagnostic_frames(
                    latest_frames, rule="automatic_disk_not_reliable",
                ),
            )

        selection_boundary_time_sec = min(
            scan_range.end_sec,
            visibility_seed.frame.time_sec + 1.0 / self.visibility_fps,
        )
        refine_start = max(
            scan_range.start_sec,
            selection_boundary_time_sec - self.final_refine_lookback_sec,
        )
        refine_frames = list(sample_frames(
                video_path,
                start_sec=refine_start,
                end_sec=selection_boundary_time_sec,
                sample_fps=self.final_refine_fps,
            ))
        observations = self._combine_observations(
            self._measure_dense(refine_frames, visibility_seed.disk),
            self._measure_dense(refine_frames, visibility_seed.disk, track=True),
        )
        selected = self._last_clear_cp02_frame(
            observations, boundary_time_sec=selection_boundary_time_sec,
        )
        if selected is None:
            features = dict(empty)
            features.update({
                "selection_boundary_time_sec": selection_boundary_time_sec,
                "last_visible_time_sec": visibility_seed.frame.time_sec,
                "hole_valid_frame_count": float(len(observations)),
            })
            self._write_table(
                observations, reason="final_selection_not_stable",
                selection_boundary_time_sec=selection_boundary_time_sec,
                last_visible_time_sec=visibility_seed.frame.time_sec,
            )
            evidence = self._write_unreliable_evidence(observations)
            if not evidence:
                evidence = self._write_diagnostic_frames(
                    latest_frames, rule="no_reliable_hole_observations",
                )
            return ExtractedEvidence(features=features, evidence=evidence)

        median_green = selected.green_ratio
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
            "selection_boundary_time_sec": selection_boundary_time_sec,
            "last_visible_time_sec": visibility_seed.frame.time_sec,
            "selected_second_largest": selected.size_rank == 2,
            "selected_hole_size_rank": float(selected.size_rank),
            "hole_ranking_confidence": float(selected.layout.ranking_confidence),
            "hole_valid_frame_count": float(len(observations)),
            "final_stable_start_sec": float(selected.frame.time_sec),
            "final_stable_end_sec": float(selected.frame.time_sec),
            "residue_green_ratio": median_green,
            "residue_before": residue,
            "cleanup_contact_observed": None,
            "residue_after": None,
        }
        evidence = self._write_evidence((selected,))
        self._write_table(
            observations, reason="criteria_satisfied",
            selection_boundary_time_sec=selection_boundary_time_sec,
            last_visible_time_sec=visibility_seed.frame.time_sec,
        )
        return ExtractedEvidence(features=features, evidence=evidence)

    def _scan_range(self, cp02_range: TimeRange) -> TimeRange:
        cp03 = next(
            (item.time_range for item in self.annotations.steps if item.checkpoint_id == "cp_03"),
            None,
        )
        return prepunch_scan_range(cp02_range, cp03) if cp03 is not None else cp02_range

    def _tail_search_windows(self, scan_range: TimeRange) -> Iterable[TimeRange]:
        end_sec = scan_range.end_sec
        while end_sec > scan_range.start_sec:
            start_sec = max(scan_range.start_sec, end_sec - self.tail_search_chunk_sec)
            yield TimeRange(start_sec=start_sec, end_sec=end_sec)
            if start_sec == scan_range.start_sec:
                break
            end_sec = start_sec + self.tail_search_overlap_sec

    def _measure_dense(
        self,
        frames: Iterable[SampledFrame],
        anchor: MovingDisk,
        *,
        track: bool = False,
    ) -> list[_HoleFrame]:
        observations: list[_HoleFrame] = []
        tracked_anchor = anchor
        for frame in frames:
            located = locate_disk_layout_near(frame.image_bgr, tracked_anchor)
            if located is None:
                continue
            disk, layout = located
            aligned = aligned_hole_opposite_handle(frame.image_bgr, disk, layout.holes)
            if aligned is None or layout.second_largest_index is None:
                continue
            order = monotonic_arc_size_order(layout.holes)
            if order is None:
                order = tuple(sorted(
                    range(len(layout.holes)),
                    key=lambda index: layout.holes[index].radius,
                    reverse=True,
                ))
            observations.append(_HoleFrame(
                frame=frame,
                disk=disk,
                layout=layout,
                aligned_index=aligned,
                size_rank=order.index(aligned) + 1,
                green_ratio=_hole_green_ratio(frame.image_bgr, layout.holes[aligned]),
                probe_contact=False,
            ))
            if track:
                tracked_anchor = disk
        return observations

    @staticmethod
    def _combine_observations(*groups: list[_HoleFrame]) -> list[_HoleFrame]:
        by_frame: dict[int, _HoleFrame] = {}
        for item in (observation for group in groups for observation in group):
            previous = by_frame.get(item.frame.frame_index)
            if (
                previous is None
                or item.layout.ranking_confidence > previous.layout.ranking_confidence
            ):
                by_frame[item.frame.frame_index] = item
        return sorted(by_frame.values(), key=lambda item: item.frame.time_sec)

    def _candidate_dense_windows(
        self,
        coarse: list[SampledFrame],
        fallback_anchor: MovingDisk,
        scan_range: TimeRange,
        *,
        locator_frames: list[np.ndarray] | None = None,
        locator_scale: float = 1.0,
    ) -> list[tuple[TimeRange, MovingDisk]]:
        """Find every moving-disk episode cheaply, then densify only nearby."""
        if locator_frames is None:
            locator_frames = [item.image_bgr for item in coarse]
        anchors: list[tuple[float, MovingDisk]] = []
        for start in range(0, len(coarse), self.coarse_chunk_frames):
            end = min(len(coarse), start + self.coarse_chunk_frames)
            if end - start < 3:
                continue
            located = locate_moving_multihole_disk(
                locator_frames[start:end],
            )
            if located is None:
                continue
            absolute_position = start + located.frame_position
            if absolute_position >= len(coarse):
                continue
            anchors.append((
                coarse[absolute_position].time_sec,
                self._rescale_disk(located, locator_scale),
            ))
        fallback_position = min(
            max(0, fallback_anchor.frame_position), len(coarse) - 1,
        )
        anchors.append((coarse[fallback_position].time_sec, fallback_anchor))
        anchors.sort(key=lambda item: item[0])

        windows: list[tuple[TimeRange, MovingDisk]] = []
        for time_sec, candidate in anchors:
            start_sec = max(scan_range.start_sec, time_sec - self.dense_radius_sec)
            end_sec = min(scan_range.end_sec, time_sec + self.dense_radius_sec)
            if end_sec <= start_sec:
                continue
            if windows and start_sec <= windows[-1][0].end_sec:
                prior, _prior_anchor = windows[-1]
                windows[-1] = (
                    TimeRange(
                        start_sec=prior.start_sec,
                        end_sec=max(prior.end_sec, end_sec),
                    ),
                    candidate,
                )
            else:
                windows.append((TimeRange(start_sec=start_sec, end_sec=end_sec), candidate))
        return windows

    @staticmethod
    def _rescale_disk(disk: MovingDisk, scale: float) -> MovingDisk:
        if not 0 < scale <= 1:
            raise ValueError("locator scale must be in (0, 1]")
        return MovingDisk(
            frame_position=disk.frame_position,
            x=disk.x / scale,
            y=disk.y / scale,
            # Downsampling makes the coarse Hough refinement lock onto the
            # inner plate edge. Inflate only the tracking search reference;
            # dense frames re-estimate the actual plate before measuring holes.
            radius=(disk.radius / scale) * (1.5 if scale < 1.0 else 1.0),
            hole_count=disk.hole_count,
            motion_ratio=disk.motion_ratio,
            surface_contrast=disk.surface_contrast,
        )

    def _final_prepunch_pair(
        self,
        observations: list[_HoleFrame],
        *,
        event_time_sec: float | None = None,
    ) -> tuple[_HoleFrame, _HoleFrame] | None:
        """Return the last repeated stable selection before a verified event."""
        if event_time_sec is None:
            return None
        before_contact = [
            item for item in observations if item.frame.time_sec < event_time_sec
        ]
        if len(before_contact) < 2:
            return None
        if event_time_sec - before_contact[-1].frame.time_sec > self.maximum_prepunch_event_gap_sec:
            return None
        for later_index in range(len(before_contact) - 1, 0, -1):
            later = before_contact[later_index]
            for earlier in reversed(before_contact[:later_index]):
                elapsed = later.frame.time_sec - earlier.frame.time_sec
                if elapsed > self.maximum_stable_gap_sec:
                    break
                if elapsed >= 0.04 and self._same_stable_selection(earlier, later):
                    return earlier, later
        return None

    def _last_clear_cp02_frame(
        self,
        observations: list[_HoleFrame],
        *,
        boundary_time_sec: float | None,
    ) -> _HoleFrame | None:
        """Return the closest clear hole view before CP03 starts."""
        if boundary_time_sec is None:
            return None
        for item in reversed(observations):
            if item.frame.time_sec >= boundary_time_sec:
                continue
            if item.green_ratio is not None:
                return item
        return None

    def _same_stable_selection(self, earlier: _HoleFrame, later: _HoleFrame) -> bool:
        if earlier.size_rank != later.size_rank:
            return False
        elapsed = later.frame.time_sec - earlier.frame.time_sec
        if not 0 < elapsed <= self.maximum_stable_gap_sec:
            return False
        scale = max(1.0, earlier.disk.radius, later.disk.radius)
        center_shift = float(np.hypot(
            later.disk.x - earlier.disk.x, later.disk.y - earlier.disk.y,
        ))
        if center_shift > 0.35 * scale:
            return False
        if abs(later.disk.radius - earlier.disk.radius) > 0.35 * scale:
            return False
        earlier_hole = earlier.layout.holes[earlier.aligned_index]
        later_hole = later.layout.holes[later.aligned_index]
        hole_shift = float(np.hypot(
            later_hole.x - earlier_hole.x, later_hole.y - earlier_hole.y,
        ))
        return hole_shift <= 0.65 * scale

    @staticmethod
    def _empty_features(*, stage_scan_reliable: bool) -> dict[str, float | bool | None]:
        return {
            "stage_scan_reliable": stage_scan_reliable,
            "selection_boundary_time_sec": None,
            "last_visible_time_sec": None,
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
        self,
        stable: tuple[_HoleFrame, ...],
    ) -> list[EvidenceItem]:
        evidence = []
        items = list(stable)
        for position, item in enumerate(items):
            overlay = item.frame.image_bgr.copy()
            cv2.circle(
                overlay,
                (round(item.disk.x), round(item.disk.y)),
                round(item.disk.radius),
                (0, 255, 255),
                3,
            )
            order = monotonic_arc_size_order(item.layout.holes)
            if order is None:
                order = tuple(sorted(
                    range(len(item.layout.holes)),
                    key=lambda index: item.layout.holes[index].radius,
                    reverse=True,
                ))
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
                rule=(
                    "earlier_cp02_disk_adjustment"
                    if position < len(stable) - 1
                    else "final_cp02_selected_hole"
                ),
            ))
        return evidence

    def _write_unreliable_evidence(
        self, observations: list[_HoleFrame],
    ) -> list[EvidenceItem]:
        if not observations:
            return []
        items = observations[-3:]
        evidence = self._write_evidence((items[0], items[-1]))
        return [
            item.model_copy(update={"rule": "unreliable_hole_selection"})
            for item in evidence
        ]

    def _write_diagnostic_frames(
        self, frames: list[SampledFrame], *, rule: str,
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        if not frames:
            return evidence
        indices = sorted({0, len(frames) // 2, len(frames) - 1})
        for index in indices:
            item = frames[index]
            overlay = item.image_bgr.copy()
            cv2.putText(
                overlay, rule, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 255), 2, cv2.LINE_AA,
            )
            relative = f"cp_02/overlays/{item.frame_index:08d}.jpg"
            output = safe_child(self.evidence_root, relative)
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), overlay):
                raise OSError(f"could not write CP02 diagnostic evidence: {output}")
            raw = safe_child(
                self.evidence_root, f"cp_02/raw/{item.frame_index:08d}.jpg",
            )
            raw.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(raw), item.image_bgr):
                raise OSError(f"could not write CP02 raw evidence: {raw}")
            evidence.append(EvidenceItem(
                time_sec=item.time_sec, overlay_path=relative, rule=rule,
            ))
        return evidence

    def _write_table(
        self,
        observations: list[_HoleFrame],
        *,
        reason: str,
        selection_boundary_time_sec: float | None = None,
        last_visible_time_sec: float | None = None,
    ) -> None:
        output = safe_child(self.evidence_root, "cp_02/hole_observations.json")
        atomic_write_json(output, {
            "reason": reason,
            "selection_boundary": {
                "time_sec": selection_boundary_time_sec,
                "last_visible_time_sec": last_visible_time_sec,
                "reason": "last_reliable_cp02_disk",
            },
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
                    "plane_axis_ratio": item.layout.plane_axis_ratio,
                    "aligned_index": item.aligned_index,
                    "selected_size_rank": item.size_rank,
                    "green_ratio": item.green_ratio,
                    "probe_contact": item.probe_contact,
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
    ratio = float(np.mean(green[interior]))
    if ratio > 0:
        return ratio
    # An empty punch aperture is visibly dark. A bright, non-green interior is
    # more likely glove/metal occlusion, so it cannot prove residue absence.
    value = hsv[..., 2]
    interior_value = float(np.median(value[interior]))
    if interior_value > 90.0:
        return None
    radius_sq = (xx - hole.x) ** 2 + (yy - hole.y) ** 2
    rim = (
        (radius_sq >= (1.05 * hole.radius) ** 2)
        & (radius_sq <= (1.35 * hole.radius) ** 2)
    )
    if not rim.any():
        return None
    rim_values = value[rim]
    if float(np.median(rim_values)) - interior_value < 35.0:
        return None
    if float(np.mean(rim_values <= interior_value + 20.0)) > 0.08:
        return None
    return 0.0
