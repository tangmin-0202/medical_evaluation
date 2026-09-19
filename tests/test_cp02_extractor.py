import json
from pathlib import Path

import numpy as np

from medical_evaluation.annotations import SegmentAnnotation, VideoAnnotations
from medical_evaluation.domain import CheckpointStatus, TimeRange
from medical_evaluation.features.cp02_punch import (
    DiskHoleLayout,
    Hole,
    MovingDisk,
)
from medical_evaluation.video import SampledFrame


def _annotations() -> VideoAnnotations:
    return VideoAnnotations(
        video_id="sample",
        steps=[
            SegmentAnnotation(
                checkpoint_id="cp_02",
                time_range=TimeRange(start_sec=10, end_sec=20),
                label=CheckpointStatus.NEEDS_REVIEW,
                reason="test",
            ),
            SegmentAnnotation(
                checkpoint_id="cp_03",
                time_range=TimeRange(start_sec=22, end_sec=30),
                label=CheckpointStatus.NEEDS_REVIEW,
                reason="test",
            ),
        ],
    )


def test_cp02_extractor_uses_last_stable_automatic_hole_rank(monkeypatch, tmp_path: Path):
    from medical_evaluation.extractors import cp02

    blank = np.full((240, 320, 3), (170, 120, 70), np.uint8)
    blank[:40] = (20, 170, 30)
    calls: list[tuple[float, float, float]] = []

    def frames(_path, *, start_sec, end_sec, sample_fps):
        calls.append((start_sec, end_sec, sample_fps))
        if sample_fps <= 2.1:
            times = (18.0, 18.5, 19.0)
        elif sample_fps >= 49.0:
            times = (19.15, 19.2)
        else:
            times = (18.8, 18.9, 19.0, 19.1)
        return iter(
            SampledFrame(time_sec=t, frame_index=round(t * 10), image_bgr=blank.copy())
            for t in times
            if start_sec <= t < end_sec
        )

    disk = MovingDisk(1, 150, 130, 45, 5, 0.5, 60)
    holes = (
        Hole(130, 115, 8),
        Hole(145, 105, 7),
        Hole(165, 108, 5),
        Hole(175, 125, 4),
        Hole(165, 145, 3),
    )
    for hole in holes:
        yy, xx = np.ogrid[:blank.shape[0], :blank.shape[1]]
        blank[(xx - hole.x) ** 2 + (yy - hole.y) ** 2 <= hole.radius**2] = 15
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.12)
    monkeypatch.setattr(cp02, "sample_frames", frames)
    monkeypatch.setattr(cp02, "locate_last_moving_multihole_disk", lambda _frames: disk)
    monkeypatch.setattr(cp02, "locate_disk_layout_near", lambda _frame, _disk: (disk, layout))
    monkeypatch.setattr(cp02, "aligned_hole_opposite_handle", lambda *_args: 0)
    result = cp02.Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=tmp_path / "evidence"
    ).extract(
        tmp_path / "unused.mp4",
        "cp_02",
        TimeRange(start_sec=10, end_sec=20),
        dense_fps=5,
        analysis_width=1280,
    )

    assert result.features["stage_scan_reliable"] is True
    assert "punch_action_observed" not in result.features
    assert result.features["selected_hole_size_rank"] == 1
    assert result.features["selected_second_largest"] is False
    assert result.features["residue_before"] is False
    assert len(result.evidence) == 1
    assert result.evidence[-1].rule == "final_cp02_selected_hole"
    assert all((tmp_path / "evidence" / item.overlay_path).is_file() for item in result.evidence)
    table = json.loads(
        (tmp_path / "evidence" / "cp_02" / "hole_observations.json").read_text()
    )
    assert table["selection_boundary"]["last_visible_time_sec"] == 19.1
    assert calls[:2] == [(17.0, 22.0, 2.0), (17.0, 22.0, 5.0)]
    refinement = next(call for call in calls if call[2] >= 50.0)
    assert refinement[:2] == (18.3, 19.3)


def test_cp02_extractor_uses_closest_clear_frame_without_requiring_repetition(
    monkeypatch, tmp_path: Path,
):
    from medical_evaluation.extractors import cp02

    blank = np.full((240, 320, 3), (170, 120, 70), np.uint8)

    def frames(_path, *, start_sec, end_sec, sample_fps):
        times = (18.8, 19.0, 19.2)
        return iter(
            SampledFrame(time_sec=t, frame_index=round(t * 10), image_bgr=blank.copy())
            for t in times
            if start_sec <= t < end_sec
        )

    disk = MovingDisk(1, 150, 130, 45, 5, 0.5, 60)
    holes = tuple(Hole(120 + i * 15, 120, 8 - i) for i in range(5))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.12)
    monkeypatch.setattr(cp02, "sample_frames", frames)
    monkeypatch.setattr(cp02, "locate_last_moving_multihole_disk", lambda _frames: disk)
    monkeypatch.setattr(cp02, "locate_disk_layout_near", lambda _frame, _disk: (disk, layout))
    monkeypatch.setattr(cp02, "aligned_hole_opposite_handle", lambda *_args: 1)
    monkeypatch.setattr(cp02, "_hole_green_ratio", lambda *_args: 0.0)

    result = cp02.Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=tmp_path / "evidence"
    ).extract(
        tmp_path / "unused.mp4",
        "cp_02",
        TimeRange(start_sec=10, end_sec=20),
        dense_fps=5,
        analysis_width=1280,
    )

    assert result.features["selected_second_largest"] is True
    assert result.features["selected_hole_size_rank"] == 2
    table = json.loads(
        (tmp_path / "evidence" / "cp_02" / "hole_observations.json").read_text()
    )
    assert table["reason"] == "criteria_satisfied"
    assert result.evidence


def test_cp02_contact_gap_rejects_different_post_gap_disk_rank() -> None:
    from medical_evaluation.extractors.cp02 import Cp02FeatureExtractor, _HoleFrame

    image = np.zeros((40, 40, 3), np.uint8)
    disk = MovingDisk(0, 20, 20, 10, 5, 0.5, 60)
    holes = tuple(Hole(10 + i * 4, 20, 6 - i * 0.5) for i in range(5))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.1)
    observations = [
        _HoleFrame(
            frame=SampledFrame(time_sec=time, frame_index=index, image_bgr=image),
            disk=disk,
            layout=layout,
            aligned_index=rank - 1,
            size_rank=rank,
            green_ratio=0.0,
            probe_contact=False,
        )
        for index, (time, rank) in enumerate(((1.0, 1), (1.1, 2), (2.0, 3)))
    ]

    pair = Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=Path("unused")
    )._final_prepunch_pair(observations, event_time_sec=1.5)

    assert pair is None


def test_cp02_final_pair_requires_stable_same_rank_tail() -> None:
    from medical_evaluation.extractors.cp02 import Cp02FeatureExtractor, _HoleFrame

    image = np.zeros((40, 40, 3), np.uint8)
    disk = MovingDisk(0, 20, 20, 10, 5, 0.5, 60)
    holes = tuple(Hole(10 + i * 4, 20, 6 - i * 0.5) for i in range(5))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.1)
    observations = [
        _HoleFrame(
            SampledFrame(time_sec=time, frame_index=index, image_bgr=image),
            disk, layout, rank - 1, rank, 0.0, False,
        )
        for index, (time, rank) in enumerate(
            ((1.0, 1), (1.1, 2), (1.2, 2), (1.3, 2))
        )
    ]

    pair = Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=Path("unused")
    )._final_prepunch_pair(observations, event_time_sec=2.2)

    assert pair is not None
    assert [item.size_rank for item in pair] == [2, 2]
    assert pair[1].frame.time_sec > pair[0].frame.time_sec


def test_cp02_final_pair_rejects_multiple_visibility_gaps() -> None:
    from medical_evaluation.extractors.cp02 import Cp02FeatureExtractor, _HoleFrame

    image = np.zeros((40, 40, 3), np.uint8)
    disk = MovingDisk(0, 20, 20, 10, 5, 0.5, 60)
    holes = tuple(Hole(10 + i * 4, 20, 6 - i * 0.5) for i in range(5))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.1)
    observations = [
        _HoleFrame(
            SampledFrame(time_sec=time, frame_index=index, image_bgr=image),
            disk, layout, 1, 2, 0.0, False,
        )
        for index, time in enumerate((1.0, 1.1, 2.0, 2.1, 3.0, 3.1))
    ]

    assert Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=Path("unused")
    )._final_prepunch_pair(observations, event_time_sec=None) is None


def test_cp02_final_pair_uses_last_stable_rank_before_explicit_punch_event() -> None:
    from medical_evaluation.extractors.cp02 import Cp02FeatureExtractor, _HoleFrame

    image = np.zeros((40, 40, 3), np.uint8)
    disk = MovingDisk(0, 20, 20, 10, 5, 0.5, 60)
    holes = tuple(Hole(10 + i * 4, 20, 6 - i * 0.5) for i in range(5))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.1)
    observations = [
        _HoleFrame(
            SampledFrame(time_sec=time, frame_index=index, image_bgr=image),
            disk, layout, rank - 1, rank, 0.0, False,
        )
        for index, (time, rank) in enumerate(
            ((1.0, 2), (1.1, 2), (1.5, 3), (1.6, 3), (2.4, 1), (2.5, 1))
        )
    ]

    pair = Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=Path("unused")
    )._final_prepunch_pair(observations, event_time_sec=2.0)

    assert pair is not None
    assert [item.size_rank for item in pair] == [3, 3]


def test_cp02_final_pair_skips_isolated_exit_candidate_after_last_clear_pair() -> None:
    from medical_evaluation.extractors.cp02 import Cp02FeatureExtractor, _HoleFrame

    image = np.zeros((40, 40, 3), np.uint8)
    disk = MovingDisk(0, 20, 20, 10, 5, 0.5, 60)
    holes = tuple(Hole(10 + i * 4, 20, 6 - i * 0.5) for i in range(5))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.30)
    exit_disk = MovingDisk(0, 34, 32, 15, 5, 0.5, 60)
    observations = [
        _HoleFrame(
            SampledFrame(time_sec=1.0, frame_index=1, image_bgr=image),
            disk, layout, 1, 2, 0.0, False,
        ),
        _HoleFrame(
            SampledFrame(time_sec=1.6, frame_index=2, image_bgr=image),
            disk, layout, 1, 2, 0.0, False,
        ),
        _HoleFrame(
            SampledFrame(time_sec=1.9, frame_index=3, image_bgr=image),
            exit_disk, layout, 4, 5, None, False,
        ),
    ]

    pair = Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=Path("unused")
    )._final_prepunch_pair(observations, event_time_sec=2.0)

    assert pair is not None
    assert [item.size_rank for item in pair] == [2, 2]
    assert pair[-1].frame.time_sec == 1.6


def test_cp02_final_pair_does_not_treat_a_visibility_gap_as_contact() -> None:
    from medical_evaluation.extractors.cp02 import Cp02FeatureExtractor, _HoleFrame

    image = np.zeros((40, 40, 3), np.uint8)
    disk = MovingDisk(0, 20, 20, 10, 5, 0.5, 60)
    holes = tuple(Hole(10 + i * 4, 20, 6 - i * 0.5) for i in range(5))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.1)
    observations = [
        _HoleFrame(
            SampledFrame(time_sec=time, frame_index=index, image_bgr=image),
            disk, layout, 1, 2, 0.0, False,
        )
        for index, time in enumerate((1.0, 1.1, 2.0, 2.1))
    ]

    assert Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=Path("unused")
    )._final_prepunch_pair(observations, event_time_sec=None) is None


def test_cp02_uses_nearest_clear_uncovered_frame_before_boundary() -> None:
    from medical_evaluation.extractors.cp02 import Cp02FeatureExtractor, _HoleFrame

    image = np.zeros((40, 40, 3), np.uint8)
    disk = MovingDisk(0, 20, 20, 10, 5, 0.5, 60)
    holes = tuple(Hole(10 + i * 4, 20, 6 - i * 0.5) for i in range(5))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.20)
    observations = [
        _HoleFrame(
            SampledFrame(time_sec=1.8, frame_index=1, image_bgr=image),
            disk, layout, 1, 2, 0.0, False,
        ),
        _HoleFrame(
            SampledFrame(time_sec=1.9, frame_index=2, image_bgr=image),
            disk, layout, 4, 5, None, False,
        ),
    ]

    selected = Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=Path("unused")
    )._last_clear_cp02_frame(observations, boundary_time_sec=2.0)

    assert selected is not None
    assert selected.frame.time_sec == 1.8
    assert selected.size_rank == 2


def test_cp02_no_automatic_anchor_is_review_with_diagnostic_evidence(
    monkeypatch, tmp_path: Path,
):
    from medical_evaluation.extractors import cp02

    blank = np.zeros((40, 60, 3), np.uint8)
    monkeypatch.setattr(
        cp02,
        "sample_frames",
        lambda *_args, **_kwargs: iter(
            SampledFrame(time_sec=float(i), frame_index=i, image_bgr=blank.copy())
            for i in range(3)
        ),
    )
    monkeypatch.setattr(cp02, "locate_last_moving_multihole_disk", lambda _frames: None)

    result = cp02.Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=tmp_path / "evidence"
    ).extract(
        tmp_path / "unused.mp4", "cp_02", TimeRange(start_sec=10, end_sec=20),
        dense_fps=5, analysis_width=1280,
    )

    assert result.features["stage_scan_reliable"] is False
    assert "punch_action_observed" not in result.features
    assert result.evidence
    table = json.loads(
        (tmp_path / "evidence" / "cp_02" / "hole_observations.json").read_text()
    )
    assert table["reason"] == "automatic_disk_not_reliable"


def test_cp02_search_starts_from_final_five_seconds(
    monkeypatch, tmp_path: Path,
):
    from medical_evaluation.extractors import cp02

    blank = np.zeros((80, 120, 3), np.uint8)
    calls: list[tuple[float, float, float]] = []

    def frames(_path, *, start_sec, end_sec, sample_fps):
        calls.append((start_sec, end_sec, sample_fps))
        times = (18.0, 18.5, 19.0) if sample_fps <= 2.1 else (20.0, 20.1, 20.2)
        return iter(
            SampledFrame(time_sec=t, frame_index=round(t * 10), image_bgr=blank.copy())
            for t in times
        )

    disk = MovingDisk(2, 60, 45, 25, 5, 0.5, 60)
    holes = tuple(Hole(40 + i * 8, 45, 7 - i * 0.6) for i in range(5))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.12)
    monkeypatch.setattr(cp02, "sample_frames", frames)
    monkeypatch.setattr(cp02, "locate_last_moving_multihole_disk", lambda _frames: disk)
    monkeypatch.setattr(cp02, "locate_disk_layout_near", lambda *_args: (disk, layout))
    monkeypatch.setattr(cp02, "aligned_hole_opposite_handle", lambda *_args: 1)

    result = cp02.Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=tmp_path / "evidence"
    ).extract(
        tmp_path / "unused.mp4", "cp_02", TimeRange(start_sec=10, end_sec=20),
        dense_fps=5, analysis_width=1280,
    )

    assert calls[:2] == [(17.0, 22.0, 2.0), (17.0, 22.0, 5.0)]
    assert len(calls) == 3
    assert calls[2] == (19.4, 20.4, 50.0)
    assert result.features["selected_second_largest"] is None


def test_cp02_final_refinement_uses_latest_visible_disk_as_anchor(
    monkeypatch, tmp_path: Path,
):
    from medical_evaluation.extractors import cp02

    blank = np.zeros((80, 120, 3), np.uint8)

    def frames(_path, *, start_sec, end_sec, sample_fps):
        if sample_fps <= 2.1:
            times = (18.0, 18.5, 19.0)
        elif sample_fps < 10.0:
            times = (19.0, 19.2, 19.4, 19.6, 19.8)
        else:
            times = (19.82, 19.84)
        return iter(
            SampledFrame(
                time_sec=t, frame_index=round(t * 100), image_bgr=blank.copy(),
            )
            for t in times if start_sec <= t < end_sec
        )

    coarse_disk = MovingDisk(2, 50, 40, 20, 4, 0.5, 60)
    latest_disk = MovingDisk(0, 58, 42, 20, 4, 0.5, 60)
    holes = tuple(Hole(35 + i * 8, 35, 7 - i) for i in range(4))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.4)
    refinement_references = []

    def locate(frame, reference):
        if frame.shape == blank.shape and reference is latest_disk:
            refinement_references.append(reference)
        return latest_disk, layout

    monkeypatch.setattr(cp02, "sample_frames", frames)
    monkeypatch.setattr(cp02, "locate_last_moving_multihole_disk", lambda _frames: coarse_disk)
    monkeypatch.setattr(cp02, "locate_disk_layout_near", locate)
    monkeypatch.setattr(cp02, "aligned_hole_opposite_handle", lambda *_args: 1)
    monkeypatch.setattr(cp02, "_hole_green_ratio", lambda *_args: 0.0)

    cp02.Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=tmp_path / "evidence",
    ).extract(
        tmp_path / "unused.mp4", "cp_02", TimeRange(start_sec=10, end_sec=20),
        dense_fps=5, analysis_width=1280,
    )

    assert refinement_references


def test_cp02_tail_windows_move_backward_without_scanning_future_cp03(tmp_path: Path):
    from medical_evaluation.extractors.cp02 import Cp02FeatureExtractor

    windows = list(Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=tmp_path,
    )._tail_search_windows(TimeRange(start_sec=10.0, end_sec=22.0)))

    assert [(item.start_sec, item.end_sec) for item in windows] == [
        (17.0, 22.0), (12.0, 17.0), (10.0, 12.0),
    ]


def test_cp02_dense_windows_cover_early_and_late_moving_disk_episodes(
    monkeypatch, tmp_path: Path,
):
    from medical_evaluation.extractors import cp02

    image = np.zeros((40, 60, 3), np.uint8)
    coarse = [
        SampledFrame(time_sec=10.0 + i * 0.5, frame_index=i, image_bgr=image)
        for i in range(15)
    ]
    candidate = MovingDisk(1, 30, 20, 10, 5, 0.5, 60)
    fallback = MovingDisk(14, 30, 20, 10, 5, 0.5, 60)
    monkeypatch.setattr(
        cp02, "locate_moving_multihole_disk", lambda _frames: candidate,
    )
    extractor = cp02.Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=tmp_path / "evidence",
    )

    windows = extractor._candidate_dense_windows(
        coarse, fallback, TimeRange(start_sec=10, end_sec=20),
    )

    assert windows[0][0].start_sec == 10
    assert windows[-1][0].end_sec >= 18.0


def test_cp02_extractor_uses_only_final_cp02_green_state(
    monkeypatch, tmp_path: Path,
):
    from medical_evaluation.extractors import cp02

    blank = np.full((240, 320, 3), (170, 120, 70), np.uint8)

    def frames(_path, *, start_sec, end_sec, sample_fps):
        times = (18.0, 18.5, 19.0) if sample_fps <= 2.1 else (18.8, 19.0, 19.8, 20.0)
        return iter(
            SampledFrame(time_sec=t, frame_index=round(t * 10), image_bgr=blank.copy())
            for t in times
            if start_sec <= t < end_sec
        )

    disk = MovingDisk(2, 150, 130, 45, 5, 0.5, 60)
    holes = tuple(Hole(120 + i * 15, 120, 8 - i) for i in range(5))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.12)
    monkeypatch.setattr(cp02, "sample_frames", frames)
    monkeypatch.setattr(cp02, "locate_last_moving_multihole_disk", lambda _frames: disk)
    monkeypatch.setattr(cp02, "locate_disk_layout_near", lambda _frame, _disk: (disk, layout))
    monkeypatch.setattr(cp02, "aligned_hole_opposite_handle", lambda *_args: 1)
    monkeypatch.setattr(cp02, "_hole_green_ratio", lambda *_args: 0.6)

    result = cp02.Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=tmp_path / "evidence"
    ).extract(
        tmp_path / "unused.mp4",
        "cp_02",
        TimeRange(start_sec=10, end_sec=20),
        dense_fps=5,
        analysis_width=1280,
    )

    assert result.features["residue_before"] is True
    assert result.features["cleanup_contact_observed"] is None
    assert result.features["residue_after"] is None


def test_cp02_green_ratio_only_uses_visible_target_hole_interior() -> None:
    from medical_evaluation.extractors.cp02 import _hole_green_ratio

    frame = np.full((120, 160, 3), (170, 120, 70), np.uint8)
    frame[:30] = (20, 170, 30)  # green dam reference away from the hole
    hole = Hole(80, 70, 12)
    yy, xx = np.ogrid[:120, :160]
    interior = (xx - hole.x) ** 2 + (yy - hole.y) ** 2 <= (0.65 * hole.radius) ** 2
    frame[interior] = (15, 15, 15)

    assert _hole_green_ratio(frame, hole) == 0.0

    frame[interior] = (20, 170, 30)
    assert _hole_green_ratio(frame, hole) is not None
    assert _hole_green_ratio(frame, hole) > 0.9


def test_dense_measurement_uses_monotonic_arc_rank_not_raw_pixel_area(
    monkeypatch, tmp_path: Path,
):
    from medical_evaluation.extractors import cp02

    center = (100.0, 100.0)
    holes = []
    for radius, angle in zip((7.8, 8.0, 5.0, 3.0), (210, 245, 280, 315), strict=True):
        radians = np.deg2rad(angle)
        holes.append(Hole(
            center[0] + 43 * np.cos(radians),
            center[1] + 43 * np.sin(radians),
            radius,
        ))
    disk = MovingDisk(0, *center, 70, 4, 0.5, 60)
    layout = DiskHoleLayout(tuple(holes), True, "criteria_satisfied", 1, 0.4)
    frame = SampledFrame(
        time_sec=1.0, frame_index=25,
        image_bgr=np.zeros((220, 220, 3), np.uint8),
    )
    monkeypatch.setattr(cp02, "locate_disk_layout_near", lambda *_args: (disk, layout))
    monkeypatch.setattr(cp02, "aligned_hole_opposite_handle", lambda *_args: 1)

    measured = cp02.Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=tmp_path,
    )._measure_dense([frame], disk)

    assert len(measured) == 1
    assert measured[0].size_rank == 2


def test_dense_measurement_tracks_from_each_reliable_disk(monkeypatch, tmp_path: Path):
    from medical_evaluation.extractors import cp02

    image = np.zeros((120, 160, 3), np.uint8)
    initial = MovingDisk(0, 50, 50, 25, 4, 0.5, 60)
    moved = MovingDisk(0, 65, 55, 25, 4, 0.5, 60)
    holes = tuple(Hole(40 + i * 9, 45, 8 - i) for i in range(4))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.4)
    references: list[MovingDisk] = []

    def locate(_frame, reference):
        references.append(reference)
        return (moved, layout)

    monkeypatch.setattr(cp02, "locate_disk_layout_near", locate)
    monkeypatch.setattr(cp02, "aligned_hole_opposite_handle", lambda *_args: 1)
    monkeypatch.setattr(cp02, "_hole_green_ratio", lambda *_args: 0.0)
    frames = [
        SampledFrame(time_sec=float(i), frame_index=i, image_bgr=image.copy())
        for i in range(2)
    ]

    cp02.Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=tmp_path,
    )._measure_dense(frames, initial)

    assert references == [initial, moved]


def test_cp02_green_ratio_rejects_bright_occluded_target_hole() -> None:
    from medical_evaluation.extractors.cp02 import _hole_green_ratio

    frame = np.full((120, 160, 3), (170, 120, 70), np.uint8)
    frame[:30] = (20, 170, 30)

    assert _hole_green_ratio(frame, Hole(80, 70, 12)) is None


def test_cp02_green_ratio_rejects_dark_tool_crossing_hole_edge() -> None:
    from medical_evaluation.extractors.cp02 import _hole_green_ratio

    frame = np.full((120, 160, 3), (180, 180, 180), np.uint8)
    frame[:30] = (20, 170, 30)
    hole = Hole(80, 70, 12)
    yy, xx = np.ogrid[:120, :160]
    interior = (xx - hole.x) ** 2 + (yy - hole.y) ** 2 <= hole.radius**2
    frame[interior] = 15
    frame[50:91, 77:84] = 15

    assert _hole_green_ratio(frame, hole) is None
