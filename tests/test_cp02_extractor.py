from pathlib import Path

import numpy as np

from medical_evaluation.annotations import SegmentAnnotation, VideoAnnotations
from medical_evaluation.domain import CheckpointStatus, TimeRange
from medical_evaluation.features.cp02_punch import DiskHoleLayout, Hole, MovingDisk
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

    def frames(_path, *, start_sec, end_sec, sample_fps):
        if sample_fps <= 2.1:
            times = (18.0, 18.5, 19.0)
        else:
            times = (18.8, 18.9, 19.0, 19.1, 19.2)
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
    assert result.features["punch_action_observed"] is True
    assert result.features["selected_hole_size_rank"] == 1
    assert result.features["selected_second_largest"] is False
    assert result.features["residue_before"] is False
    assert len(result.evidence) == 2
    assert all((tmp_path / "evidence" / item.overlay_path).is_file() for item in result.evidence)
    assert (tmp_path / "evidence" / "cp_02" / "hole_observations.json").is_file()


def test_cp02_extractor_keeps_unstable_final_selection_for_review(monkeypatch, tmp_path: Path):
    from medical_evaluation.extractors import cp02

    blank = np.full((240, 320, 3), (170, 120, 70), np.uint8)

    def frames(_path, *, start_sec, end_sec, sample_fps):
        times = (18.0, 18.5, 19.0) if sample_fps <= 2.1 else (18.8, 19.0)
        return iter(
            SampledFrame(time_sec=t, frame_index=round(t * 10), image_bgr=blank.copy())
            for t in times
            if start_sec <= t < end_sec
        )

    disk = MovingDisk(1, 150, 130, 45, 5, 0.5, 60)
    holes = tuple(Hole(120 + i * 15, 120, 8 - i) for i in range(5))
    layout = DiskHoleLayout(holes, True, "criteria_satisfied", 1, 0.12)
    selected = iter((0, 1))
    monkeypatch.setattr(cp02, "sample_frames", frames)
    monkeypatch.setattr(cp02, "locate_last_moving_multihole_disk", lambda _frames: disk)
    monkeypatch.setattr(cp02, "locate_disk_layout_near", lambda _frame, _disk: (disk, layout))
    monkeypatch.setattr(cp02, "aligned_hole_opposite_handle", lambda *_args: next(selected))

    result = cp02.Cp02FeatureExtractor(
        annotations=_annotations(), evidence_root=tmp_path / "evidence"
    ).extract(
        tmp_path / "unused.mp4",
        "cp_02",
        TimeRange(start_sec=10, end_sec=20),
        dense_fps=5,
        analysis_width=1280,
    )

    assert result.features["selected_second_largest"] is None
    assert result.features["selected_hole_size_rank"] is None
