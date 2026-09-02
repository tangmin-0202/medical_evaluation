from __future__ import annotations

import json
from pathlib import Path

from medical_evaluation.annotations import SegmentAnnotation, VideoAnnotations
from medical_evaluation.domain import CheckpointStatus, TimeRange
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.rubric import load_rubric
from scripts.smoke_sam2_cp01_cp11 import (
    build_cp01_extractor,
    run_cp01_cp11_smoke,
)


class FakeExtractor:
    model_version = "fake-combined"

    def __init__(self, features: dict[str, float | bool | None]) -> None:
        self.features = features
        self.calls: list[str] = []

    def extract(self, _video_path: Path, checkpoint_id: str, _time_range: TimeRange, **_kwargs: object) -> ExtractedEvidence:
        self.calls.append(checkpoint_id)
        return ExtractedEvidence(features=self.features)


def test_combined_smoke_writes_both_decisions_in_one_run(tmp_path: Path) -> None:
    annotations = VideoAnnotations(
        video_id="success",
        steps=[
            SegmentAnnotation(checkpoint_id=f"cp_{number:02d}", time_range=TimeRange(start_sec=number - 1, end_sec=number), label=CheckpointStatus.NEEDS_REVIEW, reason="test")
            for number in range(1, 12)
        ],
    )
    cp01 = FakeExtractor(
        {
            "dam_valid_frame_count": 10.0,
            "pen_contact_detected": True,
            "new_mark_candidate_count": 1.0,
            "mark_reference_distance": 0.01,
        }
    )
    cp11 = FakeExtractor({"dam_stage_presence_ratio": 0.5, "dam_final_presence_ratio": 0.8, "dam_area_ratio": 0.4, "nose_overlap": 0.0, "visible_frame_area_ratio": 0.0})
    run_root = tmp_path / "run"

    summary = run_cp01_cp11_smoke(
        cp01_extractor=cp01,
        cp11_extractor=cp11,
        video_path=tmp_path / "video.mp4",
        annotations=annotations,
        rubric=load_rubric(Path("config/rubric.yaml")),
        run_root=run_root,
        video_id="success",
        sample_fps=2,
    )

    assert cp01.calls == ["cp_01"]
    assert cp11.calls == ["cp_11"]
    assert set(summary["checkpoints"]) == {"cp_01", "cp_11"}
    decisions = json.loads((run_root / "decisions.json").read_text("utf-8"))
    assert decisions["cp_01"]["status"] == "correct"
    assert decisions["cp_11"]["status"] == "correct"


def test_build_cp01_extractor_uses_all_rubric_thresholds(tmp_path: Path) -> None:
    rubric = load_rubric(Path("config/rubric.yaml"))
    extractor = build_cp01_extractor(
        segmenter=FakeExtractor({}),
        annotations=VideoAnnotations(video_id="success"),
        evidence_root=tmp_path,
        reference={"reference_u": 0.7, "reference_v": 0.6},
        rubric=rubric,
    )

    assert extractor.min_pen_dam_overlap_ratio == 0.02
    assert extractor.min_pen_contact_frames == 2
    assert extractor.min_mark_observed_frames == 3
    assert extractor.min_new_mark_darkness_delta == 15.0
    assert extractor.min_mark_area_ratio == 0.0001
    assert extractor.max_mark_area_ratio == 0.01
    assert extractor.maximum_black_value == 55
    assert extractor.maximum_faint_value == 160
    assert extractor.maximum_faint_saturation == 120
    assert extractor.max_mark_aspect_ratio == 2.0
    assert extractor.min_mark_circularity == 0.35
    assert extractor.max_local_cluster_distance == 0.04
    assert extractor.local_darkness_ring_radius == 5
