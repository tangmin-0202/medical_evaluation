from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_evaluation.annotations import (
    BoxPrompt,
    PointPrompt,
    SegmentAnnotation,
    VideoAnnotations,
)
from medical_evaluation.domain import CheckpointStatus
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.rubric import load_rubric
from scripts.smoke_sam2_cp09 import run_cp09_smoke


class FakeExtractor:
    @property
    def model_version(self) -> str:
        return "fake-sam2:test"

    def extract(self, *_args: object, **_kwargs: object) -> ExtractedEvidence:
        return ExtractedEvidence(
            features={
                "frame_oral_center_offset": 0.03,
                "frame_valid_count": 3.0,
                "oral_reference_count": 1.0,
                "relative_offset_valid_count": 3.0,
            }
        )


def _annotations() -> VideoAnnotations:
    rubric = load_rubric(Path("config/rubric.yaml"))
    return VideoAnnotations(
        video_id="success",
        steps=[
            SegmentAnnotation(
                checkpoint_id=item.id,
                time_range=item.reference_time,
                label=CheckpointStatus.CORRECT,
                reason="test evidence",
            )
            for item in rubric.checkpoints
        ],
        prompts=[
            PointPrompt(
                video_id="success",
                frame_time_sec=193,
                object_id="rubber_dam_frame",
                x=0.5,
                y=0.5,
            ),
            BoxPrompt(
                video_id="success",
                frame_time_sec=174.75,
                object_id="oral_region",
                x1=0.25,
                y1=0.2,
                x2=0.75,
                y2=0.8,
            ),
        ],
    )


def test_smoke_core_writes_summary_and_correct_decision(tmp_path: Path) -> None:
    summary = run_cp09_smoke(
        extractor=FakeExtractor(),
        video_path=tmp_path / "video.mp4",
        annotations=_annotations(),
        rubric=load_rubric(Path("config/rubric.yaml")),
        run_root=tmp_path / "run",
    )

    stored_summary = json.loads((tmp_path / "run" / "summary.json").read_text("utf-8"))
    decision = json.loads((tmp_path / "run" / "decision.json").read_text("utf-8"))
    assert summary == stored_summary
    assert summary["prompt_counts"] == {
        "rubber_dam_frame": 1,
        "oral_region": 1,
    }
    assert "prompt_count" not in summary
    assert summary["valid_frame_count"] == 3.0
    assert summary["oral_reference_count"] == 1.0
    assert summary["relative_offset_valid_count"] == 3.0
    assert "oral_region_valid_count" not in summary
    assert "paired_valid_count" not in summary
    assert summary["model_version"] == "fake-sam2:test"
    assert decision["status"] == "correct"


@pytest.mark.parametrize(
    ("video_id", "checkpoint_id", "message"),
    [("unsupported", "cp_09", "video_id"), ("success", "cp_10", "cp_09")],
)
def test_smoke_core_rejects_unsupported_scope(
    tmp_path: Path,
    video_id: str,
    checkpoint_id: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_cp09_smoke(
            extractor=FakeExtractor(),
            video_path=tmp_path / "video.mp4",
            annotations=_annotations(),
            rubric=load_rubric(Path("config/rubric.yaml")),
            run_root=tmp_path / "run",
            video_id=video_id,
            checkpoint_id=checkpoint_id,
        )
