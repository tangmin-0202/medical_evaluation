from pathlib import Path

import pytest
from pydantic import ValidationError

from medical_evaluation.domain import CheckpointStatus, TimeRange


def make_segment(checkpoint_id: str, start: float, end: float):
    from medical_evaluation.annotations import SegmentAnnotation

    return SegmentAnnotation(
        checkpoint_id=checkpoint_id,
        time_range=TimeRange(start_sec=start, end_sec=end),
        label=CheckpointStatus.CORRECT,
        reason="点位正确",
    )


def test_annotation_store_round_trips_and_records_audit(tmp_path: Path) -> None:
    from medical_evaluation.annotations import AnnotationStore

    store = AnnotationStore(tmp_path)
    item = make_segment("cp_01", 0, 14)

    saved = store.save_segments("success", [item], actor="tangmin")
    loaded = store.load_segments("success")

    assert loaded.steps == [item]
    assert saved.audit_history[-1].actor == "tangmin"
    assert saved.audit_history[-1].prior_digest is None
    assert len(saved.audit_history[-1].new_digest) == 64


def test_annotations_reject_duplicate_checkpoint_ids() -> None:
    from medical_evaluation.annotations import VideoAnnotations

    with pytest.raises(ValidationError, match="duplicate checkpoint IDs"):
        VideoAnnotations(video_id="success", steps=[make_segment("cp_01", 0, 10)] * 2)


def test_annotations_reject_overlapping_steps() -> None:
    from medical_evaluation.annotations import VideoAnnotations

    with pytest.raises(ValidationError, match="step ranges must not overlap"):
        VideoAnnotations(
            video_id="success",
            steps=[make_segment("cp_01", 0, 10), make_segment("cp_02", 9, 15)],
        )


def test_point_prompt_coordinates_are_normalized() -> None:
    from medical_evaluation.annotations import PointPrompt

    prompt = PointPrompt(
        video_id="success",
        frame_time_sec=1.5,
        object_id="rubber_dam",
        x=0.5,
        y=0.25,
    )
    assert (prompt.x, prompt.y) == (0.5, 0.25)

    with pytest.raises(ValidationError):
        PointPrompt(video_id="success", frame_time_sec=1.5, object_id="rubber_dam", x=1.1, y=0.25)
