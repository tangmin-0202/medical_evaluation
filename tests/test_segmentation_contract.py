from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import SegmentationPrompt, normalize_masks
from medical_evaluation.segmentation.sam2_backend import Sam2Backend


def test_masks_are_boolean_and_keyed_by_object() -> None:
    raw = {"frame": 3, "objects": {"rubber_dam": np.array([[0.1, 0.9]])}}

    result = normalize_masks(raw, threshold=0.5, frame_time_sec=1.5)

    assert result.frame_index == 3
    assert result.frame_time_sec == 1.5
    assert result.masks["rubber_dam"].dtype == np.bool_
    assert result.masks["rubber_dam"].tolist() == [[False, True]]


def test_prompt_coordinates_are_normalized() -> None:
    prompt = SegmentationPrompt(object_id="frame", kind="point", coordinates=[0.5, 0.25])

    assert prompt.coordinates == [0.5, 0.25]
    with pytest.raises(ValidationError):
        SegmentationPrompt(object_id="frame", kind="point", coordinates=[1.5, 0.25])


def test_box_prompt_requires_ordered_corners() -> None:
    with pytest.raises(ValidationError, match="box corners"):
        SegmentationPrompt(object_id="clamp", kind="box", coordinates=[0.8, 0.2, 0.4, 0.9])


def test_sam2_rejects_text_only_prompts_before_inference(tmp_path: Path) -> None:
    backend = Sam2Backend(
        config_name="sam2.1_hiera_l.yaml",
        checkpoint=tmp_path / "weights.pt",
        predictor=object(),
    )

    with pytest.raises(ValueError, match="SAM3"):
        list(
            backend.track(
                tmp_path / "video.mp4",
                TimeRange(start_sec=0, end_sec=1),
                [SegmentationPrompt(object_id="rubber_dam", kind="text", text="green sheet")],
                sample_fps=1,
            )
        )


def test_backend_import_does_not_require_sam_packages() -> None:
    from medical_evaluation.segmentation.sam3_backend import Sam3Backend

    assert Sam3Backend.__name__ == "Sam3Backend"
