from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import SegmentationPrompt, normalize_masks
from medical_evaluation.segmentation.sam2_backend import Sam2Backend
from medical_evaluation.video import VideoMetadata


class FakeVideoPredictor:
    def __init__(self, *, fail_on_add: bool = False) -> None:
        self.add_calls: list[dict[str, object]] = []
        self.propagate_calls: list[tuple[int, bool, int]] = []
        self.reset = False
        self.fail_on_add = fail_on_add

    def init_state(self, *, video_path: str) -> dict[str, object]:
        return {"video_path": video_path}

    def add_new_points_or_box(self, **kwargs: object) -> None:
        self.add_calls.append(kwargs)
        if self.fail_on_add:
            raise RuntimeError("prompt add failed")

    def propagate_in_video(
        self,
        _state: object,
        *,
        start_frame_idx: int,
        reverse: bool,
        max_frame_num_to_track: int,
    ) -> list[tuple[int, list[int], np.ndarray]]:
        self.propagate_calls.append((start_frame_idx, reverse, max_frame_num_to_track))
        indexes = [190, 180, 170] if reverse else [190, 200]
        return [
            (index, [1], np.ones((1, 1, 4, 4), dtype=np.float32))
            for index in indexes
        ]

    def reset_state(self, _state: object) -> None:
        self.reset = True


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


def test_sam2_batches_same_frame_points_and_tracks_both_directions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predictor = FakeVideoPredictor()
    monkeypatch.setattr(
        "medical_evaluation.segmentation.sam2_backend.probe_video",
        lambda _path: VideoMetadata(
            path=tmp_path / "video.mp4",
            duration_sec=30,
            fps=10,
            frame_count=300,
            width=100,
            height=50,
        ),
    )
    backend = Sam2Backend("cfg", tmp_path / "weights.pt", predictor=predictor)
    prompts = [
        SegmentationPrompt(
            object_id="rubber_dam_frame",
            kind="point",
            frame_time_sec=19,
            coordinates=[0.2, 0.3],
        ),
        SegmentationPrompt(
            object_id="rubber_dam_frame",
            kind="point",
            frame_time_sec=19,
            coordinates=[0.8, 0.7],
        ),
    ]

    frames = list(
        backend.track(
            tmp_path / "video.mp4",
            TimeRange(start_sec=17, end_sec=20),
            prompts,
            sample_fps=1,
        )
    )

    assert len(predictor.add_calls) == 1
    assert np.asarray(predictor.add_calls[0]["points"]).shape == (2, 2)
    assert predictor.propagate_calls == [(190, False, 11), (190, True, 21)]
    assert [frame.frame_index for frame in frames] == [170, 180, 190, 200]
    assert predictor.reset is True


def test_sam2_resets_state_when_prompt_submission_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predictor = FakeVideoPredictor(fail_on_add=True)
    monkeypatch.setattr(
        "medical_evaluation.segmentation.sam2_backend.probe_video",
        lambda _path: VideoMetadata(
            path=tmp_path / "video.mp4",
            duration_sec=1,
            fps=10,
            frame_count=10,
            width=100,
            height=50,
        ),
    )
    backend = Sam2Backend("cfg", tmp_path / "weights.pt", predictor=predictor)

    with pytest.raises(RuntimeError, match="prompt add failed"):
        list(
            backend.track(
                tmp_path / "video.mp4",
                TimeRange(start_sec=0, end_sec=1),
                [
                    SegmentationPrompt(
                        object_id="frame",
                        kind="point",
                        frame_time_sec=0.5,
                        coordinates=[0.5, 0.5],
                    )
                ],
                sample_fps=1,
            )
        )

    assert predictor.reset is True


def test_backend_import_does_not_require_sam_packages() -> None:
    from medical_evaluation.segmentation.sam3_backend import Sam3Backend

    assert Sam3Backend.__name__ == "Sam3Backend"
