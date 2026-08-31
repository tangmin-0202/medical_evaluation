from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import SegmentationPrompt, normalize_masks
from medical_evaluation.segmentation.sam2_backend import Sam2Backend
from tests.fixtures.make_test_video import make_test_video


class FakeVideoPredictor:
    def __init__(self, *, fail_on_add: bool = False) -> None:
        self.add_calls: list[dict[str, object]] = []
        self.propagate_calls: list[tuple[int, bool, int]] = []
        self.init_video_path: Path | None = None
        self.frame_count = 0
        self.reset = False
        self.fail_on_add = fail_on_add

    def init_state(self, *, video_path: str) -> dict[str, object]:
        self.init_video_path = Path(video_path)
        self.frame_count = len(list(self.init_video_path.glob("*.jpg")))
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
        stop = (
            max(-1, start_frame_idx - max_frame_num_to_track)
            if reverse
            else min(self.frame_count, start_frame_idx + max_frame_num_to_track)
        )
        indexes = (
            range(start_frame_idx, stop, -1)
            if reverse
            else range(start_frame_idx, stop)
        )
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
    tmp_path: Path,
) -> None:
    predictor = FakeVideoPredictor()
    video = make_test_video(
        tmp_path / "video.mp4",
        fps=10,
        seconds=3,
        size=(100, 50),
    )
    backend = Sam2Backend("cfg", tmp_path / "weights.pt", predictor=predictor)
    prompts = [
        SegmentationPrompt(
            object_id="rubber_dam_frame",
            kind="point",
            frame_time_sec=1.3,
            coordinates=[0.2, 0.3],
        ),
        SegmentationPrompt(
            object_id="rubber_dam_frame",
            kind="point",
            frame_time_sec=1.3,
            coordinates=[0.8, 0.7],
        ),
    ]

    frames = list(
        backend.track(
            video,
            TimeRange(start_sec=1.0, end_sec=2.5),
            prompts,
            sample_fps=2,
        )
    )

    assert predictor.init_video_path is not None
    sampled_directory = predictor.init_video_path
    assert sampled_directory.is_dir() is False
    assert sampled_directory.suffix != ".mp4"
    assert len(predictor.add_calls) == 1
    prompt_call = predictor.add_calls[0]
    assert np.asarray(prompt_call["points"]).shape == (2, 2)
    np.testing.assert_allclose(prompt_call["points"], [[20.0, 15.0], [80.0, 35.0]])
    assert prompt_call["normalize_coords"] is True
    assert prompt_call["frame_idx"] == 1
    assert predictor.propagate_calls == [(1, False, 3), (1, True, 2)]
    assert [frame.frame_index for frame in frames] == [10, 13, 15, 20]
    assert [frame.frame_time_sec for frame in frames] == pytest.approx(
        [1.0, 1.3, 1.5, 2.0]
    )
    assert predictor.reset is True


def test_sam2_resets_state_when_prompt_submission_fails(
    tmp_path: Path,
) -> None:
    predictor = FakeVideoPredictor(fail_on_add=True)
    video = make_test_video(
        tmp_path / "video.mp4",
        fps=10,
        seconds=1,
        size=(100, 50),
    )
    backend = Sam2Backend("cfg", tmp_path / "weights.pt", predictor=predictor)

    with pytest.raises(RuntimeError, match="prompt add failed"):
        list(
            backend.track(
                video,
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

    assert predictor.init_video_path is not None
    assert predictor.init_video_path.exists() is False
    assert predictor.reset is True


def test_sam2_keeps_cuda_autocast_active_through_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = False
    context_checks: list[bool] = []
    autocast_calls: list[tuple[str, object]] = []
    fake_dtype = object()

    class FakeAutocast:
        def __enter__(self) -> None:
            nonlocal active
            active = True

        def __exit__(self, *_args: object) -> None:
            nonlocal active
            active = False

    def autocast(device_type: str, *, dtype: object) -> FakeAutocast:
        autocast_calls.append((device_type, dtype))
        return FakeAutocast()

    class AutocastCheckingPredictor(FakeVideoPredictor):
        def init_state(self, *, video_path: str) -> dict[str, object]:
            context_checks.append(active)
            return super().init_state(video_path=video_path)

        def add_new_points_or_box(self, **kwargs: object) -> None:
            context_checks.append(active)
            super().add_new_points_or_box(**kwargs)

        def propagate_in_video(
            self,
            state: object,
            *,
            start_frame_idx: int,
            reverse: bool,
            max_frame_num_to_track: int,
        ):
            context_checks.append(active)
            yield from super().propagate_in_video(
                state,
                start_frame_idx=start_frame_idx,
                reverse=reverse,
                max_frame_num_to_track=max_frame_num_to_track,
            )

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(bfloat16=fake_dtype, autocast=autocast),
    )
    predictor = AutocastCheckingPredictor()
    backend = Sam2Backend(
        "cfg",
        tmp_path / "weights.pt",
        device="cuda:0",
        predictor=predictor,
    )
    video = make_test_video(
        tmp_path / "video.mp4",
        fps=10,
        seconds=1,
        size=(100, 50),
    )

    list(
        backend.track(
            video,
            TimeRange(start_sec=0, end_sec=1),
            [
                SegmentationPrompt(
                    object_id="rubber_dam_frame",
                    kind="point",
                    frame_time_sec=0.5,
                    coordinates=[0.5, 0.5],
                )
            ],
            sample_fps=1,
        )
    )

    assert autocast_calls == [("cuda", fake_dtype)]
    assert context_checks
    assert all(context_checks)
    assert active is False


def test_backend_import_does_not_require_sam_packages() -> None:
    from medical_evaluation.segmentation.sam3_backend import Sam3Backend

    assert Sam3Backend.__name__ == "Sam3Backend"
