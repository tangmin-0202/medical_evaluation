from pathlib import Path

import numpy as np
import pytest

from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import SegmentationPrompt
from medical_evaluation.segmentation.sam3_backend import (
    Sam3AmbiguousTextResult,
    Sam3Backend,
)
from tests.fixtures.make_test_video import make_test_video


class FakeSam3Predictor:
    def __init__(
        self,
        *,
        object_ids: tuple[int, ...] = (7,),
        scores: tuple[float, ...] | None = None,
        fail_stream: bool = False,
        detect_at_frame: int = 0,
    ) -> None:
        self.object_ids = object_ids
        self.scores = scores
        self.fail_stream = fail_stream
        self.detect_at_frame = detect_at_frame
        self.requests: list[dict[str, object]] = []
        self.stream_requests: list[dict[str, object]] = []

    def handle_request(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "session-1"}
        if request["type"] == "add_prompt":
            object_ids = (
                self.object_ids
                if int(request["frame_index"]) >= self.detect_at_frame
                else ()
            )
            outputs: dict[str, object] = {
                "out_obj_ids": np.asarray(object_ids),
                "out_binary_masks": np.ones((len(object_ids), 20, 40), dtype=bool),
            }
            if self.scores is not None:
                outputs["out_scores"] = np.asarray(self.scores)
            return {"frame_index": 0, "outputs": outputs}
        return {"is_success": True}

    def handle_stream_request(self, request: dict[str, object]):
        self.stream_requests.append(request)
        if self.fail_stream:
            raise RuntimeError("stream failed")
        start = int(request["start_frame_index"])
        local_indices = [0, 1] if start == 0 else [1, 0]
        for local_index in local_indices:
            yield {
                "frame_index": local_index,
                "outputs": {
                    "out_obj_ids": np.asarray(self.object_ids),
                    "out_binary_masks": np.ones(
                        (len(self.object_ids), 20, 40), dtype=bool
                    ),
                },
            }


class StrictMultiplexModel:
    batched_grounding_batch_size = 16

    def init_state(
        self,
        resource_path: str,
        offload_video_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> dict[str, object]:
        return {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "async_loading_frames": async_loading_frames,
        }


class BaseStylePredictor(FakeSam3Predictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = StrictMultiplexModel()

    def handle_request(self, request: dict[str, object]) -> dict[str, object]:
        if request["type"] == "start_session":
            self.requests.append(request)
            self.model.init_state(
                resource_path=str(request["resource_path"]),
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )
            return {"session_id": "session-1"}
        return super().handle_request(request)


class MissingTrackedObjectPredictor(FakeSam3Predictor):
    def handle_stream_request(self, request: dict[str, object]):
        self.stream_requests.append(request)
        yield {
            "frame_index": 0,
            "outputs": {
                "out_obj_ids": np.asarray(self.object_ids),
                "out_binary_masks": np.ones((1, 20, 40), dtype=bool),
            },
        }
        yield {
            "frame_index": 1,
            "outputs": {
                "out_obj_ids": np.asarray([], dtype=int),
                "out_binary_masks": np.empty((0, 20, 40), dtype=bool),
            },
        }


def _text_prompt() -> SegmentationPrompt:
    return SegmentationPrompt(
        object_id="rubber_dam_frame",
        kind="text",
        frame_time_sec=1.0,
        text="white U-shaped dental frame",
    )


def test_sam3_uses_bounded_sequence_and_maps_source_frames(tmp_path: Path) -> None:
    predictor = FakeSam3Predictor()
    video = make_test_video(tmp_path / "video.mp4", fps=10, seconds=3, size=(40, 20))
    backend = Sam3Backend(tmp_path / "sam3.pt", predictor=predictor)

    frames = list(
        backend.track(
            video,
            TimeRange(start_sec=1, end_sec=3),
            [_text_prompt()],
            sample_fps=1,
        )
    )

    start = predictor.requests[0]
    assert start["type"] == "start_session"
    assert Path(str(start["resource_path"])).name == "frames"
    assert start["offload_video_to_cpu"] is True
    assert predictor.requests[1] == {
        "type": "add_prompt",
        "session_id": "session-1",
        "frame_index": 0,
        "text": "white U-shaped dental frame",
        "output_prob_thresh": 0.5,
    }
    assert predictor.stream_requests == [
        {
            "type": "propagate_in_video",
            "session_id": "session-1",
            "propagation_direction": "both",
            "start_frame_index": 0,
            "output_prob_thresh": 0.5,
        }
    ]
    assert [item.frame_index for item in frames] == [10, 20]
    assert all(set(item.masks) == {"rubber_dam_frame"} for item in frames)
    assert predictor.requests[-1]["type"] == "close_session"
    assert not Path(str(start["resource_path"])).exists()


@pytest.mark.parametrize("prompts", [[], [_text_prompt(), _text_prompt()]])
def test_sam3_requires_exactly_one_prompt(tmp_path: Path, prompts) -> None:
    backend = Sam3Backend(tmp_path / "sam3.pt", predictor=FakeSam3Predictor())
    with pytest.raises(ValueError, match="exactly one text prompt"):
        list(
            backend.track(
                tmp_path / "unused.mp4",
                TimeRange(start_sec=1, end_sec=2),
                prompts,
                sample_fps=1,
            )
        )


def test_sam3_rejects_non_text_prompt(tmp_path: Path) -> None:
    prompt = SegmentationPrompt(
        object_id="rubber_dam_frame",
        kind="point",
        frame_time_sec=1,
        coordinates=[0.5, 0.5],
    )
    backend = Sam3Backend(tmp_path / "sam3.pt", predictor=FakeSam3Predictor())
    with pytest.raises(ValueError, match="exactly one text prompt"):
        list(
            backend.track(
                tmp_path / "unused.mp4",
                TimeRange(start_sec=1, end_sec=2),
                [prompt],
                sample_fps=1,
            )
        )


def test_sam3_returns_no_frames_when_text_finds_no_candidate(tmp_path: Path) -> None:
    predictor = FakeSam3Predictor(object_ids=())
    video = make_test_video(tmp_path / "video.mp4", fps=10, seconds=3, size=(40, 20))
    backend = Sam3Backend(tmp_path / "sam3.pt", predictor=predictor)

    assert list(
        backend.track(
            video,
            TimeRange(start_sec=1, end_sec=3),
            [_text_prompt()],
            sample_fps=1,
        )
    ) == []
    assert predictor.requests[-1]["type"] == "close_session"


def test_sam3_selects_highest_scored_candidate(tmp_path: Path) -> None:
    predictor = FakeSam3Predictor(object_ids=(7, 8), scores=(0.2, 0.9))
    video = make_test_video(tmp_path / "video.mp4", fps=10, seconds=3, size=(40, 20))
    backend = Sam3Backend(tmp_path / "sam3.pt", predictor=predictor)

    frames = list(
        backend.track(
            video,
            TimeRange(start_sec=1, end_sec=3),
            [_text_prompt()],
            sample_fps=1,
        )
    )

    assert len(frames) == 2
    assert all(frame.masks["rubber_dam_frame"].all() for frame in frames)


def test_sam3_represents_tracked_object_absence_with_empty_mask(tmp_path: Path) -> None:
    predictor = MissingTrackedObjectPredictor()
    video = make_test_video(tmp_path / "video.mp4", fps=10, seconds=3, size=(40, 20))
    backend = Sam3Backend(tmp_path / "sam3.pt", predictor=predictor)

    frames = list(
        backend.track(
            video,
            TimeRange(start_sec=1, end_sec=3),
            [_text_prompt()],
            sample_fps=1,
        )
    )

    assert frames[0].masks["rubber_dam_frame"].all()
    assert frames[1].masks["rubber_dam_frame"].shape == (20, 40)
    assert not frames[1].masks["rubber_dam_frame"].any()


def test_sam3_rejects_ambiguous_unscored_candidates(tmp_path: Path) -> None:
    predictor = FakeSam3Predictor(object_ids=(7, 8))
    video = make_test_video(tmp_path / "video.mp4", fps=10, seconds=3, size=(40, 20))
    backend = Sam3Backend(tmp_path / "sam3.pt", predictor=predictor)

    with pytest.raises(Sam3AmbiguousTextResult):
        list(
            backend.track(
                video,
                TimeRange(start_sec=1, end_sec=3),
                [_text_prompt()],
                sample_fps=1,
            )
        )
    assert predictor.requests[-1]["type"] == "close_session"


def test_sam3_closes_session_when_propagation_fails(tmp_path: Path) -> None:
    predictor = FakeSam3Predictor(fail_stream=True)
    video = make_test_video(tmp_path / "video.mp4", fps=10, seconds=3, size=(40, 20))
    backend = Sam3Backend(tmp_path / "sam3.pt", predictor=predictor)

    with pytest.raises(RuntimeError, match="stream failed"):
        list(
            backend.track(
                video,
                TimeRange(start_sec=1, end_sec=3),
                [_text_prompt()],
                sample_fps=1,
            )
        )
    assert predictor.requests[-1]["type"] == "close_session"


def test_sam3_scans_later_frames_then_tracks_both_directions(tmp_path: Path) -> None:
    predictor = FakeSam3Predictor(detect_at_frame=1)
    video = make_test_video(tmp_path / "video.mp4", fps=10, seconds=3, size=(40, 20))
    backend = Sam3Backend(tmp_path / "sam3.pt", predictor=predictor)

    frames = list(
        backend.track(
            video,
            TimeRange(start_sec=1, end_sec=3),
            [_text_prompt()],
            sample_fps=1,
        )
    )

    add_frames = [
        request["frame_index"]
        for request in predictor.requests
        if request["type"] == "add_prompt"
    ]
    assert add_frames == [0, 1]
    assert any(request["type"] == "reset_session" for request in predictor.requests)
    assert predictor.stream_requests[0]["propagation_direction"] == "both"
    assert predictor.stream_requests[0]["start_frame_index"] == 1
    assert [frame.frame_index for frame in frames] == [10, 20]


def test_sam3_filters_base_predictor_kwargs_for_multiplex_init_state(
    tmp_path: Path,
) -> None:
    predictor = BaseStylePredictor()
    video = make_test_video(tmp_path / "video.mp4", fps=10, seconds=3, size=(40, 20))
    backend = Sam3Backend(tmp_path / "sam3.pt", predictor=predictor)

    frames = list(
        backend.track(
            video,
            TimeRange(start_sec=1, end_sec=3),
            [_text_prompt()],
            sample_fps=1,
        )
    )

    assert len(frames) == 2


def test_sam3_limits_model_grounding_batch_size(tmp_path: Path) -> None:
    predictor = FakeSam3Predictor()
    predictor.model = type(
        "FakeMultiplexModel",
        (),
        {"batched_grounding_batch_size": 16},
    )()

    Sam3Backend(
        tmp_path / "sam3.pt",
        predictor=predictor,
        grounding_batch_size=4,
    )

    assert predictor.model.batched_grounding_batch_size == 4
