from pathlib import Path

import cv2
import numpy as np
import pytest

from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp03 import Cp03FeatureExtractor
from medical_evaluation.segmentation.base import FrameMasks
from medical_evaluation.segmentation.sam3_backend import Sam3AmbiguousTextResult
from medical_evaluation.video import SampledFrame


def _frame_and_mask(*, flap: bool = False, hole: bool = True):
    frame = np.full((160, 200, 3), (40, 160, 40), np.uint8)
    mask = np.ones((160, 200), np.uint8)
    if hole:
        cv2.circle(frame, (100, 80), 24, (90, 70, 55), -1)
        cv2.circle(mask, (100, 80), 24, 0, -1)
    if flap:
        cv2.rectangle(frame, (97, 56), (103, 79), (40, 160, 40), -1)
        cv2.rectangle(mask, (97, 56), (103, 79), 1, -1)
    return frame, mask.astype(bool)


class FakeSegmenter:
    model_version = "fake-sam3"

    def __init__(self, windows: list[list[FrameMasks]]) -> None:
        self.windows = windows
        self.calls = []

    def track(self, video_path, time_range, prompts, sample_fps):
        self.calls.append((video_path, time_range, prompts, sample_fps))
        return iter(self.windows[len(self.calls) - 1])


class AmbiguousSegmenter:
    model_version = "fake-sam3"

    def track(self, video_path, time_range, prompts, sample_fps):
        raise Sam3AmbiguousTextResult("multiple unranked candidates")


def _tracked(masks: list[np.ndarray], *, start_sec: float = 6.0):
    return [
        FrameMasks(
            frame_index=index,
            frame_time_sec=start_sec + index * 0.5,
            masks={"rubber_dam": mask},
        )
        for index, mask in enumerate(masks)
    ]


def test_uses_final_four_seconds_at_two_fps_and_saves_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, mask = _frame_and_mask()
    segmenter = FakeSegmenter([_tracked([mask, mask, mask])])
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp03.read_frame", lambda _path, _index: frame.copy()
    )
    extractor = Cp03FeatureExtractor(segmenter=segmenter, evidence_root=tmp_path)

    result = extractor.extract(
        tmp_path / "video.mp4",
        "cp_03",
        TimeRange(start_sec=0, end_sec=10),
        dense_fps=5,
        analysis_width=1280,
    )

    call = segmenter.calls[0]
    assert call[1] == TimeRange(start_sec=6, end_sec=10)
    assert call[3] == 2.0
    assert call[2][0].object_id == "rubber_dam"
    assert result.features["hole_clear_consecutive_frames"] >= 1.0
    assert result.features["hole_adhesion_free"] is True
    assert len(result.evidence) == 1
    for evidence in result.evidence:
        assert (tmp_path / evidence.overlay_path).is_file()


def test_expands_to_final_eight_seconds_when_initial_window_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, mask = _frame_and_mask()
    _, solid = _frame_and_mask(hole=False)
    segmenter = FakeSegmenter(
        [_tracked([solid, solid]), _tracked([mask, mask, mask], start_sec=2.0)]
    )
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp03.read_frame", lambda _path, _index: frame.copy()
    )
    extractor = Cp03FeatureExtractor(segmenter=segmenter, evidence_root=tmp_path)

    result = extractor.extract(
        tmp_path / "video.mp4",
        "cp_03",
        TimeRange(start_sec=0, end_sec=10),
        dense_fps=1,
        analysis_width=960,
    )

    assert [call[1] for call in segmenter.calls] == [
        TimeRange(start_sec=6, end_sec=10),
        TimeRange(start_sec=2, end_sec=10),
    ]
    assert result.features["hole_adhesion_free"] is True


def test_latest_clear_frame_controls_simple_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_frame, clear_mask = _frame_and_mask()
    flap_frame, flap_mask = _frame_and_mask(flap=True)
    frames = {0: flap_frame, 1: flap_frame, 2: clear_frame}
    segmenter = FakeSegmenter([_tracked([flap_mask, flap_mask, clear_mask])])
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp03.read_frame",
        lambda _path, index: frames[index].copy(),
    )
    result = Cp03FeatureExtractor(segmenter=segmenter, evidence_root=tmp_path).extract(
        tmp_path / "video.mp4",
        "cp_03",
        TimeRange(start_sec=0, end_sec=10),
        dense_fps=5,
        analysis_width=1280,
    )

    assert result.features["adhesion_observed_frame_count"] == 0.0
    assert result.features["hole_adhesion_free"] is True


def test_latest_clear_hole_frame_is_used_without_complex_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_frame, clear_mask = _frame_and_mask()
    flap_frame, flap_mask = _frame_and_mask(flap=True)
    frames = {0: clear_frame, 1: flap_frame, 2: clear_frame}
    segmenter = FakeSegmenter([_tracked([clear_mask, flap_mask, clear_mask])])
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp03.read_frame",
        lambda _path, index: frames[index].copy(),
    )
    result = Cp03FeatureExtractor(segmenter=segmenter, evidence_root=tmp_path).extract(
        tmp_path / "video.mp4",
        "cp_03",
        TimeRange(start_sec=0, end_sec=10),
        dense_fps=5,
        analysis_width=1280,
    )

    assert result.features["hole_observed"] is True
    assert result.features["hole_adhesion_free"] is True


def test_reliable_frames_without_a_hole_are_incomplete_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, solid = _frame_and_mask(hole=False)
    segmenter = FakeSegmenter(
        [_tracked([solid, solid]), _tracked([solid, solid, solid], start_sec=2.0)]
    )
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp03.read_frame", lambda _path, _index: frame.copy()
    )
    result = Cp03FeatureExtractor(segmenter=segmenter, evidence_root=tmp_path).extract(
        tmp_path / "video.mp4",
        "cp_03",
        TimeRange(start_sec=0, end_sec=10),
        dense_fps=5,
        analysis_width=1280,
    )

    assert result.features["final_scan_reliable"] is True
    assert result.features["hole_observed"] is False
    assert result.features["hole_adhesion_free"] is None


def test_ambiguous_sam_dam_candidates_return_review_instead_of_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp03.sample_frames",
        lambda *_args, **_kwargs: iter(()),
    )
    result = Cp03FeatureExtractor(
        segmenter=AmbiguousSegmenter(), evidence_root=tmp_path
    ).extract(
        tmp_path / "video.mp4",
        "cp_03",
        TimeRange(start_sec=0, end_sec=10),
        dense_fps=5,
        analysis_width=1280,
    )

    assert result.features["final_scan_reliable"] is False
    assert result.features["hole_observed"] is None
    assert result.features["hole_adhesion_free"] is None
    assert result.evidence == []


def test_text_miss_falls_back_to_automatic_green_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, mask = _frame_and_mask()
    segmenter = FakeSegmenter([[], _tracked([mask], start_sec=3.5)])
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp03.sample_frames",
        lambda *_args, **_kwargs: iter(
            [SampledFrame(time_sec=3.5, frame_index=7, image_bgr=frame.copy())]
        ),
    )
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp03.read_frame", lambda _path, _index: frame.copy()
    )

    result = Cp03FeatureExtractor(segmenter=segmenter, evidence_root=tmp_path).extract(
        tmp_path / "video.mp4",
        "cp_03",
        TimeRange(start_sec=0, end_sec=4),
        dense_fps=5,
        analysis_width=1280,
    )

    assert len(segmenter.calls) == 2
    assert segmenter.calls[0][2][0].kind == "text"
    assert segmenter.calls[0][2][0].text == "green dental rubber dam cloth"
    box_prompt = segmenter.calls[1][2][0]
    assert box_prompt.kind == "box"
    assert box_prompt.frame_time_sec == 3.5
    assert box_prompt.coordinates is not None
    assert result.features["hole_adhesion_free"] is True
