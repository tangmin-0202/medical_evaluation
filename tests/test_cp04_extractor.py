from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp04 import (
    CLAMP_PROMPTS,
    HAND_PROMPT,
    Cp04FeatureExtractor,
)
from medical_evaluation.segmentation.base import FrameMasks
from medical_evaluation.segmentation.sam3_backend import Sam3AmbiguousTextResult


def test_hand_prompt_targets_the_open_display_palm() -> None:
    assert HAND_PROMPT == "open white gloved palm holding a small shiny metal clip"


def _clamp_mask() -> np.ndarray:
    mask = np.zeros((220, 220), np.uint8)
    cv2.ellipse(mask, (110, 118), (48, 62), 0, 205, 335, 1, 16)
    cv2.rectangle(mask, (49, 76), (88, 103), 1, -1)
    cv2.rectangle(mask, (132, 76), (171, 103), 1, -1)
    cv2.rectangle(mask, (67, 95), (84, 151), 1, -1)
    cv2.rectangle(mask, (136, 95), (153, 151), 1, -1)
    return mask.astype(bool)


def _hand_mask(*, left: bool = False) -> np.ndarray:
    mask = np.zeros((220, 220), np.uint8)
    center = (45, 110) if left else (110, 110)
    cv2.circle(mask, center, 100 if not left else 42, 1, -1)
    return mask.astype(bool)


def _tracked(object_id: str, masks: list[np.ndarray]) -> list[FrameMasks]:
    return [
        FrameMasks(
            frame_index=index,
            frame_time_sec=41.0 + index * 0.5,
            masks={object_id: mask},
        )
        for index, mask in enumerate(masks)
    ]


class FakeSegmenter:
    model_version = "fake-sam3"

    def __init__(self, outputs: list[list[FrameMasks] | Exception]) -> None:
        self.outputs = outputs
        self.calls = []

    def track(self, video_path, time_range, prompts, sample_fps):
        self.calls.append((video_path, time_range, prompts, sample_fps))
        output = self.outputs[len(self.calls) - 1]
        if isinstance(output, Exception):
            raise output
        return iter(output)


def _reference_dir(tmp_path: Path) -> Path:
    root = tmp_path / "reference"
    (root / "masks").mkdir(parents=True)
    assert cv2.imwrite(str(root / "masks" / "00000000.png"), _clamp_mask().astype(np.uint8) * 255)
    return root


def _frame() -> np.ndarray:
    y, x = np.indices((220, 220))
    base = ((x * 7 + y * 11) % 255).astype(np.uint8)
    return np.dstack((base, base, base))


def test_extracts_matching_clamp_from_independent_hand_and_clamp_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hand = _hand_mask()
    clamp = _clamp_mask()
    segmenter = FakeSegmenter(
        [_tracked("cp04_gloved_hand", [hand] * 3), _tracked("cp04_clamp", [clamp] * 3)]
    )
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame().copy(),
    )

    result = Cp04FeatureExtractor(
        segmenter=segmenter,
        evidence_root=tmp_path / "evidence",
        reference_dir=_reference_dir(tmp_path),
        min_similarity=0.8,
    ).extract(
        tmp_path / "video.mp4",
        "cp_04",
        TimeRange(start_sec=41, end_sec=47),
        dense_fps=5,
        analysis_width=1280,
    )

    assert [call[2][0].text for call in segmenter.calls] == [HAND_PROMPT, CLAMP_PROMPTS[0]]
    assert all(call[3] == 2.0 for call in segmenter.calls)
    assert result.features == {
        "clamp_observed": True,
        "shape_evidence_reliable": True,
        "clear_frame_count": 3.0,
        "matching_frame_count": 3.0,
        "clamp_reference_similarity": pytest.approx(1.0),
        "evidence_consistent": True,
    }
    assert len(result.evidence) == 3
    for item in result.evidence:
        assert (tmp_path / "evidence" / item.overlay_path).is_file()
    assert (tmp_path / "evidence" / "cp_04" / "frame_analysis.json").is_file()


def test_retries_approved_shape_prompt_only_after_primary_has_no_reliable_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hand = _hand_mask()
    clamp = _clamp_mask()
    outside = np.roll(clamp, 90, axis=1)
    segmenter = FakeSegmenter(
        [
            _tracked("cp04_gloved_hand", [hand] * 2),
            _tracked("cp04_clamp", [outside] * 2),
            _tracked("cp04_clamp", [clamp] * 2),
        ]
    )
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame().copy(),
    )

    result = Cp04FeatureExtractor(
        segmenter=segmenter,
        evidence_root=tmp_path / "evidence",
        reference_dir=_reference_dir(tmp_path),
    ).extract(
        tmp_path / "video.mp4",
        "cp_04",
        TimeRange(start_sec=41, end_sec=47),
        dense_fps=2,
        analysis_width=1280,
    )

    assert [call[2][0].text for call in segmenter.calls] == [
        HAND_PROMPT,
        CLAMP_PROMPTS[0],
        CLAMP_PROMPTS[1],
    ]
    assert result.features["clear_frame_count"] == 2.0


def test_clamp_on_rack_outside_glove_is_not_clear_shape_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    segmenter = FakeSegmenter(
        [
            _tracked("cp04_gloved_hand", [_hand_mask(left=True)] * 2),
            _tracked("cp04_clamp", [_clamp_mask()] * 2),
            [],
            [],
        ]
    )
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame().copy(),
    )

    result = Cp04FeatureExtractor(
        segmenter=segmenter,
        evidence_root=tmp_path / "evidence",
        reference_dir=_reference_dir(tmp_path),
    ).extract(
        tmp_path / "video.mp4",
        "cp_04",
        TimeRange(start_sec=41, end_sec=47),
        dense_fps=2,
        analysis_width=1280,
    )

    assert result.features["clamp_observed"] is True
    assert result.features["shape_evidence_reliable"] is False
    assert result.features["clear_frame_count"] == 0.0


def test_ambiguous_clamp_text_result_returns_review_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    segmenter = FakeSegmenter(
        [
            _tracked("cp04_gloved_hand", [_hand_mask()] * 2),
            Sam3AmbiguousTextResult("ambiguous"),
            [],
            [],
        ]
    )
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame().copy(),
    )

    result = Cp04FeatureExtractor(
        segmenter=segmenter,
        evidence_root=tmp_path / "evidence",
        reference_dir=_reference_dir(tmp_path),
    ).extract(
        tmp_path / "video.mp4",
        "cp_04",
        TimeRange(start_sec=41, end_sec=47),
        dense_fps=2,
        analysis_width=1280,
    )

    assert result.features["clamp_observed"] is False
    assert result.features["shape_evidence_reliable"] is False
    assert result.features["clamp_reference_similarity"] is None
