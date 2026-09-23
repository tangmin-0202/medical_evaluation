from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp04 import (
    HAND_PROMPT,
    LOCAL_CLAMP_OBJECT_ID,
    LOCAL_CLAMP_PROMPT,
    LOCAL_ROI_CLAMP_PROMPT,
    LOCAL_ROI_LOCATOR_PROMPT,
    Cp04FeatureExtractor,
    _shape_evidence_consistent,
)
from medical_evaluation.segmentation.base import FrameMasks
from medical_evaluation.segmentation.sam3_backend import Sam3AmbiguousTextResult


def test_hand_prompt_targets_the_open_display_palm() -> None:
    assert HAND_PROMPT == "open white gloved palm holding a small shiny metal clip"
    assert LOCAL_CLAMP_PROMPT == "metal clip"


def test_consistency_tolerates_one_noisy_candidate_after_two_matches() -> None:
    assert _shape_evidence_consistent([0.95, 0.91, 0.32], 0.8) is True
    assert _shape_evidence_consistent([0.95, 0.32, 0.30], 0.8) is False
    assert _shape_evidence_consistent([0.42, 0.32, 0.30], 0.8) is True


def _clamp_mask() -> np.ndarray:
    mask = np.zeros((220, 220), np.uint8)
    cv2.ellipse(mask, (110, 118), (48, 62), 0, 205, 335, 1, 16)
    cv2.rectangle(mask, (49, 76), (88, 103), 1, -1)
    cv2.rectangle(mask, (132, 76), (171, 103), 1, -1)
    cv2.rectangle(mask, (67, 95), (84, 151), 1, -1)
    cv2.rectangle(mask, (136, 95), (153, 151), 1, -1)
    matrix = cv2.getRotationMatrix2D((110, 110), 0, 0.55)
    return cv2.warpAffine(
        mask, matrix, (220, 220), flags=cv2.INTER_NEAREST
    ).astype(bool)


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


class LocalClipSegmenter(FakeSegmenter):
    def __init__(self, hand_outputs: list[FrameMasks]) -> None:
        super().__init__([hand_outputs])
        self.local_frames: list[np.ndarray] = []

    def track(self, video_path, time_range, prompts, sample_fps):
        self.calls.append((video_path, time_range, prompts, sample_fps))
        if len(self.calls) == 1:
            return iter(self.outputs[0])
        capture = cv2.VideoCapture(str(video_path))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            self.local_frames.append(frame)
        capture.release()
        outputs = []
        for index, frame in enumerate(self.local_frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = gray < 105
            outputs.append(
                FrameMasks(
                    frame_index=index,
                    frame_time_sec=index / 2,
                    sample_position=index,
                    masks={LOCAL_CLAMP_OBJECT_ID: mask},
                )
            )
        return iter(outputs)


def _reference_dir(tmp_path: Path) -> Path:
    root = tmp_path / "reference"
    (root / "masks").mkdir(parents=True)
    assert cv2.imwrite(str(root / "masks" / "00000000.png"), _clamp_mask().astype(np.uint8) * 255)
    return root


def _frame(
    *, hand: np.ndarray | None = None, clamp: np.ndarray | None = None
) -> np.ndarray:
    frame = np.full((220, 220, 3), (180, 110, 45), np.uint8)
    if hand is not None:
        frame[hand] = (220, 225, 230)
    if clamp is not None:
        frame[clamp] = (82, 86, 90)
    return frame


def test_extracts_matching_clamp_from_one_hand_and_one_local_sam3_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hand = _hand_mask()
    clamp = _clamp_mask()
    segmenter = LocalClipSegmenter(_tracked("cp04_gloved_hand", [hand] * 3))
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame(hand=hand, clamp=clamp),
    )

    result = Cp04FeatureExtractor(
        segmenter=segmenter,
        evidence_root=tmp_path / "evidence",
        reference_dir=_reference_dir(tmp_path),
        min_similarity=0.7,
    ).extract(
        tmp_path / "video.mp4",
        "cp_04",
        TimeRange(start_sec=41, end_sec=47),
        dense_fps=5,
        analysis_width=1280,
    )

    assert [call[2][0].text for call in segmenter.calls] == [
        HAND_PROMPT,
        LOCAL_CLAMP_PROMPT,
    ]
    assert segmenter.calls[1][2][0].object_id == LOCAL_CLAMP_OBJECT_ID
    assert segmenter.calls[1][2][0].kind == "text"
    assert segmenter.calls[1][2][0].frame_time_sec == 0.0
    assert segmenter.calls[1][0] == tmp_path / "evidence" / "cp_04" / "local_input" / "display_clip.mp4"
    assert all(call[3] == 2.0 for call in segmenter.calls)
    assert 3 <= len(segmenter.local_frames) <= 5
    assert all(frame.shape[:2] == (640, 640) for frame in segmenter.local_frames)
    assert result.features == {
        "clamp_observed": True,
        "shape_evidence_reliable": True,
        "clear_frame_count": 3.0,
        "matching_frame_count": 3.0,
        "clamp_reference_similarity": pytest.approx(0.76, abs=0.03),
        "evidence_consistent": True,
    }
    assert len(result.evidence) == 3
    for item in result.evidence:
        assert (tmp_path / "evidence" / item.overlay_path).is_file()
    assert (tmp_path / "evidence" / "cp_04" / "frame_analysis.json").is_file()


def test_empty_open_glove_does_not_start_another_sam3_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hand = _hand_mask()
    segmenter = FakeSegmenter([_tracked("cp04_gloved_hand", [hand] * 2)])
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame(hand=hand),
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

    assert [call[2][0].text for call in segmenter.calls] == [HAND_PROMPT]
    assert result.features["clamp_observed"] is False
    assert result.features["clear_frame_count"] == 0.0


def test_local_sam3_masks_are_mapped_back_to_original_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hand = _hand_mask()
    full_clamp = _clamp_mask()
    segmenter = LocalClipSegmenter(_tracked("cp04_gloved_hand", [hand] * 2))
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame(hand=hand, clamp=full_clamp),
    )

    result = Cp04FeatureExtractor(
        segmenter=segmenter,
        evidence_root=tmp_path / "evidence",
        reference_dir=_reference_dir(tmp_path),
        min_similarity=0.7,
    ).extract(
        tmp_path / "video.mp4",
        "cp_04",
        TimeRange(start_sec=41, end_sec=47),
        dense_fps=2,
        analysis_width=1280,
    )

    assert len(segmenter.calls) == 2
    prompt = segmenter.calls[1][2][0]
    assert prompt.kind == "text"
    assert prompt.object_id == LOCAL_CLAMP_OBJECT_ID
    assert prompt.text == LOCAL_CLAMP_PROMPT
    assert result.features["matching_frame_count"] == 2.0
    saved = cv2.imread(
        str(tmp_path / "evidence" / "cp_04" / "masks" / "clamp" / "00000000.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    assert saved is not None
    assert saved.shape == full_clamp.shape
    assert np.logical_and(saved > 0, full_clamp).sum() > 0.8 * full_clamp.sum()


def test_one_local_sam3_mask_uses_isolated_crop_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hand = _hand_mask()
    clamp = _clamp_mask()

    class OneMaskSegmenter(LocalClipSegmenter):
        def track(self, video_path, time_range, prompts, sample_fps):
            outputs = list(super().track(video_path, time_range, prompts, sample_fps))
            return iter(outputs[:1] if len(self.calls) == 2 else outputs)

    segmenter = OneMaskSegmenter(_tracked("cp04_gloved_hand", [hand] * 3))
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame(hand=hand, clamp=clamp),
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
    assert result.features["shape_evidence_reliable"] is True
    assert result.features["clear_frame_count"] >= 2.0
    analysis = (tmp_path / "evidence" / "cp_04" / "frame_analysis.json").read_text(
        "utf-8"
    )
    assert "opencv:isolated-hand-crop" in analysis


def test_empty_local_sam3_uses_only_isolated_crop_opencv_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hand = _hand_mask()
    clamp = _clamp_mask()

    class EmptyLocalSegmenter(LocalClipSegmenter):
        def track(self, video_path, time_range, prompts, sample_fps):
            outputs = list(super().track(video_path, time_range, prompts, sample_fps))
            return iter([] if len(self.calls) == 2 else outputs)

    segmenter = EmptyLocalSegmenter(_tracked("cp04_gloved_hand", [hand] * 3))
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame(hand=hand, clamp=clamp),
    )

    result = Cp04FeatureExtractor(
        segmenter=segmenter,
        evidence_root=tmp_path / "evidence",
        reference_dir=_reference_dir(tmp_path),
        min_similarity=0.7,
    ).extract(
        tmp_path / "video.mp4",
        "cp_04",
        TimeRange(start_sec=41, end_sec=47),
        dense_fps=2,
        analysis_width=1280,
    )

    assert len(segmenter.calls) == 2
    assert result.features["clamp_observed"] is True
    assert result.features["shape_evidence_reliable"] is True
    analysis = (tmp_path / "evidence" / "cp_04" / "frame_analysis.json").read_text(
        "utf-8"
    )
    assert "opencv:isolated-hand-crop" in analysis


def test_empty_local_clip_uses_lossless_roi_image_segmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hand = _hand_mask()
    clamp = _clamp_mask()

    class LosslessRoiSegmenter(LocalClipSegmenter):
        def __init__(self):
            super().__init__(_tracked("cp04_gloved_hand", [hand] * 3))
            self.image_prompts: list[str] = []

        def track(self, video_path, time_range, prompts, sample_fps):
            outputs = list(super().track(video_path, time_range, prompts, sample_fps))
            return iter([] if len(self.calls) == 2 else outputs)

        def segment_image(self, image_bgr, prompt):
            self.image_prompts.append(prompt.text)
            mask = np.zeros(image_bgr.shape[:2], bool)
            if prompt.text == LOCAL_ROI_LOCATOR_PROMPT:
                mask[250:330, 270:370] = True
            else:
                resized = cv2.resize(
                    _clamp_mask().astype(np.uint8),
                    (360, 360),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                mask[140:500, 140:500] = resized
            return FrameMasks(
                frame_index=0,
                frame_time_sec=0.0,
                sample_position=0,
                masks={LOCAL_CLAMP_OBJECT_ID: mask},
            )

    segmenter = LosslessRoiSegmenter()
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame(hand=hand, clamp=clamp),
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

    assert result.features["shape_evidence_reliable"] is True
    assert segmenter.image_prompts[0] == LOCAL_ROI_LOCATOR_PROMPT
    assert segmenter.image_prompts.count(LOCAL_ROI_CLAMP_PROMPT) >= 2
    analysis = (tmp_path / "evidence" / "cp_04" / "frame_analysis.json").read_text("utf-8")
    assert "sam3:lossless-clamp-roi" in analysis


def test_one_lossless_roi_mask_is_reliable_without_opencv_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hand = _hand_mask()
    clamp = _clamp_mask()

    class OneLosslessRoiSegmenter(LocalClipSegmenter):
        def __init__(self):
            super().__init__(_tracked("cp04_gloved_hand", [hand] * 2))
            self.image_call = 0

        def track(self, video_path, time_range, prompts, sample_fps):
            outputs = list(super().track(video_path, time_range, prompts, sample_fps))
            return iter([] if len(self.calls) == 2 else outputs)

        def segment_image(self, image_bgr, prompt):
            self.image_call += 1
            mask = np.zeros(image_bgr.shape[:2], bool)
            if prompt.text == LOCAL_ROI_LOCATOR_PROMPT:
                mask[250:330, 270:370] = True
            elif self.image_call == 2:
                resized = cv2.resize(_clamp_mask().astype(np.uint8), (360, 360), interpolation=cv2.INTER_NEAREST)
                mask[140:500, 140:500] = resized.astype(bool)
            else:
                return None
            return FrameMasks(frame_index=0, frame_time_sec=0.0, sample_position=0, masks={LOCAL_CLAMP_OBJECT_ID: mask})

    segmenter = OneLosslessRoiSegmenter()
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame(hand=hand, clamp=clamp),
    )
    result = Cp04FeatureExtractor(
        segmenter=segmenter,
        evidence_root=tmp_path / "evidence",
        reference_dir=_reference_dir(tmp_path),
    ).extract(tmp_path / "video.mp4", "cp_04", TimeRange(start_sec=41, end_sec=47), dense_fps=2, analysis_width=1280)

    assert result.features["shape_evidence_reliable"] is True
    assert result.features["clear_frame_count"] == 1.0
    assert result.features["evidence_consistent"] is True


def test_clamp_on_rack_outside_glove_is_not_clear_shape_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    segmenter = FakeSegmenter(
        [_tracked("cp04_gloved_hand", [_hand_mask(left=True)] * 2)]
    )
    outside_clamp = cv2.warpAffine(
        _clamp_mask().astype(np.uint8),
        np.float32([[1, 0, 60], [0, 1, 0]]),
        (220, 220),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame(
            hand=_hand_mask(left=True), clamp=outside_clamp
        ),
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
    assert result.features["clear_frame_count"] == 0.0


def test_ambiguous_hand_text_result_returns_missing_observation_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    segmenter = FakeSegmenter([Sam3AmbiguousTextResult("ambiguous")])
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame(),
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
