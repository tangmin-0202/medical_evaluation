from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp04 import (
    HAND_PROMPT,
    LOCAL_CLAMP_OBJECT_ID,
    LOCAL_CLAMP_PROMPT,
    LOCAL_CLAMP_SOURCE,
    Cp04FeatureExtractor,
)
from medical_evaluation.segmentation.base import FrameMasks
from scripts.build_cp04_reference import build_parser


def _clamp_mask() -> np.ndarray:
    mask = np.zeros((180, 180), np.uint8)
    cv2.ellipse(mask, (90, 98), (38, 48), 0, 205, 335, 1, 14)
    cv2.rectangle(mask, (42, 60), (72, 84), 1, -1)
    cv2.rectangle(mask, (108, 60), (138, 84), 1, -1)
    return mask.astype(bool)


def _hand_mask() -> np.ndarray:
    mask = np.zeros((180, 180), np.uint8)
    cv2.circle(mask, (90, 90), 82, 1, -1)
    return mask.astype(bool)


def _tracked(object_id: str, masks: list[np.ndarray]) -> list[FrameMasks]:
    return [
        FrameMasks(
            frame_index=index + 10,
            frame_time_sec=42.0 + index * 0.5,
            masks={object_id: mask},
        )
        for index, mask in enumerate(masks)
    ]


class FakeSegmenter:
    model_version = "fake-sam3-reference"

    def __init__(self) -> None:
        self.hand_outputs = _tracked("cp04_gloved_hand", [_hand_mask()] * 3)
        self.calls = []

    def track(self, video_path, time_range, prompts, sample_fps):
        assert sample_fps == 2.0
        self.calls.append((video_path, time_range, prompts))
        if len(self.calls) == 1:
            return iter(self.hand_outputs)
        capture = cv2.VideoCapture(str(video_path))
        outputs = []
        position = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) < 105
            outputs.append(
                FrameMasks(
                    frame_index=position,
                    frame_time_sec=position / 2,
                    sample_position=position,
                    masks={LOCAL_CLAMP_OBJECT_ID: mask},
                )
            )
            position += 1
        capture.release()
        return iter(outputs)


def _frame() -> np.ndarray:
    frame = np.full((180, 180, 3), (180, 110, 45), np.uint8)
    frame[_hand_mask()] = (220, 225, 230)
    frame[_clamp_mask()] = (82, 86, 90)
    return frame


def test_builder_cli_is_fixed_to_success_reference() -> None:
    args = build_parser().parse_args(["--replace"])

    assert args.video_id == "success"
    assert args.replace is True


def test_builder_can_be_executed_directly() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_cp04_reference.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "CP04 clamp reference" in completed.stdout


def test_build_reference_writes_versioned_masks_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame().copy(),
    )
    reference_dir = tmp_path / "reference"
    segmenter = FakeSegmenter()
    extractor = Cp04FeatureExtractor(
        segmenter=segmenter,
        evidence_root=tmp_path / "evidence",
        reference_dir=reference_dir,
    )

    manifest_path = extractor.build_reference(
        tmp_path / "success.mp4",
        TimeRange(start_sec=41, end_sec=47),
        video_id="success",
        replace=False,
    )

    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["reference_version"] == "cp04-reference-v1"
    assert manifest["video_id"] == "success"
    assert manifest["video_filename"] == "success.mp4"
    assert "video_path" not in manifest
    assert manifest["model_version"] == extractor.model_version
    assert manifest["candidate_source"] == LOCAL_CLAMP_SOURCE
    assert len(manifest["frames"]) == 3
    assert [row["frame_index"] for row in manifest["frames"]] == [10, 11, 12]
    assert [row["crop_clip_position"] for row in manifest["frames"]] == [0, 1, 2]
    assert all((reference_dir / row["mask_path"]).is_file() for row in manifest["frames"])
    assert [call[2][0].text for call in segmenter.calls] == [
        HAND_PROMPT,
        LOCAL_CLAMP_PROMPT,
    ]


def test_build_reference_requires_two_complete_local_sam3_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OneMaskSegmenter(FakeSegmenter):
        def track(self, video_path, time_range, prompts, sample_fps):
            outputs = list(super().track(video_path, time_range, prompts, sample_fps))
            return iter(outputs[:1] if len(self.calls) == 2 else outputs)

    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame().copy(),
    )
    extractor = Cp04FeatureExtractor(
        segmenter=OneMaskSegmenter(),
        evidence_root=tmp_path / "evidence",
        reference_dir=tmp_path / "reference",
    )

    with pytest.raises(RuntimeError, match="fewer than two reliable"):
        extractor.build_reference(
            tmp_path / "success.mp4",
            TimeRange(start_sec=41, end_sec=47),
            video_id="success",
            replace=False,
        )


def test_build_reference_refuses_silent_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "medical_evaluation.extractors.cp04.read_frame",
        lambda _path, _index: _frame().copy(),
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / "manifest.json").write_text("{}", encoding="utf-8")
    extractor = Cp04FeatureExtractor(
        segmenter=FakeSegmenter(),
        evidence_root=tmp_path / "evidence",
        reference_dir=reference_dir,
    )

    with pytest.raises(FileExistsError, match="--replace"):
        extractor.build_reference(
            tmp_path / "success.mp4",
            TimeRange(start_sec=41, end_sec=47),
            video_id="success",
            replace=False,
        )
