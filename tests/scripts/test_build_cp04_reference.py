from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp04 import Cp04FeatureExtractor
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
        self.outputs = [
            _tracked("cp04_gloved_hand", [_hand_mask()] * 3),
            _tracked("cp04_clamp", [_clamp_mask()] * 3),
        ]
        self.calls = 0

    def track(self, _video_path, _time_range, _prompts, sample_fps):
        assert sample_fps == 2.0
        output = self.outputs[self.calls]
        self.calls += 1
        return iter(output)


def _frame() -> np.ndarray:
    y, x = np.indices((180, 180))
    base = ((x * 5 + y * 13) % 255).astype(np.uint8)
    return np.dstack((base, base, base))


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
    extractor = Cp04FeatureExtractor(
        segmenter=FakeSegmenter(),
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
    assert len(manifest["frames"]) == 3
    assert [row["frame_index"] for row in manifest["frames"]] == [10, 11, 12]
    assert all((reference_dir / row["mask_path"]).is_file() for row in manifest["frames"])


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
