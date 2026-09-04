import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.audit import AuditedVideoSegmenter
from medical_evaluation.segmentation.base import FrameMasks, SegmentationPrompt


class FakeSegmenter:
    model_version = "fake-sam3"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def track(
        self,
        _video_path: Path,
        _time_range: TimeRange,
        _prompts: list[SegmentationPrompt],
        _sample_fps: float,
    ) -> Iterator[FrameMasks]:
        if self.fail:
            raise RuntimeError("segmentation failed")
        yield FrameMasks(
            frame_index=10,
            frame_time_sec=10,
            masks={"rubber_dam_frame": np.ones((2, 2), dtype=bool)},
        )


def _prompt() -> SegmentationPrompt:
    return SegmentationPrompt(
        object_id="rubber_dam_frame",
        kind="text",
        frame_time_sec=10,
        text="white U-shaped dental frame",
    )


def test_audited_segmenter_records_completed_call(tmp_path: Path) -> None:
    output = tmp_path / "segmentation_metadata.json"
    wrapped = AuditedVideoSegmenter(
        FakeSegmenter(),
        output_path=output,
        static_metadata={
            "backend": "sam3",
            "source_revision": "660a5e9",
            "checkpoint_sha256": "0567debe",
        },
    )

    assert len(
        list(
            wrapped.track(
                Path("video.mp4"),
                TimeRange(start_sec=10, end_sec=20),
                [_prompt()],
                sample_fps=2,
            )
        )
    ) == 1

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backend"] == "sam3"
    assert payload["tracks"][0]["status"] == "completed"
    assert payload["tracks"][0]["prompt_texts"] == ["white U-shaped dental frame"]
    assert payload["tracks"][0]["time_range"] == {"start_sec": 10.0, "end_sec": 20.0}
    assert payload["tracks"][0]["frame_count"] == 1


def test_audited_segmenter_records_failure_without_swallowing_it(tmp_path: Path) -> None:
    output = tmp_path / "segmentation_metadata.json"
    wrapped = AuditedVideoSegmenter(
        FakeSegmenter(fail=True),
        output_path=output,
        static_metadata={"backend": "sam3"},
    )

    with pytest.raises(RuntimeError, match="segmentation failed"):
        list(
            wrapped.track(
                Path("video.mp4"),
                TimeRange(start_sec=10, end_sec=20),
                [_prompt()],
                sample_fps=2,
            )
        )

    track = json.loads(output.read_text(encoding="utf-8"))["tracks"][0]
    assert track["status"] == "failed"
    assert track["error_type"] == "RuntimeError"
    assert track["elapsed_seconds"] >= 0
