from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from medical_evaluation.segmentation.base import FrameMasks


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "run_cp08_sam3_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("run_cp08_sam3_gate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_annotation(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "video_id": "success",
                "steps": [
                    {
                        "checkpoint_id": "cp_08",
                        "time_range": {"start_sec": 134.0, "end_sec": 174.0},
                        "label": "correct",
                        "reason": "fixture",
                    },
                    {
                        "checkpoint_id": "cp_09",
                        "time_range": {"start_sec": 175.0, "end_sec": 195.0},
                        "label": "correct",
                        "reason": "fixture",
                    },
                ],
                "prompts": [
                    {
                        "kind": "point",
                        "video_id": "success",
                        "frame_time_sec": 139.814028,
                        "object_id": "blunt-tipped instrument",
                        "x": 0.53,
                        "y": 0.16,
                        "positive": True,
                    }
                ],
                "audit_history": [],
            }
        ),
        encoding="utf-8",
    )


def test_prompt_catalog_has_independent_cp08_and_cp09_objects() -> None:
    gate = _load_gate_module()

    assert set(gate.PROMPT_CANDIDATES) == {
        "blunt_instrument",
        "sharp_probe",
        "target_tooth",
        "rubber_dam_clamp",
        "rubber_dam",
    }
    assert len(gate.PROMPT_CANDIDATES["blunt_instrument"]) >= 2
    assert len(gate.PROMPT_CANDIDATES["sharp_probe"]) >= 2
    assert "green dental rubber dam" in gate.PROMPT_CANDIDATES["rubber_dam"]


def test_load_windows_uses_all_of_cp08_and_last_three_seconds_of_cp09(
    tmp_path: Path,
) -> None:
    gate = _load_gate_module()
    annotation = tmp_path / "success.json"
    _write_annotation(annotation)

    windows = gate.load_gate_windows(annotation)

    assert windows.cp08 == pytest.approx((134.0, 174.0))
    assert windows.cp09_tail == pytest.approx((192.0, 195.0))


def test_summarize_masks_requires_three_consecutive_valid_frames() -> None:
    gate = _load_gate_module()
    masks = [
        np.zeros((8, 8), dtype=bool),
        np.pad(np.ones((3, 3), dtype=bool), ((2, 3), (2, 3))),
        np.pad(np.ones((3, 3), dtype=bool), ((2, 3), (2, 3))),
        np.pad(np.ones((3, 3), dtype=bool), ((2, 3), (2, 3))),
    ]

    result = gate.summarize_masks(masks, min_area_px=4)

    assert result["valid_frame_count"] == 3
    assert result["max_consecutive_valid_frames"] == 3
    assert result["median_dominant_component_ratio"] == pytest.approx(1.0)
    assert result["automatic_gate_passed"] is True


def test_summarize_masks_rejects_fragmented_mask() -> None:
    gate = _load_gate_module()
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[7:9, 7:9] = True

    result = gate.summarize_masks([mask, mask, mask], min_area_px=2)

    assert result["median_dominant_component_ratio"] == pytest.approx(0.5)
    assert result["automatic_gate_passed"] is False


def test_select_prompt_prefers_pass_then_continuity_then_coherence() -> None:
    gate = _load_gate_module()
    candidates = [
        {
            "prompt": "a",
            "automatic_gate_passed": False,
            "max_consecutive_valid_frames": 9,
            "median_dominant_component_ratio": 0.9,
        },
        {
            "prompt": "b",
            "automatic_gate_passed": True,
            "max_consecutive_valid_frames": 3,
            "median_dominant_component_ratio": 0.7,
        },
        {
            "prompt": "c",
            "automatic_gate_passed": True,
            "max_consecutive_valid_frames": 4,
            "median_dominant_component_ratio": 0.6,
        },
    ]

    assert gate.select_prompt(candidates)["prompt"] == "c"


class FakeSegmenter:
    model_version = "fake-sam3"

    def __init__(self, gate) -> None:
        self.gate = gate
        self.calls: list[dict[str, object]] = []

    def track(self, video_path, time_range, prompts, sample_fps):
        prompt = prompts[0]
        self.calls.append(
            {
                "video_path": video_path,
                "time_range": time_range,
                "prompt": prompt,
                "sample_fps": sample_fps,
            }
        )
        if prompt.object_id == "blunt_instrument":
            if prompt.text != self.gate.PROMPT_CANDIDATES["blunt_instrument"][0]:
                return iter(())
            start = 140.0 if sample_fps == self.gate.CP08_SPARSE_FPS else time_range.start_sec
        elif prompt.object_id == "sharp_probe":
            return iter(())
        else:
            start = time_range.start_sec
        mask = np.zeros((24, 24), dtype=bool)
        mask[7:17, 7:17] = True
        return iter(
            FrameMasks(
                frame_index=1000 + index,
                frame_time_sec=start + index / sample_fps,
                masks={prompt.object_id: mask.copy()},
                sample_position=index,
            )
            for index in range(4)
        )


def _fake_sample_frames(_path, *, start_sec, end_sec, sample_fps):
    del end_sec, sample_fps
    yield SimpleNamespace(
        frame_index=999,
        time_sec=start_sec,
        image_bgr=np.zeros((24, 24, 3), dtype=np.uint8),
    )


def test_run_gate_uses_text_only_and_independent_windows(tmp_path: Path) -> None:
    gate = _load_gate_module()
    annotation = tmp_path / "success.json"
    _write_annotation(annotation)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    output = tmp_path / "gate"
    segmenter = FakeSegmenter(gate)

    summary = gate.run_gate(
        segmenter=segmenter,
        video_path=video,
        annotation_path=annotation,
        output_dir=output,
        read_frame_fn=lambda _path, _index: np.zeros((24, 24, 3), dtype=np.uint8),
        sample_frames_fn=_fake_sample_frames,
    )

    sparse_calls = [
        call
        for call in segmenter.calls
        if call["prompt"].object_id in {"blunt_instrument", "sharp_probe"}
        and call["sample_fps"] == gate.CP08_SPARSE_FPS
    ]
    dense_calls = [
        call
        for call in segmenter.calls
        if call["prompt"].object_id in {"blunt_instrument", "sharp_probe"}
        and call["sample_fps"] == gate.CP08_DENSE_FPS
    ]
    tail_calls = [
        call
        for call in segmenter.calls
        if call["prompt"].object_id
        in {"target_tooth", "rubber_dam_clamp", "rubber_dam"}
    ]

    assert len(sparse_calls) == sum(
        len(gate.PROMPT_CANDIDATES[name])
        for name in ("blunt_instrument", "sharp_probe")
    )
    assert all(
        (call["time_range"].start_sec, call["time_range"].end_sec)
        == pytest.approx((134.0, 174.0))
        for call in sparse_calls
    )
    assert len(dense_calls) == 1
    assert (
        dense_calls[0]["time_range"].start_sec,
        dense_calls[0]["time_range"].end_sec,
    ) == pytest.approx((138.5, 141.5))
    assert all(call["sample_fps"] == gate.CP09_TAIL_FPS for call in tail_calls)
    assert all(
        (call["time_range"].start_sec, call["time_range"].end_sec)
        == pytest.approx((192.0, 195.0))
        for call in tail_calls
    )
    assert all(call["prompt"].kind == "text" for call in segmenter.calls)
    assert all(call["prompt"].frame_time_sec != pytest.approx(139.814028) for call in segmenter.calls)
    assert summary["automatic_gate_passed"] is True


def test_run_gate_writes_auditable_relative_artifact_paths(tmp_path: Path) -> None:
    gate = _load_gate_module()
    annotation = tmp_path / "success.json"
    _write_annotation(annotation)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    output = tmp_path / "gate"

    gate.run_gate(
        segmenter=FakeSegmenter(gate),
        video_path=video,
        annotation_path=annotation,
        output_dir=output,
        read_frame_fn=lambda _path, _index: np.zeros((24, 24, 3), dtype=np.uint8),
        sample_frames_fn=_fake_sample_frames,
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    selected = json.loads((output / "selected_prompts.json").read_text(encoding="utf-8"))
    review = json.loads((output / "manual_review.json").read_text(encoding="utf-8"))
    artifact_paths = [
        row["overlay_path"]
        for prompt_runs in summary["objects"].values()
        for run in prompt_runs
        for phase in ("sparse", "dense", "tail")
        for row in run.get(phase, {}).get("frames", [])
    ]

    assert selected["session_strategy"] == "independent"
    assert {item["object"] for item in review["items"]} == set(gate.PROMPT_CANDIDATES)
    assert artifact_paths
    assert all(not Path(path).is_absolute() for path in artifact_paths)
    assert all((output / path).is_file() for path in artifact_paths)


def test_run_gate_refuses_non_empty_output_directory(tmp_path: Path) -> None:
    gate = _load_gate_module()
    output = tmp_path / "gate"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        gate.run_gate(
            segmenter=FakeSegmenter(gate),
            video_path=tmp_path / "video.mp4",
            annotation_path=tmp_path / "success.json",
            output_dir=output,
            read_frame_fn=lambda _path, _index: np.zeros((24, 24, 3), dtype=np.uint8),
            sample_frames_fn=_fake_sample_frames,
        )
