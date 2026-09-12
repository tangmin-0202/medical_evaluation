from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


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
