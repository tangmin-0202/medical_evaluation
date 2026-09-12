from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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

