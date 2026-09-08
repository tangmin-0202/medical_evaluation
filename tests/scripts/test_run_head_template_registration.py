from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def _load_script():
    path = Path(__file__).parents[2] / "scripts" / "run_head_template_registration.py"
    spec = importlib.util.spec_from_file_location("run_head_template_registration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_template_requires_normalized_nose_polygon_and_provenance() -> None:
    module = _load_script()

    with pytest.raises(TypeError, match="nose_polygon_normalized"):
        module.validate_template(
            {
                "schema_version": 1,
                "template_id": "mannequin-v1",
                "head_prompt": "dental training mannequin head",
                "reference": {"video_id": "success", "stage": "cp_11", "frame_index": 9},
                "demo_only": True,
            }
        )


def test_consecutive_gate_requires_adjacent_positions_and_continuity() -> None:
    module = _load_script()
    rows = [
        {"sample_position": 0, "accepted": True, "continuous": True},
        {"sample_position": 1, "accepted": True, "continuous": True},
        {"sample_position": 2, "accepted": True, "continuous": True},
    ]
    assert module.has_consecutive_acceptance(rows, minimum=3)
    rows[1]["continuous"] = False
    assert not module.has_consecutive_acceptance(rows, minimum=3)


def test_tail_sampling_caps_each_stage_without_breaking_adjacency() -> None:
    module = _load_script()
    rows = [
        {"sample_key": "success:cp_09", "sample_position": position}
        for position in range(10)
    ] + [
        {"sample_key": "success:cp_11", "sample_position": position}
        for position in range(6)
    ]

    selected = module.select_tail_rows(rows, max_frames_per_sample=4)

    assert [row["sample_position"] for row in selected if row["sample_key"] == "success:cp_09"] == [6, 7, 8, 9]
    assert [row["sample_position"] for row in selected if row["sample_key"] == "success:cp_11"] == [2, 3, 4, 5]


def test_each_stage_uses_its_own_tail_anchor_except_the_canonical_stage() -> None:
    module = _load_script()
    rows = [
        {"sample_key": "success:cp_09", "sample_position": 4, "frame_index": 40},
        {"sample_key": "success:cp_09", "sample_position": 5, "frame_index": 50},
        {"sample_key": "success:cp_11", "sample_position": 8, "frame_index": 80},
        {"sample_key": "success:cp_11", "sample_position": 9, "frame_index": 90},
    ]
    reference = {"video_id": "success", "stage": "cp_11", "frame_index": 80}

    anchors = module.select_sample_anchors(rows, reference)

    assert anchors["success:cp_09"]["frame_index"] == 50
    assert anchors["success:cp_11"]["frame_index"] == 80


def test_overall_gate_requires_template_compatibility_only_for_declared_samples() -> None:
    module = _load_script()
    samples = {
        "success:cp_09": {"accepted": True, "template_compatible": True},
        "success:cp_11": {"accepted": True, "template_compatible": True},
        "failure:cp_09": {"accepted": True, "template_compatible": False},
    }

    assert module.gate_accepts(samples, ["success:cp_09", "success:cp_11"])
    assert not module.gate_accepts(samples, ["failure:cp_09"])


def test_gate_reuses_saved_raw_masks_and_writes_auditable_overlays(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    trial = source / "overlays" / "success" / "head" / "dental-training-mannequin-head"
    (trial / "raw").mkdir(parents=True)
    (trial / "masks").mkdir()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.ellipse(mask, (80, 60), (55, 42), 0, 0, 360, 255, -1)
    rng = np.random.default_rng(12)
    for _ in range(100):
        x, y = (int(value) for value in rng.integers((30, 25), (130, 95)))
        if mask[y, x]:
            cv2.circle(image, (x, y), 2, (180, 220, 90), -1)
    for index in (10, 11, 12):
        assert cv2.imwrite(str(trial / "raw" / f"{index:08d}.png"), image)
        assert cv2.imwrite(str(trial / "masks" / f"{index:08d}.png"), mask)
    summary = {
        "git_revision": "source-revision",
        "sam3_source_revision": "sam3-revision",
        "model_version": "sam3.1:test",
        "rows_by_candidate": {
            "dental training mannequin head": [
                {
                    "sample_key": "success:cp_11",
                    "video_id": "success",
                    "stage": "cp_11",
                    "frame_index": index,
                    "sample_position": position,
                    "source_time_sec": float(position),
                    "valid": True,
                }
                for position, index in enumerate((10, 11, 12))
            ]
        },
    }
    (source / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    template = {
        "schema_version": 1,
        "template_id": "mannequin-v1",
        "head_prompt": "dental training mannequin head",
        "reference": {"video_id": "success", "stage": "cp_11", "frame_index": 10},
        "nose_polygon_normalized": [[0.45, 0.72], [0.55, 0.72], [0.55, 0.90], [0.45, 0.90]],
        "demo_only": True,
    }
    output = tmp_path / "result"

    exit_code = module.run_gate(source, output, template, minimum_consecutive=3)

    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["accepted"] is True
    assert result["source_git_revision"] == "source-revision"
    assert result["template"]["template_id"] == "mannequin-v1"
    assert result["samples"]["success:cp_11"]["accepted"] is True
    assert len(result["rows"]) == 3
    assert all(Path(row["overlay_path"]).is_file() for row in result["rows"])
    assert all(len(row["matrix"]) == 2 for row in result["rows"])


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    assert module.run_gate(tmp_path, output, {}) == 2
    assert marker.read_text(encoding="utf-8") == "keep"
