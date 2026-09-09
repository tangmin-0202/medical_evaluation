from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError


def test_reviewed_rubric_has_11_equal_weight_checkpoints() -> None:
    from medical_evaluation.rubric import Rubric, load_rubric

    rubric = load_rubric(Path("config/rubric.yaml"))

    assert isinstance(rubric, Rubric)
    assert [cp.id for cp in rubric.checkpoints] == [f"cp_{n:02d}" for n in range(1, 12)]
    assert abs(sum(cp.weight for cp in rubric.checkpoints) - 1.0) < 1e-9
    assert len({round(cp.weight, 12) for cp in rubric.checkpoints}) == 1


def test_each_checkpoint_defines_a_judge_and_criteria() -> None:
    from medical_evaluation.rubric import load_rubric

    rubric = load_rubric(Path("config/rubric.yaml"))

    assert all(cp.judge_type and cp.criteria for cp in rubric.checkpoints)


def test_cp09_requires_frame_and_fixed_oral_reference() -> None:
    from medical_evaluation.rubric import load_rubric

    rubric = load_rubric(Path("config/rubric.yaml"))
    cp09 = next(cp for cp in rubric.checkpoints if cp.id == "cp_09")

    assert cp09.required_objects == ["rubber_dam_frame", "oral_region"]


def test_cp01_requires_pen_presence_mark_tracking_thresholds() -> None:
    from medical_evaluation.rubric import load_rubric

    rubric = load_rubric(Path("config/rubric.yaml"))
    cp01 = next(cp for cp in rubric.checkpoints if cp.id == "cp_01")

    assert cp01.required_objects == ["rubber_dam", "marking_pen"]
    assert cp01.thresholds == {
        "max_mark_distance": 0.05,
        "min_pen_presence_frames": 2.0,
        "min_mark_observed_frames": 3.0,
        "min_mark_area_ratio": 0.0001,
        "max_mark_area_ratio": 0.01,
        "maximum_black_value": 55.0,
        "maximum_faint_value": 160.0,
        "maximum_faint_saturation": 120.0,
        "max_mark_aspect_ratio": 2.0,
        "min_mark_circularity": 0.35,
        "max_local_cluster_distance": 0.04,
        "local_darkness_ring_radius": 5.0,
    }


def test_cp11_uses_final_coverage_objects_and_thresholds() -> None:
    from medical_evaluation.rubric import load_rubric

    rubric = load_rubric(Path("config/rubric.yaml"))
    cp11 = next(cp for cp in rubric.checkpoints if cp.id == "cp_11")

    assert cp11.required_objects == ["rubber_dam", "nose_region"]
    assert cp11.thresholds == {
        "min_stage_dam_presence_ratio": 0.05,
        "min_final_dam_presence_ratio": 0.5,
        "min_dam_area_ratio": 0.20,
        "max_nose_overlap": 0.02,
        "max_visible_frame_area_ratio": 0.005,
        "min_expected_frame_dam_coverage_ratio": 0.70,
        "max_visible_frame_ratio": 0.20,
    }


def test_cp10_defines_temporal_contact_threshold() -> None:
    from medical_evaluation.rubric import load_rubric

    rubric = load_rubric(Path("config/rubric.yaml"))
    cp10 = next(cp for cp in rubric.checkpoints if cp.id == "cp_10")

    assert cp10.required_objects == ["target_tooth", "dental_floss"]
    assert cp10.thresholds == {"min_contact_frames_per_side": 2.0}


def test_parse_time_range_converts_excel_text_to_seconds() -> None:
    from medical_evaluation.rubric import parse_time_range

    result = parse_time_range("0:00:23-0:00:40")

    assert result.start_sec == 23.0
    assert result.end_sec == 40.0


def test_split_criteria_extracts_numbered_items() -> None:
    from medical_evaluation.rubric import split_criteria

    assert split_criteria("1、检查残留；\n2、选择第二个孔洞。") == [
        "检查残留",
        "选择第二个孔洞。",
    ]


def test_split_criteria_accepts_chinese_comma_between_numbered_items() -> None:
    from medical_evaluation.rubric import split_criteria

    assert split_criteria("1、橡皮布不遮盖患者口鼻，2、橡皮布撑开至支架上") == [
        "橡皮布不遮盖患者口鼻",
        "橡皮布撑开至支架上",
    ]


def test_rubric_rejects_overlapping_reference_ranges() -> None:
    from medical_evaluation.rubric import Rubric

    payload = yaml.safe_load(Path("config/rubric.yaml").read_text(encoding="utf-8"))
    payload["checkpoints"][3]["reference_time"]["start_sec"] = 23.0

    with pytest.raises(ValidationError, match="reference times must not overlap"):
        Rubric.model_validate(payload)
