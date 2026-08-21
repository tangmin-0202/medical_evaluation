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
