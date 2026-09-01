from medical_evaluation.judges.cp09_cp11 import judge_cp11

THRESHOLDS = {
    "min_stage_dam_presence_ratio": 0.05,
    "min_dam_area_ratio": 0.20,
    "max_nose_overlap": 0.02,
    "max_visible_frame_area_ratio": 0.005,
}


def _features(**updates: float | bool | None) -> dict[str, float | bool | None]:
    values: dict[str, float | bool | None] = {
        "dam_stage_presence_ratio": 0.5,
        "dam_area_ratio": 0.4,
        "nose_overlap": 0.0,
        "visible_frame_area_ratio": 0.0,
    }
    values.update(updates)
    return values


def test_unreliable_stage_presence_requires_review() -> None:
    result = judge_cp11(_features(dam_stage_presence_ratio=None), THRESHOLDS)

    assert result.status.value == "needs_review"


def test_no_dam_during_stage_is_incomplete() -> None:
    result = judge_cp11(_features(dam_stage_presence_ratio=0.0), THRESHOLDS)

    assert result.status.value == "incomplete"
    assert result.reason_code == "rubber_dam_not_observed"


def test_attempted_stage_with_good_final_state_is_correct() -> None:
    assert judge_cp11(_features(), THRESHOLDS).status.value == "correct"


def test_attempted_stage_with_small_final_dam_is_incorrect() -> None:
    result = judge_cp11(_features(dam_area_ratio=0.1), THRESHOLDS)

    assert result.status.value == "incorrect"
    assert "dam_area_sufficient" in result.matched_rules


def test_attempted_stage_with_covered_nose_is_incorrect() -> None:
    result = judge_cp11(_features(nose_overlap=0.1), THRESHOLDS)

    assert result.status.value == "incorrect"
    assert "nose_clear" in result.matched_rules


def test_attempted_stage_with_visible_frame_is_incorrect() -> None:
    result = judge_cp11(_features(visible_frame_area_ratio=0.02), THRESHOLDS)

    assert result.status.value == "incorrect"
    assert "frame_covered" in result.matched_rules
