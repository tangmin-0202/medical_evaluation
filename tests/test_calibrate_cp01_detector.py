from __future__ import annotations

import importlib.util

import cv2
import numpy as np
import pytest


def test_sweep_keeps_only_thresholds_with_two_selectable_stable_marks() -> None:
    spec = importlib.util.find_spec("scripts.calibrate_cp01_detector")
    if spec is None:
        pytest.fail("CP01 detector calibration script is missing")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    samples = []
    for frame_index in (1, 2, 3):
        frame = np.full((120, 120, 3), (30, 170, 90), dtype=np.uint8)
        cv2.circle(frame, (105, 12), 1, (20, 20, 20), -1)
        cv2.circle(frame, (78, 82), 2, (145, 145, 145), -1)
        cv2.circle(frame, (35, 70), 3, (20, 80, 40), -1)
        samples.append((frame, np.ones((120, 120), dtype=bool), frame_index, frame_index / 2))

    results = module.sweep_cp01_detector(
        samples,
        maximum_black_values=[55],
        maximum_faint_values=[140, 160],
        maximum_faint_saturations=[80, 120],
        minimum_area_ratios=[0.0001],
        minimum_observed_frames=3,
    )

    assert len(results) == 2
    assert {item["maximum_faint_value"] for item in results} == {160}
    assert all(item["candidate_count"] == 2 for item in results)
    assert all(item["selection_status"] == "selected" for item in results)
