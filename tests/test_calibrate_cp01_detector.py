from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.calibrate_cp01_detector import sweep_cp01_detector


def test_sweep_reports_pen_presence_and_stable_marks_with_overlay(
    tmp_path: Path,
) -> None:
    samples = []
    for frame_index in range(1, 7):
        frame = np.full((120, 120, 3), (30, 170, 90), dtype=np.uint8)
        cv2.circle(frame, (20, 20), 2, (20, 20, 20), -1)
        if frame_index >= 4:
            cv2.circle(frame, (78, 82), 2, (20, 20, 20), -1)
            cv2.circle(frame, (105, 12), 1, (10, 10, 10), -1)
        dam = np.ones((120, 120), dtype=bool)
        pen = np.zeros((120, 120), dtype=bool)
        if frame_index >= 4:
            pen[60:90, 60:90] = True
        samples.append((frame, dam, pen, frame_index, frame_index / 2))

    results = sweep_cp01_detector(
        samples,
        maximum_black_values=[55],
        maximum_faint_values=[160],
        maximum_faint_saturations=[120],
        minimum_area_ratios=[0.00005],
        minimum_observed_frames_values=[3],
        minimum_pen_presence_frames_values=[2],
        reference_u=0.65,
        reference_v=0.65,
        output_dir=tmp_path,
    )

    assert len(results) == 1
    assert results[0]["mark_candidate_count"] == 3
    assert results[0]["pen_presence_detected"] is True
    assert Path(results[0]["overlay_path"]).is_file()
