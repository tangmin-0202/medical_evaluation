from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp10 import Cp10FeatureExtractor


def _video(path: Path, *, include_floss: bool) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (400, 300))
    assert writer.isOpened()
    for index in range(40):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[:] = (30, 170, 90)
        cv2.rectangle(frame, (155, 105), (245, 205), (35, 35, 35), -1)
        cv2.rectangle(frame, (180, 130), (220, 180), (225, 225, 225), -1)
        if include_floss and 10 <= index < 20:
            cv2.line(frame, (90, 140), (310, 150), (235, 235, 235), 2)
        if include_floss and 20 <= index < 30:
            cv2.line(frame, (90, 170), (310, 180), (235, 235, 235), 2)
        writer.write(frame)
    writer.release()
    return path


def test_extracts_two_sided_floss_contact_and_writes_evidence(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    result = Cp10FeatureExtractor(evidence_root=root).extract(
        _video(tmp_path / "success.avi", include_floss=True),
        "cp_10",
        TimeRange(start_sec=0, end_sec=4),
        dense_fps=5,
        analysis_width=1280,
    )

    assert result.features["tooth_anchor_reliable"] is True
    assert result.features["floss_observed_frame_count"] >= 2
    assert result.features["upper_contact_frame_count"] >= 1
    assert result.features["lower_contact_frame_count"] >= 1
    assert len(result.evidence) >= 2
    assert all((root / item.overlay_path).is_file() for item in result.evidence)


def test_no_floss_is_reported_without_false_contact(tmp_path: Path) -> None:
    result = Cp10FeatureExtractor(evidence_root=tmp_path / "evidence").extract(
        _video(tmp_path / "missing.avi", include_floss=False),
        "cp_10",
        TimeRange(start_sec=0, end_sec=4),
        dense_fps=5,
        analysis_width=1280,
    )

    assert result.features["tooth_anchor_reliable"] is True
    assert result.features["floss_observed_frame_count"] == 0
    assert result.features["upper_contact_frame_count"] == 0
    assert result.features["lower_contact_frame_count"] == 0
