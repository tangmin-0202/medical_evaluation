from __future__ import annotations

import json
from pathlib import Path

from medical_evaluation.jobs import JobRecord
from medical_evaluation.pipeline import AnalysisPipeline, ExtractedEvidence
from medical_evaluation.rubric import load_rubric


class FakeExtractor:
    def __init__(self) -> None:
        self.fail_once_with_oom = False
        self.calls: list[tuple[str, float, int]] = []

    def extract(
        self,
        _video_path: Path,
        checkpoint_id: str,
        _time_range: object,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        self.calls.append((checkpoint_id, dense_fps, analysis_width))
        if self.fail_once_with_oom:
            self.fail_once_with_oom = False
            raise RuntimeError("CUDA out of memory")
        if checkpoint_id == "cp_09":
            return ExtractedEvidence(features={"frame_oral_center_offset": 0.03})
        return ExtractedEvidence(
            features={
                "mouth_nose_visible": True,
                "face_overlap": 0.01,
                "frame_coverage": 0.9,
            }
        )


class FakeLocalizer:
    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence

    def checkpoint_confidence(self, _checkpoint_id: str) -> float:
        return self.confidence


def make_pipeline(tmp_path: Path) -> tuple[AnalysisPipeline, FakeExtractor, FakeLocalizer]:
    rubric = load_rubric(Path(__file__).parents[1] / "config" / "rubric.yaml")
    extractor = FakeExtractor()
    localizer = FakeLocalizer()
    pipeline = AnalysisPipeline(
        rubric=rubric,
        extractor=extractor,
        localizer=localizer,
        output_root=tmp_path / "jobs",
        dense_fps=5,
        analysis_width=1280,
        fallback_dense_fps=1,
        fallback_analysis_width=960,
    )
    return pipeline, extractor, localizer


def make_job(tmp_path: Path) -> JobRecord:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    return JobRecord(id="job-1", video_id="success", video_path=str(video))


def test_pipeline_writes_two_real_decisions_and_nine_review_results(tmp_path: Path) -> None:
    pipeline, _, _ = make_pipeline(tmp_path)

    report = pipeline.run(make_job(tmp_path))

    assert len(report.checkpoints) == 11
    assert report.checkpoints[8].status.value == "correct"
    assert report.checkpoints[10].status.value == "correct"
    assert report.checkpoints[0].status.value == "needs_review"
    stored = json.loads((tmp_path / "jobs" / "job-1" / "report.json").read_text("utf-8"))
    assert len(stored["checkpoints"]) == 11


def test_pipeline_retries_oom_once_with_degraded_sampling(tmp_path: Path) -> None:
    pipeline, extractor, _ = make_pipeline(tmp_path)
    extractor.fail_once_with_oom = True

    report = pipeline.run(make_job(tmp_path))

    assert report.audit.degradations == ["cuda_oom:dense_fps=1.0,analysis_width=960"]
    assert extractor.calls[:2] == [("cp_01", 5, 1280), ("cp_01", 1, 960)]


def test_low_alignment_confidence_pauses_checkpoint(tmp_path: Path) -> None:
    pipeline, extractor, localizer = make_pipeline(tmp_path)
    localizer.confidence = 0.2

    report = pipeline.run(make_job(tmp_path))

    assert all(item.status.value == "needs_review" for item in report.checkpoints)
    assert report.checkpoints[0].reason_code == "low_step_alignment_confidence"
    assert extractor.calls == []
