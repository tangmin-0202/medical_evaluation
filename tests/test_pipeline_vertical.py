from __future__ import annotations

import json
from pathlib import Path

from medical_evaluation.jobs import JobRecord
from medical_evaluation.pipeline import (
    AnalysisPipeline,
    EvaluationInputMissing,
    ExtractedEvidence,
)
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.rubric import load_rubric
from medical_evaluation.vlm.schemas import VlmReview


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
                "dam_stage_presence_ratio": 0.5,
                "dam_final_presence_ratio": 0.8,
                "dam_area_ratio": 0.4,
                "nose_overlap": 0.01,
                "visible_frame_area_ratio": 0.0,
            }
        )


class MissingCp09Extractor(FakeExtractor):
    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: object,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id == "cp_09":
            self.calls.append((checkpoint_id, dense_fps, analysis_width))
            raise EvaluationInputMissing("cp_09 requires prompts")
        return super().extract(
            video_path,
            checkpoint_id,
            time_range,
            dense_fps=dense_fps,
            analysis_width=analysis_width,
        )


class EvidenceExtractor(FakeExtractor):
    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: object,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        extracted = super().extract(
            video_path,
            checkpoint_id,
            time_range,
            dense_fps=dense_fps,
            analysis_width=analysis_width,
        )
        if checkpoint_id != "cp_09":
            return extracted
        return extracted.model_copy(
            update={
                "evidence": [
                    EvidenceItem(
                        time_sec=1.0,
                        overlay_path="cp_09/overlays/frame.jpg",
                        rule="test_evidence",
                    )
                ]
            }
        )


class FakeLocalizer:
    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence

    def checkpoint_confidence(self, _checkpoint_id: str) -> float:
        return self.confidence


class FakeReviewer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests = []

    def review(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("review service failed")
        return VlmReview(
            evidence_supported=True,
            semantic_status="supports",
            reason_zh=f"证据支持{request.checkpoint_id}的确定性结论。",
            suggestion_zh="继续保持规范操作。",
            cited_evidence_indices=[],
        )


def make_pipeline(
    tmp_path: Path,
    *,
    extractor: FakeExtractor | None = None,
    reviewer: FakeReviewer | None = None,
) -> tuple[AnalysisPipeline, FakeExtractor, FakeLocalizer]:
    rubric = load_rubric(Path(__file__).parents[1] / "config" / "rubric.yaml")
    extractor = extractor or FakeExtractor()
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
        commentary_provider=reviewer,
    )
    return pipeline, extractor, localizer


def make_job(tmp_path: Path) -> JobRecord:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    return JobRecord(id="job-1", video_id="success", video_path=str(video))


def test_pipeline_writes_two_real_decisions_and_nine_review_results(tmp_path: Path) -> None:
    pipeline, extractor, _ = make_pipeline(tmp_path)

    report = pipeline.run(make_job(tmp_path))

    assert len(report.checkpoints) == 11
    assert report.checkpoints[8].status.value == "correct"
    assert report.checkpoints[10].status.value == "correct"
    assert report.checkpoints[0].status.value == "needs_review"
    assert report.checkpoints[0].reason_code == "automatic_evaluation_not_implemented"
    assert report.checkpoints[0].included_in_provisional_score is False
    assert report.summary.evaluated_count == 2
    assert [call[0] for call in extractor.calls] == ["cp_09", "cp_11"]
    stored = json.loads((tmp_path / "jobs" / "job-1" / "report.json").read_text("utf-8"))
    assert len(stored["checkpoints"]) == 11


def test_pipeline_retries_oom_once_with_degraded_sampling(tmp_path: Path) -> None:
    pipeline, extractor, _ = make_pipeline(tmp_path)
    extractor.fail_once_with_oom = True

    report = pipeline.run(make_job(tmp_path))

    assert report.audit.degradations == ["cuda_oom:dense_fps=1.0,analysis_width=960"]
    assert extractor.calls[:2] == [("cp_09", 5, 1280), ("cp_09", 1, 960)]


def test_low_alignment_confidence_pauses_checkpoint(tmp_path: Path) -> None:
    pipeline, extractor, localizer = make_pipeline(tmp_path)
    localizer.confidence = 0.2

    report = pipeline.run(make_job(tmp_path))

    assert all(item.status.value == "needs_review" for item in report.checkpoints)
    assert report.checkpoints[0].reason_code == "automatic_evaluation_not_implemented"
    assert report.checkpoints[8].reason_code == "low_step_alignment_confidence"
    assert report.checkpoints[10].reason_code == "low_step_alignment_confidence"
    assert extractor.calls == []


def test_missing_cp09_input_excludes_cp09_but_still_runs_cp11(tmp_path: Path) -> None:
    pipeline, extractor, _ = make_pipeline(tmp_path, extractor=MissingCp09Extractor())

    report = pipeline.run(make_job(tmp_path))

    assert report.checkpoints[8].reason_code == "automatic_evaluation_input_missing"
    assert report.checkpoints[8].included_in_provisional_score is False
    assert report.checkpoints[10].included_in_provisional_score is True
    assert report.summary.evaluated_count == 1
    assert [call[0] for call in extractor.calls] == ["cp_09", "cp_11"]


def test_commentary_is_attached_without_mutating_deterministic_decision(tmp_path: Path) -> None:
    reviewer = FakeReviewer()
    pipeline, _, _ = make_pipeline(tmp_path, reviewer=reviewer)

    report = pipeline.run(make_job(tmp_path))

    cp09 = report.checkpoints[8]
    assert [request.checkpoint_id for request in reviewer.requests] == ["cp_09", "cp_11"]
    assert reviewer.requests[0].deterministic_status == "correct"
    assert cp09.status.value == "correct"
    assert cp09.reason_code == "criteria_satisfied"
    assert cp09.ai_commentary is not None
    assert cp09.ai_commentary.reason_zh == "证据支持cp_09的确定性结论。"
    assert report.checkpoints[0].ai_commentary is None


def test_commentary_resolves_evidence_inside_job_directory(tmp_path: Path) -> None:
    evidence_path = tmp_path / "jobs" / "job-1" / "cp_09" / "overlays" / "frame.jpg"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_bytes(b"jpeg")
    reviewer = FakeReviewer()
    pipeline, _, _ = make_pipeline(
        tmp_path,
        extractor=EvidenceExtractor(),
        reviewer=reviewer,
    )

    pipeline.run(make_job(tmp_path))

    assert reviewer.requests[0].evidence_images == [evidence_path]
    assert reviewer.requests[0].evidence_images[0].is_file()


def test_commentary_provider_failure_uses_template_and_still_writes_report(
    tmp_path: Path,
) -> None:
    pipeline, _, _ = make_pipeline(tmp_path, reviewer=FakeReviewer(fail=True))

    report = pipeline.run(make_job(tmp_path))

    assert report.checkpoints[8].status.value == "correct"
    assert report.checkpoints[8].ai_commentary is not None
    assert report.checkpoints[8].ai_commentary.source == "template_fallback"
    assert (tmp_path / "jobs" / "job-1" / "report.json").is_file()
