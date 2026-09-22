from __future__ import annotations

import json
from pathlib import Path

import pytest

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

    @property
    def model_version(self) -> str:
        return "fake-cp02-cp04-cp08-cp09-cp10-cp11-v1"

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
        if checkpoint_id == "cp_02":
            return ExtractedEvidence(
                features={
                    "stage_scan_reliable": True,
                    "punch_action_observed": True,
                    "selected_second_largest": True,
                    "residue_before": False,
                    "cleanup_contact_observed": None,
                    "residue_after": False,
                }
            )
        if checkpoint_id == "cp_08":
            return ExtractedEvidence(
                features={
                    "instrument_observed": True,
                    "instrument_shape_reliable": True,
                    "instrument_shape_match": True,
                    "final_state_observable": True,
                    "rubber_dam_positioned": True,
                    "left_wing_complete": True,
                    "right_wing_complete": True,
                    "left_wing_hole_detected": True,
                    "right_wing_hole_detected": True,
                    "left_wing_hole_valid_frame_count": 3.0,
                    "right_wing_hole_valid_frame_count": 3.0,
                    "left_wing_hole_dam_color_ratio": 0.9,
                    "right_wing_hole_dam_color_ratio": 0.9,
                    "left_wing_hole_non_dam_color_ratio": 0.1,
                    "right_wing_hole_non_dam_color_ratio": 0.1,
                }
            )
        if checkpoint_id == "cp_03":
            return ExtractedEvidence(
                features={
                    "final_scan_reliable": True,
                    "hole_observed": True,
                    "hole_clear_consecutive_frames": 3.0,
                    "hole_adhesion_free": True,
                    "adhesion_observed_frame_count": 0.0,
                }
            )
        if checkpoint_id == "cp_04":
            return ExtractedEvidence(
                features={
                    "clamp_observed": True,
                    "shape_evidence_reliable": True,
                    "clear_frame_count": 3.0,
                    "matching_frame_count": 3.0,
                    "clamp_reference_similarity": 0.9,
                    "evidence_consistent": True,
                }
            )
        if checkpoint_id == "cp_09":
            return ExtractedEvidence(features={"frame_oral_center_offset": 0.03})
        if checkpoint_id == "cp_10":
            return ExtractedEvidence(
                features={
                    "tooth_anchor_reliable": True,
                    "floss_observed_frame_count": 4.0,
                    "upper_contact_frame_count": 2.0,
                    "lower_contact_frame_count": 2.0,
                }
            )
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


class UncertainReviewer(FakeReviewer):
    def review(self, request):
        review = super().review(request)
        return review.model_copy(
            update={
                "semantic_status": "uncertain",
                "reason_zh": "无法确认确定性结论。",
            }
        )


class UnsafeCp08Reviewer(FakeReviewer):
    def __init__(self, tip_claim: str) -> None:
        super().__init__()
        self.tip_claim = tip_claim

    def review(self, request):
        review = super().review(request)
        if request.checkpoint_id != "cp_08":
            return review
        return review.model_copy(
            update={
                "reason_zh": (
                    "器具符合约定外形，长柄细杆弯曲端证据一致；"
                    f"已经证明使用的是{self.tip_claim}器具。"
                ),
                "suggestion_zh": f"继续使用{self.tip_claim}器具。",
            }
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


def test_pipeline_writes_three_real_decisions_and_eight_review_results(tmp_path: Path) -> None:
    pipeline, extractor, _ = make_pipeline(tmp_path)

    report = pipeline.run(make_job(tmp_path))

    assert len(report.checkpoints) == 11
    assert report.checkpoints[8].status.value == "correct"
    assert report.checkpoints[9].status.value == "correct"
    assert report.checkpoints[10].status.value == "correct"
    assert report.checkpoints[0].status.value == "needs_review"
    assert report.checkpoints[0].reason_code == "automatic_evaluation_not_implemented"
    assert report.checkpoints[0].included_in_provisional_score is False
    assert report.summary.evaluated_count == 3
    assert [call[0] for call in extractor.calls] == ["cp_09", "cp_10", "cp_11"]
    stored = json.loads((tmp_path / "jobs" / "job-1" / "report.json").read_text("utf-8"))
    assert len(stored["checkpoints"]) == 11


def test_pipeline_can_enable_cp04_as_seven_real_decisions(tmp_path: Path) -> None:
    pipeline, extractor, _ = make_pipeline(tmp_path)
    pipeline.enabled_checkpoint_ids = frozenset(
        {"cp_02", "cp_03", "cp_04", "cp_08", "cp_09", "cp_10", "cp_11"}
    )

    report = pipeline.run(make_job(tmp_path))

    assert report.checkpoints[1].status.value == "correct"
    assert report.checkpoints[1].reason_code == "criteria_satisfied"
    assert report.checkpoints[2].status.value == "correct"
    assert report.checkpoints[2].reason_code == "criteria_satisfied"
    assert report.checkpoints[3].status.value == "correct"
    assert report.checkpoints[3].reason_code == "criteria_satisfied"
    assert report.checkpoints[7].status.value == "correct"
    assert report.checkpoints[7].reason_code == "criteria_satisfied"
    assert report.summary.evaluated_count == 7
    assert report.summary.final_score is None
    assert report.audit.model_versions == {
        "pipeline": "vertical-cp02-cp03-cp04-cp08-cp09-cp10-cp11",
        "extractor": "fake-cp02-cp04-cp08-cp09-cp10-cp11-v1",
    }
    assert [call[0] for call in extractor.calls] == [
        "cp_02",
        "cp_03",
        "cp_04",
        "cp_08",
        "cp_09",
        "cp_10",
        "cp_11",
    ]


def test_pipeline_audit_uses_extractor_created_by_factory(tmp_path: Path) -> None:
    pipeline, extractor, _ = make_pipeline(tmp_path)
    pipeline.extractor = None
    pipeline.extractor_factory = lambda _job: extractor

    report = pipeline.run(make_job(tmp_path))

    assert report.audit.model_versions["extractor"] == extractor.model_version


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
    assert report.checkpoints[9].reason_code == "low_step_alignment_confidence"
    assert report.checkpoints[10].reason_code == "low_step_alignment_confidence"
    assert extractor.calls == []


def test_missing_cp09_input_excludes_cp09_but_still_runs_cp11(tmp_path: Path) -> None:
    pipeline, extractor, _ = make_pipeline(tmp_path, extractor=MissingCp09Extractor())

    report = pipeline.run(make_job(tmp_path))

    assert report.checkpoints[8].reason_code == "automatic_evaluation_input_missing"
    assert report.checkpoints[8].included_in_provisional_score is False
    assert report.checkpoints[10].included_in_provisional_score is True
    assert report.summary.evaluated_count == 2
    assert [call[0] for call in extractor.calls] == ["cp_09", "cp_10", "cp_11"]


def test_commentary_is_attached_without_mutating_deterministic_decision(tmp_path: Path) -> None:
    reviewer = FakeReviewer()
    pipeline, _, _ = make_pipeline(tmp_path, reviewer=reviewer)

    report = pipeline.run(make_job(tmp_path))

    cp09 = report.checkpoints[8]
    assert [request.checkpoint_id for request in reviewer.requests] == [
        "cp_09",
        "cp_10",
        "cp_11",
    ]
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


def test_uncertain_commentary_cannot_replace_deterministic_explanation(
    tmp_path: Path,
) -> None:
    pipeline, _, _ = make_pipeline(tmp_path, reviewer=UncertainReviewer())

    report = pipeline.run(make_job(tmp_path))

    assert report.checkpoints[8].status.value == "correct"
    assert report.checkpoints[8].ai_commentary is not None
    assert report.checkpoints[8].ai_commentary.source == "template_fallback"
    assert report.checkpoints[8].ai_commentary.reason_zh == (
        "现有证据支持确定性规则的通过结论。"
    )


@pytest.mark.parametrize(
    "tip_claim",
    ["钝头", "尖锐", "圆钝", "钝性", "尖头", "锐利", "针尖", "探针"],
)
def test_cp08_commentary_cannot_claim_an_unmeasured_tip_type(
    tmp_path: Path,
    tip_claim: str,
) -> None:
    pipeline, _, _ = make_pipeline(
        tmp_path,
        reviewer=UnsafeCp08Reviewer(tip_claim),
    )
    pipeline.enabled_checkpoint_ids = frozenset({"cp_08"})

    report = pipeline.run(make_job(tmp_path))

    cp08 = report.checkpoints[7]
    assert cp08.status.value == "correct"
    assert cp08.reason_code == "criteria_satisfied"
    assert cp08.ai_commentary is not None
    assert cp08.ai_commentary.source == "template_fallback"
    commentary = cp08.ai_commentary.reason_zh + cp08.ai_commentary.suggestion_zh
    assert "符合约定外形" in commentary
    assert "长柄细杆弯曲端" in commentary
    assert tip_claim not in commentary
