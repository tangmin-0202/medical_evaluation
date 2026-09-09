from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from medical_evaluation.annotations import AnnotationStore
from medical_evaluation.domain import CheckpointStatus, TimeRange
from medical_evaluation.jobs import JobRecord
from medical_evaluation.judges.base import JudgeDecision
from medical_evaluation.judges.cp01_cp03 import judge_cp01, judge_cp02, judge_cp03
from medical_evaluation.judges.cp04_cp06 import judge_cp04, judge_cp05, judge_cp06
from medical_evaluation.judges.cp07_cp08 import judge_cp07, judge_cp08
from medical_evaluation.judges.cp09_cp11 import judge_cp09, judge_cp10, judge_cp11
from medical_evaluation.reporting import (
    CheckpointResult,
    EvaluationReport,
    EvidenceItem,
    RunAudit,
)
from medical_evaluation.rubric import CheckpointRule, Rubric
from medical_evaluation.storage import atomic_write_json
from medical_evaluation.vlm.client import template_fallback
from medical_evaluation.vlm.schemas import VlmReview, VlmReviewRequest


class ExtractedEvidence(BaseModel):
    features: dict[str, float | bool | None]
    evidence: list[EvidenceItem] = Field(default_factory=list)


class EvaluationInputMissing(ValueError):
    """Raised when a supported checkpoint lacks required model inputs."""


class FeatureExtractor(Protocol):
    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence: ...


class ConfidenceProvider(Protocol):
    def checkpoint_confidence(self, checkpoint_id: str) -> float: ...


class CommentaryProvider(Protocol):
    def review(self, request: VlmReviewRequest) -> VlmReview: ...


Judge = Callable[[dict[str, float | bool | None], dict[str, float]], JudgeDecision]

JUDGES: dict[str, Judge] = {
    "cp_01": judge_cp01,
    "cp_02": judge_cp02,
    "cp_03": judge_cp03,
    "cp_04": judge_cp04,
    "cp_05": judge_cp05,
    "cp_06": judge_cp06,
    "cp_07": judge_cp07,
    "cp_08": judge_cp08,
    "cp_09": judge_cp09,
    "cp_10": judge_cp10,
    "cp_11": judge_cp11,
}


class AnalysisPipeline:
    def __init__(
        self,
        *,
        rubric: Rubric,
        extractor: FeatureExtractor | None = None,
        extractor_factory: Callable[[JobRecord], FeatureExtractor] | None = None,
        localizer: ConfidenceProvider,
        output_root: Path,
        annotation_store: AnnotationStore | None = None,
        dense_fps: float = 5,
        analysis_width: int = 1280,
        fallback_dense_fps: float = 1,
        fallback_analysis_width: int = 960,
        minimum_alignment_confidence: float = 0.5,
        enabled_checkpoint_ids: frozenset[str] = frozenset(
            {"cp_09", "cp_10", "cp_11"}
        ),
        commentary_provider: CommentaryProvider | None = None,
    ) -> None:
        if (extractor is None) == (extractor_factory is None):
            raise ValueError("provide exactly one extractor or extractor_factory")
        self.rubric = rubric
        self.extractor = extractor
        self.extractor_factory = extractor_factory
        self.localizer = localizer
        self.output_root = output_root
        self.annotation_store = annotation_store
        self.dense_fps = dense_fps
        self.analysis_width = analysis_width
        self.fallback_dense_fps = fallback_dense_fps
        self.fallback_analysis_width = fallback_analysis_width
        self.minimum_alignment_confidence = minimum_alignment_confidence
        self.enabled_checkpoint_ids = enabled_checkpoint_ids
        self.commentary_provider = commentary_provider
        self.judges = dict(JUDGES)

    def run(
        self,
        job: JobRecord,
        progress: Callable[[float, str], None] | None = None,
    ) -> EvaluationReport:
        started = datetime.now(UTC)
        degradations: list[str] = []
        results: list[CheckpointResult] = []
        extractor = (
            self.extractor_factory(job)
            if self.extractor_factory is not None
            else self.extractor
        )
        assert extractor is not None
        annotated_ranges = self._annotation_ranges(job.video_id)
        for index, checkpoint in enumerate(self.rubric.checkpoints, start=1):
            time_range = annotated_ranges.get(checkpoint.id, checkpoint.reference_time)
            if checkpoint.id not in self.enabled_checkpoint_ids:
                result = self._review_result(
                    checkpoint,
                    time_range,
                    "automatic_evaluation_not_implemented",
                    "该考核点尚未接入当前自动评估纵向切片。",
                    included_in_provisional_score=False,
                )
            else:
                alignment_confidence = self.localizer.checkpoint_confidence(checkpoint.id)
                if alignment_confidence < self.minimum_alignment_confidence:
                    result = self._review_result(
                        checkpoint,
                        time_range,
                        "low_step_alignment_confidence",
                        "该步骤的自动时间对齐置信度不足。",
                        confidence=alignment_confidence,
                    )
                else:
                    try:
                        evidence = self._extract_with_retry(
                            job,
                            checkpoint,
                            time_range,
                            degradations,
                            extractor,
                        )
                    except EvaluationInputMissing:
                        result = self._review_result(
                            checkpoint,
                            time_range,
                            "automatic_evaluation_input_missing",
                            "该考核点缺少自动评估所需的有效提示或参考输入。",
                            included_in_provisional_score=False,
                        )
                    else:
                        decision = self.judges[checkpoint.id](
                            evidence.features,
                            checkpoint.thresholds,
                        )
                        result = self._to_result(decision, time_range, evidence.evidence)
                        if self.commentary_provider is not None:
                            result = result.model_copy(
                                update={
                                    "ai_commentary": self._review(
                                        checkpoint,
                                        result,
                                        self.output_root / job.id,
                                    )
                                }
                            )
            results.append(result)
            if progress is not None:
                progress(index / 11, checkpoint.id)

        completed = datetime.now(UTC)
        report = EvaluationReport(
            job_id=job.id,
            video_id=job.video_id,
            checkpoints=results,
            audit=RunAudit(
                rubric_version=self.rubric.version,
                model_versions={"pipeline": "vertical-cp09-cp10-cp11"},
                started_at=started,
                completed_at=completed,
                runtime_sec=(completed - started).total_seconds(),
                degradations=degradations,
            ),
            overall_feedback=self._overall_feedback(results),
        )
        atomic_write_json(
            self.output_root / job.id / "report.json",
            report.model_dump(mode="json"),
        )
        return report

    def _review(
        self,
        checkpoint: CheckpointRule,
        result: CheckpointResult,
        evidence_root: Path,
    ) -> VlmReview:
        assert self.commentary_provider is not None
        request = VlmReviewRequest(
            checkpoint_id=checkpoint.id,
            checkpoint_name=checkpoint.name,
            criteria=checkpoint.criteria,
            deterministic_status=result.status.value,
            reason_code=result.reason_code,
            features=result.features,
            evidence_images=[
                evidence_root / item.overlay_path for item in result.evidence
            ],
        )
        try:
            review = self.commentary_provider.review(request)
        except RuntimeError:
            return template_fallback(result.reason_code)
        if result.status is not CheckpointStatus.NEEDS_REVIEW and (
            review.semantic_status != "supports"
        ):
            return template_fallback(result.reason_code)
        return review

    @staticmethod
    def _overall_feedback(results: list[CheckpointResult]) -> str:
        evaluated = [item for item in results if item.included_in_provisional_score]
        prefix = (
            f"本报告仅自动评估 {len(evaluated)}/11 项，"
            "以下为阶段性结果，不是最终成绩。"
        )
        comments = [
            f"{item.checkpoint_id.upper()}：{item.ai_commentary.reason_zh}"
            for item in evaluated
            if item.ai_commentary is not None
        ]
        return " ".join([prefix, *comments])

    def _extract_with_retry(
        self,
        job: JobRecord,
        checkpoint: CheckpointRule,
        time_range: TimeRange,
        degradations: list[str],
        extractor: FeatureExtractor,
    ) -> ExtractedEvidence:
        try:
            return extractor.extract(
                Path(job.video_path),
                checkpoint.id,
                time_range,
                dense_fps=self.dense_fps,
                analysis_width=self.analysis_width,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            _release_cuda_cache()
            degradation = (
                f"cuda_oom:dense_fps={self.fallback_dense_fps:.1f},"
                f"analysis_width={self.fallback_analysis_width}"
            )
            degradations.append(degradation)
            return extractor.extract(
                Path(job.video_path),
                checkpoint.id,
                time_range,
                dense_fps=self.fallback_dense_fps,
                analysis_width=self.fallback_analysis_width,
            )

    def _annotation_ranges(self, video_id: str) -> dict[str, TimeRange]:
        if self.annotation_store is None:
            return {}
        try:
            annotations = self.annotation_store.load_segments(video_id)
        except FileNotFoundError:
            return {}
        return {step.checkpoint_id: step.time_range for step in annotations.steps}

    @staticmethod
    def _review_result(
        checkpoint: CheckpointRule,
        time_range: TimeRange,
        reason_code: str,
        reason: str,
        *,
        confidence: float = 0,
        included_in_provisional_score: bool = True,
    ) -> CheckpointResult:
        return CheckpointResult(
            checkpoint_id=checkpoint.id,
            status=CheckpointStatus.NEEDS_REVIEW,
            confidence=confidence,
            time_range=time_range,
            reason_code=reason_code,
            reason=reason,
            suggestion="请人工复核该步骤。",
            included_in_provisional_score=included_in_provisional_score,
        )

    @staticmethod
    def _to_result(
        decision: JudgeDecision,
        time_range: TimeRange,
        evidence: list[EvidenceItem],
    ) -> CheckpointResult:
        return CheckpointResult(
            checkpoint_id=decision.checkpoint_id,
            status=decision.status,
            confidence=decision.confidence,
            time_range=time_range,
            evidence=evidence,
            features=decision.features,
            matched_rules=decision.matched_rules,
            reason_code=decision.reason_code,
            reason=decision.reason,
            suggestion=decision.suggestion,
        )


def _release_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
