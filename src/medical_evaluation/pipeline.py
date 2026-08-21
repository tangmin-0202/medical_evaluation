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
from medical_evaluation.judges.cp09_cp11 import judge_cp09, judge_cp11
from medical_evaluation.reporting import (
    CheckpointResult,
    EvaluationReport,
    EvidenceItem,
    RunAudit,
)
from medical_evaluation.rubric import CheckpointRule, Rubric
from medical_evaluation.storage import atomic_write_json


class ExtractedEvidence(BaseModel):
    features: dict[str, float | bool | None]
    evidence: list[EvidenceItem] = Field(default_factory=list)


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


Judge = Callable[[dict[str, float | bool | None], dict[str, float]], JudgeDecision]


class AnalysisPipeline:
    def __init__(
        self,
        *,
        rubric: Rubric,
        extractor: FeatureExtractor,
        localizer: ConfidenceProvider,
        output_root: Path,
        annotation_store: AnnotationStore | None = None,
        dense_fps: float = 5,
        analysis_width: int = 1280,
        fallback_dense_fps: float = 1,
        fallback_analysis_width: int = 960,
        minimum_alignment_confidence: float = 0.5,
    ) -> None:
        self.rubric = rubric
        self.extractor = extractor
        self.localizer = localizer
        self.output_root = output_root
        self.annotation_store = annotation_store
        self.dense_fps = dense_fps
        self.analysis_width = analysis_width
        self.fallback_dense_fps = fallback_dense_fps
        self.fallback_analysis_width = fallback_analysis_width
        self.minimum_alignment_confidence = minimum_alignment_confidence
        self.judges: dict[str, Judge] = {"cp_09": judge_cp09, "cp_11": judge_cp11}

    def run(
        self,
        job: JobRecord,
        progress: Callable[[float, str], None] | None = None,
    ) -> EvaluationReport:
        started = datetime.now(UTC)
        degradations: list[str] = []
        results: list[CheckpointResult] = []
        annotated_ranges = self._annotation_ranges(job.video_id)
        for index, checkpoint in enumerate(self.rubric.checkpoints, start=1):
            time_range = annotated_ranges.get(checkpoint.id, checkpoint.reference_time)
            alignment_confidence = self.localizer.checkpoint_confidence(checkpoint.id)
            if alignment_confidence < self.minimum_alignment_confidence:
                result = self._review_result(
                    checkpoint,
                    time_range,
                    "low_step_alignment_confidence",
                    "该步骤的自动时间对齐置信度不足。",
                    confidence=alignment_confidence,
                )
            elif checkpoint.id not in self.judges:
                result = self._review_result(
                    checkpoint,
                    time_range,
                    "judge_not_implemented",
                    "该考核点的判定规则尚未接入当前纵切片。",
                )
            else:
                evidence = self._extract_with_retry(job, checkpoint, time_range, degradations)
                decision = self.judges[checkpoint.id](evidence.features, checkpoint.thresholds)
                result = self._to_result(decision, time_range, evidence.evidence)
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
                model_versions={"pipeline": "vertical-cp09-cp11"},
                started_at=started,
                completed_at=completed,
                runtime_sec=(completed - started).total_seconds(),
                degradations=degradations,
            ),
            overall_feedback="考核点 9 和 11 已自动判定，其余考核点等待规则接入或人工复核。",
        )
        atomic_write_json(
            self.output_root / job.id / "report.json",
            report.model_dump(mode="json"),
        )
        return report

    def _extract_with_retry(
        self,
        job: JobRecord,
        checkpoint: CheckpointRule,
        time_range: TimeRange,
        degradations: list[str],
    ) -> ExtractedEvidence:
        try:
            return self.extractor.extract(
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
            return self.extractor.extract(
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
    ) -> CheckpointResult:
        return CheckpointResult(
            checkpoint_id=checkpoint.id,
            status=CheckpointStatus.NEEDS_REVIEW,
            confidence=confidence,
            time_range=time_range,
            reason_code=reason_code,
            reason=reason,
            suggestion="请人工复核该步骤。",
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
