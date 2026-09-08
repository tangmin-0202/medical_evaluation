from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from medical_evaluation.annotations import AnnotationStore
from medical_evaluation.extractors.cp09 import Cp09FeatureExtractor
from medical_evaluation.extractors.cp09_cp11 import Cp09Cp11FeatureExtractor
from medical_evaluation.extractors.cp11 import Cp11FeatureExtractor
from medical_evaluation.jobs import JobRecord
from medical_evaluation.pipeline import AnalysisPipeline, ConfidenceProvider, FeatureExtractor
from medical_evaluation.rubric import load_rubric
from medical_evaluation.segmentation.audit import AuditedVideoSegmenter
from medical_evaluation.segmentation.prompt_policy import (
    AnnotationPromptPolicy,
    TextPromptPolicy,
)
from medical_evaluation.segmentation.sam2_backend import Sam2Backend
from medical_evaluation.segmentation.sam3_backend import Sam3Backend
from medical_evaluation.settings import Settings
from medical_evaluation.vlm.client import QwenVlmClient

SegmenterFactory = Callable[..., Any]


class ConfirmedAnnotationLocalizer(ConfidenceProvider):
    """Treat manually confirmed per-video stage ranges as fully aligned."""

    def checkpoint_confidence(self, _checkpoint_id: str) -> float:
        return 1.0


def build_analysis_pipeline(
    settings: Settings,
    *,
    segmenter_factory: SegmenterFactory | None = None,
) -> AnalysisPipeline:
    rubric = load_rubric(settings.rubric_path)
    rules = {item.id: item for item in rubric.checkpoints}
    annotation_store = AnnotationStore(settings.data_dir / "annotations")
    if settings.sam_backend == "sam2":
        checkpoint_path = settings.sam2_checkpoint_path
        if not checkpoint_path.is_file():
            raise ValueError(f"SAM2 checkpoint is missing: {checkpoint_path}")
        factory = segmenter_factory or Sam2Backend
        segmenter = factory(
            settings.sam2_model_config,
            checkpoint_path,
            device=settings.sam_device,
        )
        prompt_policy = AnnotationPromptPolicy()
    else:
        checkpoint_path = settings.sam3_checkpoint_path
        if not checkpoint_path.is_file():
            raise ValueError(f"SAM3 checkpoint is missing: {checkpoint_path}")
        if not settings.sam3_bpe_path.is_file():
            raise ValueError(f"SAM3 bpe is missing: {settings.sam3_bpe_path}")
        factory = segmenter_factory or Sam3Backend
        segmenter = factory(
            checkpoint_path,
            bpe_path=settings.sam3_bpe_path,
            device=settings.sam_device,
            output_prob_threshold=settings.sam3_output_prob_threshold,
            grounding_batch_size=settings.sam3_grounding_batch_size,
        )
        prompt_policy = TextPromptPolicy()
    template: dict[str, object] = {}
    if settings.sam_backend == "sam3":
        if not settings.mannequin_template_path.is_file():
            raise ValueError(
                f"mannequin template is missing: {settings.mannequin_template_path}"
            )
        template = json.loads(settings.mannequin_template_path.read_text(encoding="utf-8"))

    def extractor_factory(job: JobRecord) -> FeatureExtractor:
        annotations = annotation_store.load_segments(job.video_id)
        segments = {item.checkpoint_id: item for item in annotations.steps}
        if "cp_09" not in segments:
            raise ValueError(f"video {job.video_id} has no CP09 time range")
        evidence_root = settings.data_dir / "jobs" / job.id
        static_metadata = {
            "backend": settings.sam_backend,
            "model_version": segmenter.model_version,
        }
        if settings.sam_backend == "sam3":
            static_metadata.update(
                {
                    "source_revision": settings.sam3_source_revision,
                    "checkpoint_sha256": settings.sam3_checkpoint_sha256,
                    "grounding_batch_size": settings.sam3_grounding_batch_size,
                }
            )
        audited_segmenter = AuditedVideoSegmenter(
            segmenter,
            output_path=evidence_root / "segmentation_metadata.json",
            static_metadata=static_metadata,
        )
        cp09 = Cp09FeatureExtractor(
            segmenter=audited_segmenter,
            annotations=annotations,
            evidence_root=evidence_root,
            prompt_policy=prompt_policy,
            template=template,
        )
        cp11_rule = rules["cp_11"]
        cp11 = Cp11FeatureExtractor(
            segmenter=audited_segmenter,
            annotations=annotations,
            evidence_root=evidence_root,
            frame_reference_time_range=segments["cp_09"].time_range,
            min_stage_dam_presence_ratio=cp11_rule.thresholds[
                "min_stage_dam_presence_ratio"
            ],
            min_final_dam_presence_ratio=cp11_rule.thresholds[
                "min_final_dam_presence_ratio"
            ],
            prompt_policy=prompt_policy,
            template=template,
        )
        return Cp09Cp11FeatureExtractor(
            cp09=cp09,
            cp11=cp11,
            annotations=annotations,
            prompt_policy=prompt_policy,
        )

    return AnalysisPipeline(
        rubric=rubric,
        extractor_factory=extractor_factory,
        localizer=ConfirmedAnnotationLocalizer(),
        output_root=settings.data_dir / "jobs",
        annotation_store=annotation_store,
        dense_fps=settings.sample_fps,
        analysis_width=settings.analysis_width,
        commentary_provider=QwenVlmClient(
            settings.vlm_base_url,
            settings.vlm_model,
        ),
    )
