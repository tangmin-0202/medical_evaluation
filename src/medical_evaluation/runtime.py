from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from medical_evaluation.annotations import AnnotationStore
from medical_evaluation.extractors.cp09 import Cp09FeatureExtractor
from medical_evaluation.extractors.cp09_cp11 import Cp09Cp11FeatureExtractor
from medical_evaluation.extractors.cp11 import Cp11FeatureExtractor
from medical_evaluation.jobs import JobRecord
from medical_evaluation.pipeline import AnalysisPipeline, ConfidenceProvider, FeatureExtractor
from medical_evaluation.rubric import load_rubric
from medical_evaluation.segmentation.sam2_backend import Sam2Backend
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
    segmenter_factory: SegmenterFactory = Sam2Backend,
) -> AnalysisPipeline:
    checkpoint_path = settings.sam2_checkpoint_path
    if not checkpoint_path.is_file():
        raise ValueError(f"SAM2 checkpoint is missing: {checkpoint_path}")

    rubric = load_rubric(settings.rubric_path)
    rules = {item.id: item for item in rubric.checkpoints}
    annotation_store = AnnotationStore(settings.data_dir / "annotations")
    segmenter = segmenter_factory(
        settings.sam2_model_config,
        checkpoint_path,
        device=settings.sam_device,
    )

    def extractor_factory(job: JobRecord) -> FeatureExtractor:
        annotations = annotation_store.load_segments(job.video_id)
        segments = {item.checkpoint_id: item for item in annotations.steps}
        if "cp_09" not in segments:
            raise ValueError(f"video {job.video_id} has no CP09 time range")
        evidence_root = settings.data_dir / "jobs" / job.id
        cp09 = Cp09FeatureExtractor(
            segmenter=segmenter,
            annotations=annotations,
            evidence_root=evidence_root,
        )
        cp11_rule = rules["cp_11"]
        cp11 = Cp11FeatureExtractor(
            segmenter=segmenter,
            annotations=annotations,
            evidence_root=evidence_root,
            frame_reference_time_range=segments["cp_09"].time_range,
            min_stage_dam_presence_ratio=cp11_rule.thresholds[
                "min_stage_dam_presence_ratio"
            ],
            min_final_dam_presence_ratio=cp11_rule.thresholds[
                "min_final_dam_presence_ratio"
            ],
        )
        return Cp09Cp11FeatureExtractor(
            cp09=cp09,
            cp11=cp11,
            annotations=annotations,
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
