from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from medical_evaluation.annotations import AnnotationStore
from medical_evaluation.extractors.cp02 import Cp02FeatureExtractor
from medical_evaluation.jobs import JobRecord
from medical_evaluation.pipeline import AnalysisPipeline
from medical_evaluation.rubric import load_rubric
from medical_evaluation.runtime import ConfirmedAnnotationLocalizer, build_analysis_pipeline
from medical_evaluation.settings import Settings

VIDEO_FILENAMES = {
    "success": "橡皮障完整.mp4",
    "failure": "橡皮障失败.mp4",
    "clamp_failure": "橡皮障夹子飞了.mp4",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the real CP02/CP08/CP09/CP10/CP11 report pipeline",
    )
    parser.add_argument("--video-id", choices=tuple(VIDEO_FILENAMES), required=True)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--no-commentary", action="store_true")
    parser.add_argument(
        "--only",
        choices=("cp_02", "cp_08", "cp_09", "cp_10", "cp_11"),
        default=None,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = Settings(project_root=Path.cwd(), pipeline_mode="real", sam_backend="sam3")
    if args.only == "cp_02":
        annotation_store = AnnotationStore(settings.data_dir / "annotations")

        def cp02_factory(job: JobRecord) -> Cp02FeatureExtractor:
            return Cp02FeatureExtractor(
                annotations=annotation_store.load_segments(job.video_id),
                evidence_root=settings.data_dir / "jobs" / job.id,
            )

        pipeline = AnalysisPipeline(
            rubric=load_rubric(settings.rubric_path),
            extractor_factory=cp02_factory,
            localizer=ConfirmedAnnotationLocalizer(),
            output_root=settings.data_dir / "jobs",
            annotation_store=annotation_store,
            dense_fps=settings.sample_fps,
            analysis_width=settings.analysis_width,
            enabled_checkpoint_ids=frozenset({"cp_02"}),
            commentary_provider=None,
        )
    else:
        pipeline = build_analysis_pipeline(settings)
    if args.no_commentary:
        pipeline.commentary_provider = None
    if args.only:
        pipeline.enabled_checkpoint_ids = frozenset({args.only})
    job = JobRecord(
        id=args.job_id or f"sam3-{args.video_id}-{uuid4().hex[:12]}",
        video_id=args.video_id,
        video_path=str(settings.videos_dir / VIDEO_FILENAMES[args.video_id]),
    )
    report = pipeline.run(job)
    selected = {
        item.checkpoint_id: {
            "status": item.status.value,
            "reason_code": item.reason_code,
            "features": item.features,
            "evidence_count": len(item.evidence),
            "commentary_source": (
                item.ai_commentary.source if item.ai_commentary is not None else None
            ),
        }
        for item in report.checkpoints
        if item.checkpoint_id in {"cp_02", "cp_08", "cp_09", "cp_10", "cp_11"}
    }
    print(json.dumps({"job_id": job.id, "results": selected}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
