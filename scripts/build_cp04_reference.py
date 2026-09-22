from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from medical_evaluation.annotations import AnnotationStore
from medical_evaluation.jobs import JobRecord
from medical_evaluation.runtime import build_analysis_pipeline
from medical_evaluation.settings import Settings
from scripts.run_real_cp09_cp11 import VIDEO_FILENAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the versioned CP04 clamp reference from the success video",
    )
    parser.add_argument("--video-id", choices=("success",), default="success")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--job-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = Settings(project_root=Path.cwd(), pipeline_mode="real", sam_backend="sam3")
    pipeline = build_analysis_pipeline(settings)
    job = JobRecord(
        id=args.job_id or f"cp04-reference-{uuid4().hex[:12]}",
        video_id="success",
        video_path=str(settings.videos_dir / VIDEO_FILENAMES["success"]),
    )
    composite = pipeline.extractor_factory(job)
    if composite.cp04 is None:
        raise RuntimeError("CP04 extractor is not configured")
    annotations = AnnotationStore(settings.data_dir / "annotations").load_segments("success")
    cp04_step = next(item for item in annotations.steps if item.checkpoint_id == "cp_04")
    manifest = composite.cp04.build_reference(
        Path(job.video_path),
        cp04_step.time_range,
        video_id="success",
        replace=args.replace,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
