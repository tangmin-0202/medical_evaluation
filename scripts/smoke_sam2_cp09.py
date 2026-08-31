from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol

from medical_evaluation.annotations import AnnotationStore, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp09 import Cp09FeatureExtractor
from medical_evaluation.judges.cp09_cp11 import judge_cp09
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.rubric import Rubric, load_rubric
from medical_evaluation.segmentation.sam2_backend import Sam2Backend
from medical_evaluation.storage import atomic_write_json, safe_child


class Cp09Extractor(Protocol):
    @property
    def model_version(self) -> str: ...

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence: ...


def run_cp09_smoke(
    *,
    extractor: Cp09Extractor,
    video_path: Path,
    annotations: VideoAnnotations,
    rubric: Rubric,
    run_root: Path,
    video_id: str = "success",
    checkpoint_id: str = "cp_09",
    sample_fps: float = 2.0,
    degradations: list[str] | None = None,
) -> dict[str, object]:
    if video_id != "success":
        raise ValueError("CP09 smoke only accepts video_id=success")
    if checkpoint_id != "cp_09":
        raise ValueError("CP09 smoke only accepts checkpoint_id=cp_09")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    rule = next(item for item in rubric.checkpoints if item.id == checkpoint_id)
    segment = next(item for item in annotations.steps if item.checkpoint_id == checkpoint_id)
    started = datetime.now(UTC)
    started_timer = perf_counter()
    result = extractor.extract(
        video_path,
        checkpoint_id,
        segment.time_range,
        dense_fps=sample_fps,
        analysis_width=1280,
    )
    decision = judge_cp09(result.features, rule.thresholds)
    completed = datetime.now(UTC)
    boundary_tolerance_sec = Cp09FeatureExtractor.prompt_boundary_tolerance_sec
    allowed_start = segment.time_range.start_sec - boundary_tolerance_sec
    allowed_end = segment.time_range.end_sec + boundary_tolerance_sec
    prompt_counts = {
        object_id: sum(
            prompt.object_id == object_id
            and allowed_start <= prompt.frame_time_sec <= allowed_end
            for prompt in annotations.prompts
        )
        for object_id in ("rubber_dam_frame", "oral_region")
    }
    summary: dict[str, object] = {
        "video_id": video_id,
        "checkpoint_id": checkpoint_id,
        "time_range": segment.time_range.model_dump(mode="json"),
        "prompt_counts": prompt_counts,
        "valid_frame_count": result.features.get("frame_valid_count", 0.0),
        "oral_reference_count": result.features.get("oral_reference_count", 0.0),
        "relative_offset_valid_count": result.features.get(
            "relative_offset_valid_count",
            0.0,
        ),
        "evidence_count": len(result.evidence),
        "features": result.features,
        "model_version": extractor.model_version,
        "sample_fps": sample_fps,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "runtime_sec": perf_counter() - started_timer,
        "degradations": list(degradations or []),
    }
    atomic_write_json(run_root / "summary.json", summary)
    atomic_write_json(run_root / "decision.json", decision.model_dump(mode="json"))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real SAM2 CP09 smoke slice")
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-fps", default=2.0, type=float)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", help="Run directory name beneath data/runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    runs_root = (data_dir / "runs").resolve()
    run_name = args.output_dir or datetime.now(UTC).strftime("smoke-cp09-%Y%m%dT%H%M%SZ")
    run_root = safe_child(runs_root, run_name)
    run_root.mkdir(parents=True, exist_ok=True)
    annotations = AnnotationStore(data_dir / "annotations").load_segments("success")
    rubric = load_rubric(Path("config/rubric.yaml"))
    backend = Sam2Backend(
        args.model_config,
        args.checkpoint_path.resolve(),
        device=args.device,
    )
    extractor = Cp09FeatureExtractor(
        segmenter=backend,
        annotations=annotations,
        evidence_root=run_root,
    )
    video_path = args.videos_dir.resolve() / "橡皮障完整.mp4"
    degradations: list[str] = []
    try:
        summary = run_cp09_smoke(
            extractor=extractor,
            video_path=video_path,
            annotations=annotations,
            rubric=rubric,
            run_root=run_root,
            sample_fps=args.sample_fps,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        fallback_fps = min(args.sample_fps, 1.0)
        degradations.append(f"cuda_oom:sample_fps={fallback_fps:.1f}")
        _release_cuda_cache()
        summary = run_cp09_smoke(
            extractor=extractor,
            video_path=video_path,
            annotations=annotations,
            rubric=rubric,
            run_root=run_root,
            sample_fps=fallback_fps,
            degradations=degradations,
        )
    print(f"summary: {run_root / 'summary.json'}")
    print(f"overlays: {run_root / 'cp_09' / 'overlays'}")
    print(f"status inputs: {summary['features']}")


def _release_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
