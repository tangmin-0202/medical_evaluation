from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol

from medical_evaluation.annotations import AnnotationStore, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.extractors.cp01 import Cp01FeatureExtractor
from medical_evaluation.extractors.cp11 import Cp11FeatureExtractor
from medical_evaluation.judges.cp01_cp03 import judge_cp01
from medical_evaluation.judges.cp09_cp11 import judge_cp11
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.rubric import Rubric, load_rubric
from medical_evaluation.segmentation.sam2_backend import Sam2Backend
from medical_evaluation.storage import atomic_write_json, safe_child

VIDEO_FILENAMES = {
    "success": "橡皮障完整.mp4",
    "failure": "橡皮障失败.mp4",
    "clamp_failure": "橡皮障夹子飞了.mp4",
}


class Extractor(Protocol):
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


def run_cp01_cp11_smoke(
    *,
    cp01_extractor: Extractor,
    cp11_extractor: Extractor,
    video_path: Path,
    annotations: VideoAnnotations,
    rubric: Rubric,
    run_root: Path,
    video_id: str,
    sample_fps: float,
    degradations: list[str] | None = None,
) -> dict[str, object]:
    if video_id not in VIDEO_FILENAMES:
        raise ValueError(f"unsupported combined smoke video_id: {video_id}")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    run_root.mkdir(parents=True, exist_ok=True)
    rules = {item.id: item for item in rubric.checkpoints}
    segments = {item.checkpoint_id: item for item in annotations.steps}
    extractors = {"cp_01": cp01_extractor, "cp_11": cp11_extractor}
    judges = {"cp_01": judge_cp01, "cp_11": judge_cp11}
    started = datetime.now(UTC)
    timer = perf_counter()
    checkpoint_summaries: dict[str, object] = {}
    decisions: dict[str, object] = {}
    for checkpoint_id in ("cp_01", "cp_11"):
        extractor = extractors[checkpoint_id]
        segment = segments[checkpoint_id]
        result = extractor.extract(
            video_path,
            checkpoint_id,
            segment.time_range,
            dense_fps=sample_fps,
            analysis_width=1280,
        )
        decision = judges[checkpoint_id](
            result.features,
            rules[checkpoint_id].thresholds,
        )
        checkpoint_summaries[checkpoint_id] = {
            "time_range": segment.time_range.model_dump(mode="json"),
            "features": result.features,
            "evidence_count": len(result.evidence),
            "model_version": extractor.model_version,
        }
        decisions[checkpoint_id] = decision.model_dump(mode="json")
    summary: dict[str, object] = {
        "video_id": video_id,
        "checkpoints": checkpoint_summaries,
        "sample_fps": sample_fps,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_sec": perf_counter() - timer,
        "degradations": list(degradations or []),
    }
    atomic_write_json(run_root / "summary.json", summary)
    atomic_write_json(run_root / "decisions.json", decisions)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CP01 and CP11 in one SAM2 job")
    parser.add_argument("--video-id", choices=tuple(VIDEO_FILENAMES), required=True)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-fps", default=2.0, type=float)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--cp01-reference", type=Path)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    reference_path = args.cp01_reference or data_dir / "calibration/cp01_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    annotations = AnnotationStore(data_dir / "annotations").load_segments(args.video_id)
    rubric = load_rubric(Path("config/rubric.yaml"))
    cp11_rule = next(item for item in rubric.checkpoints if item.id == "cp_11")
    cp09_segment = next(item for item in annotations.steps if item.checkpoint_id == "cp_09")
    run_name = args.output_dir or datetime.now(UTC).strftime(
        f"smoke-cp01-cp11-{args.video_id}-%Y%m%dT%H%M%SZ"
    )
    run_root = safe_child((data_dir / "runs").resolve(), run_name)
    backend = Sam2Backend(
        args.model_config,
        args.checkpoint_path.resolve(),
        device=args.device,
    )
    cp01 = Cp01FeatureExtractor(
        segmenter=backend,
        annotations=annotations,
        evidence_root=run_root,
        reference_u=float(reference["reference_u"]),
        reference_v=float(reference["reference_v"]),
    )
    cp11 = Cp11FeatureExtractor(
        segmenter=backend,
        annotations=annotations,
        evidence_root=run_root,
        frame_reference_time_range=cp09_segment.time_range,
        min_stage_dam_presence_ratio=cp11_rule.thresholds[
            "min_stage_dam_presence_ratio"
        ],
        min_final_dam_presence_ratio=cp11_rule.thresholds[
            "min_final_dam_presence_ratio"
        ],
    )
    summary = run_cp01_cp11_smoke(
        cp01_extractor=cp01,
        cp11_extractor=cp11,
        video_path=args.videos_dir.resolve() / VIDEO_FILENAMES[args.video_id],
        annotations=annotations,
        rubric=rubric,
        run_root=run_root,
        video_id=args.video_id,
        sample_fps=args.sample_fps,
    )
    print(f"summary: {run_root / 'summary.json'}")
    print(f"decisions: {run_root / 'decisions.json'}")
    print(f"status inputs: {summary['checkpoints']}")


if __name__ == "__main__":
    main()
