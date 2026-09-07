from __future__ import annotations

import argparse
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np

from medical_evaluation.annotations import AnnotationStore
from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import SegmentationPrompt
from medical_evaluation.segmentation.sam3_backend import Sam3Backend
from medical_evaluation.settings import Settings
from medical_evaluation.storage import atomic_write_json
from medical_evaluation.video import read_frame, write_sampled_frame_sequence
from medical_evaluation.web.routes import PRESETS

HEAD_PROMPTS = (
    "dental training mannequin head",
    "plastic dental mannequin face",
    "dental mannequin head and face",
)
NOSE_PROMPTS = (
    "nose of the dental mannequin",
    "plastic nose on the dental mannequin face",
    "mannequin nose",
)
REQUIRED_VIDEO_IDS = ("success", "failure", "clamp_failure")


class TrialFailure(RuntimeError):
    def __init__(
        self,
        *,
        sample: Mapping[str, object],
        prompt: str,
        rows: list[dict[str, object]],
        cause: Exception,
    ) -> None:
        super().__init__(str(cause))
        self.sample = sample
        self.prompt = prompt
        self.rows = rows
        self.cause = cause
        self.rows_by_candidate: dict[str, list[dict[str, object]]] = {}
        self.elapsed_seconds = 0.0
        self.cuda_peak_allocated_mib: float | None = None


def build_parser() -> argparse.ArgumentParser:
    settings = Settings()
    parser = argparse.ArgumentParser(
        description=(
            "Test bounded SAM3 head/nose prompts over CP09 and CP11 annotation ranges. "
            "Video IDs resolve through the project's preset manifest."
        ),
        epilog=(
            "Server example: PYTHONPATH=\"$PWD/src\" python "
            "scripts/run_sam3_prompt_matrix.py --annotations data/annotations "
            "--videos videos --video-ids success failure clamp_failure "
            "--output data/runs/head-nose-prompt-gate --sample-fps 2 "
            "--output-threshold 0.2 --grounding-batch-size 4 --probe-multiplex"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument("--videos", type=Path, default=Path("videos"))
    parser.add_argument("--video-ids", nargs="+", default=("success", "failure", "clamp_failure"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--output-threshold", type=float, default=0.2)
    parser.add_argument("--grounding-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=settings.sam3_checkpoint_path)
    parser.add_argument("--bpe-path", type=Path, default=settings.sam3_bpe_path)
    parser.add_argument("--device", default=settings.sam_device)
    parser.add_argument("--probe-multiplex", action="store_true")
    return parser


def mask_measurements(mask: np.ndarray) -> tuple[float, float, bool]:
    """Return full-frame area, dominant component fraction, and row validity."""
    binary = np.asarray(mask, dtype=bool)
    total = int(binary.sum())
    if total == 0:
        return 0.0, 0.0, False
    _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    dominant = int(stats[1:, cv2.CC_STAT_AREA].max(initial=0))
    dominant_ratio = dominant / total
    return float(binary.mean()), dominant_ratio, dominant_ratio >= 0.60


def candidate_passes(rows: Sequence[Mapping[str, object]], minimum_consecutive: int = 3) -> bool:
    consecutive = 0
    prior_trial_id: str | None = None
    prior_position: int | None = None
    for row in rows:
        trial_id = row.get("trial_id")
        position = row.get("sample_position")
        if not isinstance(trial_id, str) or not isinstance(position, int):
            consecutive = 0
            prior_trial_id = None
            prior_position = None
            continue
        if row.get("valid") is not True:
            consecutive = 0
        elif (
            trial_id == prior_trial_id
            and prior_position is not None
            and position == prior_position + 1
        ):
            consecutive += 1
        else:
            consecutive = 1
        prior_trial_id = trial_id
        prior_position = position
        if consecutive >= minimum_consecutive:
            return True
    return False


def rank_candidates(
    rows_by_candidate: Mapping[str, Sequence[Mapping[str, object]]],
    candidates: Sequence[str],
) -> list[str]:
    def rank_key(candidate: str) -> tuple[int, float, int]:
        by_sample: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows_by_candidate.get(candidate, []):
            by_sample[str(row["sample_key"])].append(row)
        continuity_count = sum(candidate_passes(rows) for rows in by_sample.values())
        scores = [float(row["score"]) for row in rows_by_candidate.get(candidate, []) if row.get("score") is not None]
        median_score = float(np.median(scores)) if scores else float("-inf")
        return continuity_count, median_score, -candidates.index(candidate)

    return sorted(candidates, key=rank_key, reverse=True)


def select_prompts(
    rows_by_candidate: Mapping[str, Sequence[Mapping[str, object]]],
    required_samples: Mapping[str, Sequence[str]],
    head_candidates: Sequence[str] = HEAD_PROMPTS,
    nose_candidates: Sequence[str] = NOSE_PROMPTS,
) -> dict[str, object]:
    def accepted(candidate: str, required: Sequence[str]) -> bool:
        grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows_by_candidate.get(candidate, []):
            grouped[str(row["sample_key"])].append(row)
        return all(candidate_passes(grouped[sample]) for sample in required)

    ranked_head = rank_candidates(rows_by_candidate, head_candidates)
    ranked_nose = rank_candidates(rows_by_candidate, nose_candidates)
    head_prompt = next((item for item in ranked_head if accepted(item, required_samples["head"])), None)
    nose_prompt = next((item for item in ranked_nose if accepted(item, required_samples["nose"])), None)
    return {
        "accepted": head_prompt is not None and nose_prompt is not None,
        "head_prompt": head_prompt,
        "nose_prompt": nose_prompt,
        "head_ranking": ranked_head,
        "nose_ranking": ranked_nose,
        "session_strategy": "independent",
    }


def exit_status(selected_prompts: Mapping[str, object]) -> int:
    return 0 if selected_prompts.get("accepted") is True else 2


def required_samples(video_ids: Sequence[str]) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    provided = set(video_ids)
    missing = tuple(video_id for video_id in REQUIRED_VIDEO_IDS if video_id not in provided)
    return (
        {
            "head": tuple(
                f"{video_id}:{stage}"
                for video_id in REQUIRED_VIDEO_IDS
                for stage in ("cp_09", "cp_11")
            ),
            "nose": tuple(f"{video_id}:cp_11" for video_id in REQUIRED_VIDEO_IDS),
        },
        missing,
    )


def handle_multiplex_probe(probe: Callable[[], Mapping[str, object]]) -> dict[str, object]:
    try:
        payload = probe()
        consecutive = 0
        expected_ids: frozenset[int] | None = None
        previous_index: int | None = None
        for item in payload["source_frames"]:
            if not isinstance(item, Mapping):
                raise TypeError("multiplex probe frame metadata must be a mapping")
            frame_index = int(item["local_sample_index"])
            object_ids = frozenset(int(value) for value in item["object_ids"])
            if len(object_ids) != 2:
                consecutive = 0
                expected_ids = None
                previous_index = None
                continue
            if (
                expected_ids != object_ids
                or previous_index is None
                or frame_index != previous_index + 1
            ):
                expected_ids = object_ids
                consecutive = 1
            else:
                consecutive += 1
            previous_index = frame_index
            if consecutive >= 3:
                return {
                    "supported": True,
                    "result": "two distinct object IDs persisted for 3 consecutive source frames",
                }
        return {"supported": False, "result": "merged or missing object identities"}
    except Exception as exc:  # noqa: BLE001 - unsupported official APIs must be recorded
        return {"supported": False, "result": f"{type(exc).__name__}: {exc}"}


def required_prompt_time(time_range: TimeRange) -> float:
    """Include the annotation end frame; one-second stages then yield 0, .5, and 1.0s."""
    return time_range.end_sec


def _run_gate_impl(
    *,
    annotations_dir: Path,
    videos_dir: Path,
    video_ids: Sequence[str],
    output_dir: Path,
    sample_fps: float,
    output_threshold: float,
    backend_factory: Callable[[], Sam3Backend],
    read_frame_fn: Callable[[Path, int], np.ndarray] = read_frame,
    git_revision_fn: Callable[[], str] | None = None,
    cuda_peak_mib_fn: Callable[[], float | None] | None = None,
    frame_score_adapter: Callable[[object, str], float | None] | None = None,
    probe_multiplex: bool = False,
    grounding_batch_size: int | None = None,
    checkpoint_path: Path | None = None,
    sam3_revision_fn: Callable[[], str] | None = None,
) -> int:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if len(video_ids) != len(REQUIRED_VIDEO_IDS) or set(video_ids) != set(
        REQUIRED_VIDEO_IDS
    ):
        raise ValueError(
            "video_ids must contain success, failure, and clamp_failure exactly once"
        )
    git_revision_fn = git_revision_fn or _git_revision
    cuda_peak_mib_fn = cuda_peak_mib_fn or _cuda_peak_allocated_mib
    sam3_revision_fn = sam3_revision_fn or _sam3_revision
    frame_score_adapter = frame_score_adapter or extract_frame_score
    output_dir.mkdir(parents=True, exist_ok=False)
    samples = _load_samples(annotations_dir, videos_dir, video_ids)
    backend = backend_factory()  # one loaded model, with a fresh session per prompt trial
    started = time.perf_counter()
    rows_by_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)

    for sample in samples:
        candidates = HEAD_PROMPTS if sample["object"] == "head" else NOSE_PROMPTS
        for prompt_text in candidates:
            try:
                rows_by_candidate[prompt_text].extend(
                    _run_sample(
                    backend=backend,
                    sample=sample,
                    prompt_text=prompt_text,
                    output_dir=output_dir,
                    sample_fps=sample_fps,
                    read_frame_fn=read_frame_fn,
                    frame_score_adapter=frame_score_adapter,
                    )
                )
            except TrialFailure as exc:
                rows_by_candidate[prompt_text].extend(exc.rows)
                exc.rows_by_candidate = dict(rows_by_candidate)
                exc.elapsed_seconds = time.perf_counter() - started
                exc.cuda_peak_allocated_mib = cuda_peak_mib_fn()
                raise

    required, missing_required_video_ids = required_samples(video_ids)
    selected = select_prompts(rows_by_candidate, required)
    selected["missing_required_video_ids"] = list(missing_required_video_ids)
    multiplex = None
    if probe_multiplex:
        representative = next(item for item in samples if item["object"] == "head")
        multiplex = handle_multiplex_probe(
            lambda: _probe_multiplex_session(
                backend.predictor,
                representative["video_path"],
                representative["time_range"],
                sample_fps,
                output_threshold,
            )
        )
    selected["multiplex_probe"] = multiplex
    atomic_write_json(output_dir / "selected_prompts.json", selected)
    atomic_write_json(
        output_dir / "summary.json",
        {
            "output_threshold": output_threshold,
            "sample_fps": sample_fps,
            "grounding_batch_size": grounding_batch_size,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "model_version": backend.model_version,
            "sam3_source_revision": sam3_revision_fn(),
            "git_revision": git_revision_fn(),
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_peak_allocated_mib": cuda_peak_mib_fn(),
            "samples": [_sample_metadata(sample) for sample in samples],
            "rows_by_candidate": dict(rows_by_candidate),
            "missing_required_video_ids": list(missing_required_video_ids),
            "multiplex_probe": multiplex,
        },
    )
    return exit_status(selected)


def run_gate(**kwargs: Any) -> int:
    """Run the gate and persist a failed audit record for any trial/setup exception."""
    try:
        return _run_gate_impl(**kwargs)
    except Exception as exc:  # noqa: BLE001 - experiment failures require an audit artifact
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(exc, TrialFailure):
            failure = {
                "video": exc.sample["video_id"],
                "stage": exc.sample["stage"],
                "object": exc.sample["object"],
                "prompt": exc.prompt,
                "exception_type": type(exc.cause).__name__,
                "message": str(exc.cause),
                "completed_row_count": len(exc.rows),
                "elapsed_seconds": exc.elapsed_seconds,
                "cuda_peak_allocated_mib": exc.cuda_peak_allocated_mib,
            }
            rows_by_candidate = exc.rows_by_candidate
        else:
            failure = {
                "video": None,
                "stage": None,
                "object": None,
                "prompt": None,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "completed_row_count": 0,
                "elapsed_seconds": 0.0,
                "cuda_peak_allocated_mib": None,
            }
            rows_by_candidate = {}
        selected = {
            "accepted": False,
            "head_prompt": None,
            "nose_prompt": None,
            "session_strategy": "independent",
            "failures": [failure],
        }
        atomic_write_json(output_dir / "selected_prompts.json", selected)
        atomic_write_json(
            output_dir / "summary.json",
            {"accepted": False, "failures": [failure], "rows_by_candidate": rows_by_candidate},
        )
        return 2


def _load_samples(annotations_dir: Path, videos_dir: Path, video_ids: Sequence[str]) -> list[dict[str, object]]:
    store = AnnotationStore(annotations_dir)
    samples: list[dict[str, object]] = []
    for video_id in video_ids:
        if video_id not in PRESETS:
            raise ValueError(f"unknown preset video ID: {video_id}")
        annotations = store.load_segments(video_id)
        ranges = {step.checkpoint_id: step.time_range for step in annotations.steps}
        for stage in ("cp_09", "cp_11"):
            if stage not in ranges:
                raise ValueError(f"{video_id} is missing {stage} annotation range")
        video_path = videos_dir / PRESETS[video_id]
        samples.extend(
            (
                {"video_id": video_id, "stage": "cp_09", "object": "head", "time_range": ranges["cp_09"], "video_path": video_path},
                {"video_id": video_id, "stage": "cp_11", "object": "head", "time_range": ranges["cp_11"], "video_path": video_path},
                {"video_id": video_id, "stage": "cp_11", "object": "nose", "time_range": TimeRange(start_sec=max(ranges["cp_11"].start_sec, ranges["cp_11"].end_sec - 3), end_sec=ranges["cp_11"].end_sec), "video_path": video_path},
            )
        )
    return samples


def _sample_metadata(sample: Mapping[str, object]) -> dict[str, object]:
    time_range = sample["time_range"]
    video_path = sample["video_path"]
    assert isinstance(time_range, TimeRange) and isinstance(video_path, Path)
    return {
        "video_id": sample["video_id"],
        "stage": sample["stage"],
        "object": sample["object"],
        "time_range": time_range.model_dump(mode="json"),
        "video_path": str(video_path),
    }


def _run_sample(
    *,
    backend: Sam3Backend,
    sample: Mapping[str, object],
    prompt_text: str,
    output_dir: Path,
    sample_fps: float,
    read_frame_fn: Callable[[Path, int], np.ndarray],
    frame_score_adapter: Callable[[object, str], float | None],
) -> list[dict[str, object]]:
    video_id = str(sample["video_id"])
    stage = str(sample["stage"])
    object_id = str(sample["object"])
    time_range = sample["time_range"]
    video_path = sample["video_path"]
    assert isinstance(time_range, TimeRange) and isinstance(video_path, Path)
    artifact_dir = output_dir / "overlays" / video_id / object_id / _slug(prompt_text)
    for directory in (artifact_dir / "raw", artifact_dir / "masks", artifact_dir / "overlays"):
        directory.mkdir(parents=True, exist_ok=True)
    prompt = SegmentationPrompt(
        object_id=object_id,
        kind="text",
        frame_time_sec=required_prompt_time(time_range),
        text=prompt_text,
    )
    rows: list[dict[str, object]] = []
    stream = backend.track(video_path, time_range, [prompt], sample_fps)
    try:
        for frame in stream:
            mask = np.asarray(
                frame.masks.get(object_id, np.zeros((1, 1), dtype=bool)),
                dtype=bool,
            )
            area_ratio, dominant_ratio, valid = mask_measurements(mask)
            frame_name = f"{frame.frame_index:08d}"
            raw = read_frame_fn(video_path, frame.frame_index)
            _write_artifacts(artifact_dir, frame_name, raw, mask)
            rows.append(
                {
                    "sample_key": f"{video_id}:{stage}",
                    "video_id": video_id,
                    "stage": stage,
                    "source_time_sec": frame.frame_time_sec,
                    "frame_index": frame.frame_index,
                    "mask_area_ratio": area_ratio,
                    "dominant_component_ratio": dominant_ratio,
                    "score": frame_score_adapter(frame, object_id),
                    "score_source": getattr(frame, "score_sources", {}).get(object_id),
                    "trial_id": f"{video_id}:{stage}:{object_id}:{_slug(prompt_text)}",
                    "sample_position": _sample_position(
                        time_range, sample_fps, frame.frame_time_sec
                    ),
                    "valid": valid,
                }
            )
    except Exception as exc:
        raise TrialFailure(
            sample=sample, prompt=prompt_text, rows=rows, cause=exc
        ) from exc
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    return rows


def _sample_position(time_range: TimeRange, sample_fps: float, frame_time_sec: float) -> int:
    """Index in the expected sampled timeline; duplicate and missing positions stay visible."""
    return round((frame_time_sec - time_range.start_sec) * sample_fps)


def _write_artifacts(directory: Path, name: str, raw: np.ndarray, mask: np.ndarray) -> None:
    if not cv2.imwrite(str(directory / "raw" / f"{name}.jpg"), raw):
        raise OSError(f"could not write raw frame {name}")
    mask_image = mask.astype(np.uint8) * 255
    if not cv2.imwrite(str(directory / "masks" / f"{name}.png"), mask_image):
        raise OSError(f"could not write mask {name}")
    overlay = raw.copy()
    overlay[mask] = (0, 255, 0)
    if not cv2.imwrite(str(directory / "overlays" / f"{name}.png"), cv2.addWeighted(raw, 0.6, overlay, 0.4, 0)):
        raise OSError(f"could not write overlay {name}")


def extract_frame_score(frame: object, object_id: str) -> float | None:
    """Read optional score metadata without changing the VideoSegmenter contract."""
    for source in (getattr(frame, "scores", None), getattr(frame, "metadata", None)):
        if not isinstance(source, Mapping):
            continue
        candidate = source.get(object_id)
        if candidate is None and isinstance(source.get("scores"), Mapping):
            candidate = source["scores"].get(object_id)
        if candidate is not None:
            value = float(candidate)
            return value if np.isfinite(value) else None
    value = getattr(frame, "score", None)
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _probe_multiplex_session(
    predictor: Any, video_path: Path, time_range: TimeRange, sample_fps: float, output_threshold: float
) -> dict[str, object]:
    with TemporaryDirectory(prefix="medical-evaluation-sam3-probe-") as temporary:
        sequence = write_sampled_frame_sequence(video_path, Path(temporary) / "frames", time_range=time_range, sample_fps=sample_fps, required_times_sec=[time_range.start_sec])
        session = predictor.handle_request({"type": "start_session", "resource_path": str(sequence.directory), "offload_video_to_cpu": True})
        session_id = str(session["session_id"])
        try:
            head = _select_candidate_id(predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": HEAD_PROMPTS[0], "output_prob_thresh": output_threshold}))
            frame = _select_candidate_id(predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": "thin white U-shaped plastic frame around the mouth", "output_prob_thresh": output_threshold}))
            if head is None or frame is None or head == frame:
                return {"source_frames": []}
            source_frames: list[dict[str, object]] = []
            for item in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id, "propagation_direction": "both", "start_frame_index": 0, "output_prob_thresh": output_threshold}):
                source_frames.append(
                    {
                        "local_sample_index": int(item["frame_index"]),
                        "source_frame_index": sequence.entries[
                            int(item["frame_index"])
                        ].source_frame_index,
                        "source_time_sec": sequence.entries[
                            int(item["frame_index"])
                        ].source_time_sec,
                        "object_ids": {
                            int(value)
                            for value in _as_numpy(item["outputs"]["out_obj_ids"]).reshape(-1)
                            if int(value) in {head, frame}
                        },
                    }
                )
            return {"source_frames": source_frames}
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})


def _slug(text: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in text).strip("-")


def _as_numpy(value: object) -> np.ndarray:
    detached = value.detach() if hasattr(value, "detach") else value
    cpu_value = detached.cpu() if hasattr(detached, "cpu") else detached
    return np.asarray(cpu_value)


def _select_candidate_id(prompt_result: Mapping[str, object]) -> int | None:
    outputs = prompt_result.get("outputs")
    if not isinstance(outputs, Mapping):
        raise TypeError("SAM3 add_prompt response has no outputs mapping")
    object_ids = _as_numpy(outputs.get("out_obj_ids", [])).reshape(-1)
    if object_ids.size == 0:
        return None
    if object_ids.size == 1:
        return int(object_ids[0])
    scores = outputs.get("out_scores")
    if scores is None:
        raise RuntimeError("SAM3 text prompt returned multiple candidates without scores")
    score_values = _as_numpy(scores).reshape(-1)
    if score_values.size != object_ids.size:
        raise ValueError("SAM3 candidate scores do not match object IDs")
    return int(object_ids[int(np.argmax(score_values))])


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cuda_peak_allocated_mib() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)


def _sam3_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", "external/sam3", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_gate(
        annotations_dir=args.annotations,
        videos_dir=args.videos,
        video_ids=tuple(args.video_ids),
        output_dir=args.output,
        sample_fps=args.sample_fps,
        output_threshold=args.output_threshold,
        backend_factory=lambda: Sam3Backend(
            args.checkpoint,
            bpe_path=args.bpe_path,
            device=args.device,
            output_prob_threshold=args.output_threshold,
            grounding_batch_size=args.grounding_batch_size,
        ),
        probe_multiplex=args.probe_multiplex,
        grounding_batch_size=args.grounding_batch_size,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
