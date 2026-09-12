from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import FrameMasks, SegmentationPrompt
from medical_evaluation.storage import atomic_write_json
from medical_evaluation.video import SampledFrame, read_frame, sample_frames


PROMPT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "blunt_instrument": (
        "blunt dental instrument",
        "dental instrument with a rounded blunt working tip",
        "metal dental instrument with a curved shaft and rounded working tip",
    ),
    "sharp_probe": (
        "sharp dental explorer probe",
        "pointed dental probe with a needle-like tip",
    ),
    "target_tooth": (
        "isolated white tooth inside the rubber dam clamp",
        "white molar enclosed by the metal rubber dam clamp",
    ),
    "rubber_dam_clamp": (
        "metal rubber dam clamp around the tooth",
        "complete stainless steel rubber dam clamp with two side wings",
    ),
    "rubber_dam": ("green dental rubber dam",),
}

CP08_SPARSE_FPS = 1.0
CP08_DENSE_FPS = 5.0
CP08_DENSE_HALF_WINDOW_SEC = 1.5
CP09_TAIL_FPS = 5.0
MIN_MASK_AREA_PX = 64


@dataclass(frozen=True)
class GateWindows:
    cp08: tuple[float, float]
    cp09_tail: tuple[float, float]


def load_gate_windows(annotation_path: Path) -> GateWindows:
    annotations = VideoAnnotations.model_validate_json(
        annotation_path.read_text(encoding="utf-8")
    )
    ranges = {step.checkpoint_id: step.time_range for step in annotations.steps}
    for checkpoint_id in ("cp_08", "cp_09"):
        if checkpoint_id not in ranges:
            raise ValueError(f"Missing {checkpoint_id} annotation range")
    cp08 = ranges["cp_08"]
    cp09 = ranges["cp_09"]
    return GateWindows(
        cp08=(cp08.start_sec, cp08.end_sec),
        cp09_tail=(max(cp09.start_sec, cp09.end_sec - 3.0), cp09.end_sec),
    )


def _max_true_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _dominant_component_ratio(mask: np.ndarray) -> float:
    binary = np.asarray(mask, dtype=np.uint8)
    foreground = int(binary.sum())
    if foreground == 0:
        return 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    return largest / foreground


def summarize_masks(masks: list[np.ndarray], *, min_area_px: int) -> dict[str, Any]:
    areas = [int(np.asarray(mask, dtype=bool).sum()) for mask in masks]
    valid = [area >= min_area_px for area in areas]
    coherence = [
        _dominant_component_ratio(mask)
        for mask, is_valid in zip(masks, valid, strict=True)
        if is_valid
    ]
    median_coherence = float(np.median(coherence)) if coherence else 0.0
    longest = _max_true_run(valid)
    return {
        "frame_count": len(masks),
        "valid_frame_count": sum(valid),
        "max_consecutive_valid_frames": longest,
        "median_dominant_component_ratio": median_coherence,
        "automatic_gate_passed": longest >= 3 and median_coherence >= 0.60,
    }


def select_prompt(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one prompt result is required")
    return max(
        candidates,
        key=lambda item: (
            bool(item["automatic_gate_passed"]),
            int(item["max_consecutive_valid_frames"]),
            float(item["median_dominant_component_ratio"]),
        ),
    )


def _slug(text: str) -> str:
    return "-".join(text.lower().split())


def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_artifacts(
    *,
    output_dir: Path,
    artifact_dir: Path,
    frame_index: int,
    frame_time_sec: float,
    object_id: str,
    prompt_text: str,
    raw: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    if raw.ndim != 3 or raw.shape[:2] != mask.shape:
        raise ValueError("raw frame and mask dimensions do not match")
    for name in ("raw", "masks", "overlays"):
        (artifact_dir / name).mkdir(parents=True, exist_ok=True)
    stem = f"{frame_index:08d}"
    raw_path = artifact_dir / "raw" / f"{stem}.jpg"
    mask_path = artifact_dir / "masks" / f"{stem}.png"
    overlay_path = artifact_dir / "overlays" / f"{stem}.jpg"
    mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
    overlay = raw.copy()
    colored = raw.copy()
    colored[np.asarray(mask, dtype=bool)] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 0.65, colored, 0.35, 0)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    cv2.putText(
        overlay,
        f"{object_id} t={frame_time_sec:.3f}s area={int(mask_u8.sum() / 255)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        prompt_text[:90],
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for path, image in ((raw_path, raw), (mask_path, mask_u8), (overlay_path, overlay)):
        if not cv2.imwrite(str(path), image):
            raise OSError(f"could not write artifact: {path}")

    def relative(path: Path) -> str:
        return path.relative_to(output_dir).as_posix()

    return {
        "frame_index": frame_index,
        "frame_time_sec": frame_time_sec,
        "mask_area_px": int(np.asarray(mask, dtype=bool).sum()),
        "raw_path": relative(raw_path),
        "mask_path": relative(mask_path),
        "overlay_path": relative(overlay_path),
    }


def _run_prompt(
    *,
    segmenter: Any,
    video_path: Path,
    time_range: TimeRange,
    object_id: str,
    prompt_text: str,
    sample_fps: float,
    phase: str,
    output_dir: Path,
    read_frame_fn: Callable[[Path, int], np.ndarray],
    sample_frames_fn: Callable[..., Iterable[SampledFrame]],
) -> dict[str, Any]:
    prompt = SegmentationPrompt(
        object_id=object_id,
        kind="text",
        frame_time_sec=time_range.end_sec,
        text=prompt_text,
    )
    tracked = list(segmenter.track(video_path, time_range, [prompt], sample_fps))
    artifact_dir = output_dir / object_id / _slug(prompt_text) / phase
    frames: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    if tracked:
        source: Iterable[tuple[FrameMasks, np.ndarray]] = (
            (item, read_frame_fn(video_path, item.frame_index)) for item in tracked
        )
    else:
        sampled = sample_frames_fn(
            video_path,
            start_sec=time_range.start_sec,
            end_sec=time_range.end_sec,
            sample_fps=sample_fps,
        )
        source = (
            (
                FrameMasks(
                    frame_index=item.frame_index,
                    frame_time_sec=item.time_sec,
                    masks={object_id: np.zeros(item.image_bgr.shape[:2], dtype=bool)},
                ),
                item.image_bgr,
            )
            for item in sampled
        )
    for item, raw in source:
        mask = np.asarray(
            item.masks.get(object_id, np.zeros(raw.shape[:2], dtype=bool)),
            dtype=bool,
        )
        masks.append(mask)
        frames.append(
            _write_artifacts(
                output_dir=output_dir,
                artifact_dir=artifact_dir,
                frame_index=item.frame_index,
                frame_time_sec=item.frame_time_sec,
                object_id=object_id,
                prompt_text=prompt_text,
                raw=raw,
                mask=mask,
            )
        )
    metrics = summarize_masks(masks, min_area_px=MIN_MASK_AREA_PX)
    first_candidate = next(
        (
            frame["frame_time_sec"]
            for frame in frames
            if frame["mask_area_px"] >= MIN_MASK_AREA_PX
        ),
        None,
    )
    return {
        "time_range": time_range.model_dump(mode="json"),
        "sample_fps": sample_fps,
        "first_candidate_time_sec": first_candidate,
        "frames": frames,
        **metrics,
    }


def _selection_metrics(run: dict[str, Any]) -> dict[str, Any]:
    for phase in ("dense", "tail", "sparse"):
        if phase in run:
            return {"prompt": run["prompt"], **run[phase]}
    raise ValueError("prompt run has no phase results")


def run_gate(
    *,
    segmenter: Any,
    video_path: Path,
    annotation_path: Path,
    output_dir: Path,
    read_frame_fn: Callable[[Path, int], np.ndarray] = read_frame,
    sample_frames_fn: Callable[..., Iterable[SampledFrame]] = sample_frames,
) -> dict[str, Any]:
    """Run the CP08 feasibility experiment with text prompts only."""
    _prepare_output(output_dir)
    windows = load_gate_windows(annotation_path)
    cp08_range = TimeRange(start_sec=windows.cp08[0], end_sec=windows.cp08[1])
    tail_range = TimeRange(
        start_sec=windows.cp09_tail[0], end_sec=windows.cp09_tail[1]
    )
    objects: dict[str, list[dict[str, Any]]] = {
        object_id: [] for object_id in PROMPT_CANDIDATES
    }
    for object_id in ("blunt_instrument", "sharp_probe"):
        for prompt_text in PROMPT_CANDIDATES[object_id]:
            sparse = _run_prompt(
                segmenter=segmenter,
                video_path=video_path,
                time_range=cp08_range,
                object_id=object_id,
                prompt_text=prompt_text,
                sample_fps=CP08_SPARSE_FPS,
                phase="sparse",
                output_dir=output_dir,
                read_frame_fn=read_frame_fn,
                sample_frames_fn=sample_frames_fn,
            )
            run: dict[str, Any] = {"prompt": prompt_text, "sparse": sparse}
            candidate_time = sparse["first_candidate_time_sec"]
            if candidate_time is not None:
                dense_range = TimeRange(
                    start_sec=max(
                        cp08_range.start_sec,
                        float(candidate_time) - CP08_DENSE_HALF_WINDOW_SEC,
                    ),
                    end_sec=min(
                        cp08_range.end_sec,
                        float(candidate_time) + CP08_DENSE_HALF_WINDOW_SEC,
                    ),
                )
                run["dense"] = _run_prompt(
                    segmenter=segmenter,
                    video_path=video_path,
                    time_range=dense_range,
                    object_id=object_id,
                    prompt_text=prompt_text,
                    sample_fps=CP08_DENSE_FPS,
                    phase="dense",
                    output_dir=output_dir,
                    read_frame_fn=read_frame_fn,
                    sample_frames_fn=sample_frames_fn,
                )
            objects[object_id].append(run)

    for object_id in ("target_tooth", "rubber_dam_clamp", "rubber_dam"):
        for prompt_text in PROMPT_CANDIDATES[object_id]:
            tail = _run_prompt(
                segmenter=segmenter,
                video_path=video_path,
                time_range=tail_range,
                object_id=object_id,
                prompt_text=prompt_text,
                sample_fps=CP09_TAIL_FPS,
                phase="tail",
                output_dir=output_dir,
                read_frame_fn=read_frame_fn,
                sample_frames_fn=sample_frames_fn,
            )
            objects[object_id].append({"prompt": prompt_text, "tail": tail})

    selected = {
        object_id: select_prompt([_selection_metrics(run) for run in runs])
        for object_id, runs in objects.items()
    }
    selected_payload = {
        "session_strategy": "independent",
        "objects": selected,
    }
    required = ("blunt_instrument", "target_tooth", "rubber_dam_clamp", "rubber_dam")
    automatic_gate_passed = all(
        selected[object_id]["automatic_gate_passed"] for object_id in required
    )
    summary = {
        "status": "completed",
        "video_path": str(video_path),
        "annotation_path": str(annotation_path),
        "model_version": segmenter.model_version,
        "windows": {
            "cp08": cp08_range.model_dump(mode="json"),
            "cp09_tail": tail_range.model_dump(mode="json"),
        },
        "objects": objects,
        "automatic_gate_passed": automatic_gate_passed,
    }
    manual_review = {
        "overall_status": "pending",
        "items": [
            {"object": object_id, "status": "pending", "note_zh": ""}
            for object_id in PROMPT_CANDIDATES
        ],
    }
    atomic_write_json(output_dir / "selected_prompts.json", selected_payload)
    atomic_write_json(output_dir / "manual_review.json", manual_review)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
