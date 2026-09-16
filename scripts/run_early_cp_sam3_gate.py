"""Text-only early-CP object feasibility experiment; never emits a score."""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.features.cp02_punch import (
    locate_held_punch_box,
    locate_moving_multihole_disk,
    measure_held_punch_mask,
)
from medical_evaluation.presets import PRESETS
from medical_evaluation.segmentation.base import SegmentationPrompt
from medical_evaluation.segmentation.sam3_backend import Sam3AmbiguousTextResult
from medical_evaluation.storage import atomic_write_json
from medical_evaluation.video import read_frame, sample_frames

# Reuse the audited evidence writer without changing the paused CP08 experiment.
_spec = importlib.util.spec_from_file_location(
    "early_gate_support", Path(__file__).with_name("run_cp08_sam3_gate.py")
)
_support = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _support
_spec.loader.exec_module(_support)

CATALOG = {
    "cp_03": {"rubber_dam": ("large green sheet with a small hole", "large green sheet")},
    "cp_01": {
        "template_board": (
            "white rectangular card with a black cross and rows of black dots",
            "white card under the green sheet",
        ),
        "rubber_dam": ("flat green rubber sheet", "large flat green rectangular sheet"),
        "marking_pen": (
            "thin black pen touching the green sheet",
            "black pen held in a gloved hand",
        ),
    },
    "cp_02": {
        "rubber_dam_punch": (
            "metal pliers held in a hand with a round disk containing several holes",
            "handheld metal pliers with a multi-hole wheel at the tip",
            "small round metal disk with several holes held in a gloved hand",
            "round silver disk with several dark holes",
        ),
        "cleaning_instrument": ("metal dental probe", "dental instrument held in a gloved hand"),
        "rubber_dam": ("large green sheet", "large flat green rectangular sheet"),
    },
}


def visible_appearance_is_plausible(
    object_id: str, raw_bgr: np.ndarray, mask: np.ndarray,
) -> bool:
    """Reject obvious semantic swaps while retaining every raw mask for audit."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or raw_bgr.shape[:2] != binary.shape or not binary.any():
        return False
    hsv = cv2.cvtColor(np.asarray(raw_bgr, dtype=np.uint8), cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1][binary]
    value = hsv[..., 2][binary]
    if object_id == "template_board":
        return bool(np.mean((saturation < 90) & (value > 145)) >= 0.35)
    if object_id == "marking_pen":
        return bool(np.mean(value < 105) >= 0.25)
    return True


def collect_gate(
    backend, video_path, checkpoint_id, time_range, output_dir, *, sample_fps=2,
    read_frame_fn=read_frame,
    sample_frames_fn=sample_frames,
):
    _support._prepare_output(output_dir)
    objects = {}
    for object_id, prompts in CATALOG[checkpoint_id].items():
        objects[object_id] = []
        for text in prompts:
            print(f"{checkpoint_id} {object_id}: {text}", flush=True)
            try:
                result = _support._run_prompt(
                    segmenter=backend, video_path=video_path, time_range=time_range,
                    object_id=object_id, prompt_text=text, sample_fps=sample_fps,
                    phase="scan", output_dir=output_dir, read_frame_fn=read_frame_fn,
                    sample_frames_fn=sample_frames_fn,
                    appearance_validator=lambda raw, mask, object_id=object_id: (
                        visible_appearance_is_plausible(object_id, raw, mask)
                    ),
                )
            except Sam3AmbiguousTextResult as exc:
                result = {"status": "prompt_failed", "error_type": type(exc).__name__,
                          "error_message": str(exc), "automatic_gate_passed": False,
                          "valid_frame_count": 0, "max_consecutive_valid_frames": 0}
            objects[object_id].append({"prompt": text, **result})
    if checkpoint_id == "cp_02":
        objects["rubber_dam_punch"].append(
            _run_automatic_punch_box(
                backend=backend,
                video_path=video_path,
                time_range=time_range,
                output_dir=output_dir,
                sample_fps=sample_fps,
                read_frame_fn=read_frame_fn,
                sample_frames_fn=sample_frames_fn,
            )
        )
    payload = {
        "status": "completed", "checkpoint_id": checkpoint_id,
        "time_range": time_range.model_dump(mode="json"),
        "model_version": backend.model_version, "objects": objects,
        "visual_review_status": "pending", "session_strategy": "independent_text_only",
        "score": None,
    }
    atomic_write_json(output_dir / "summary.json", payload)
    return payload


def _run_automatic_punch_box(
    *, backend, video_path, time_range, output_dir, sample_fps, read_frame_fn,
    sample_frames_fn,
):
    """Generate a whole-tool SAM3 box from a moving disk, without manual points."""
    prompt_text = "automatic held punch box"
    sampled = list(sample_frames_fn(
        video_path,
        start_sec=time_range.start_sec,
        end_sec=time_range.end_sec,
        sample_fps=sample_fps,
    ))
    locator = locate_moving_multihole_disk([item.image_bgr for item in sampled])
    if locator is None:
        return {
            "prompt": prompt_text,
            "prompt_kind": "box",
            "status": "locator_not_found",
            "automatic_gate_passed": False,
            "valid_frame_count": 0,
            "max_consecutive_valid_frames": 0,
            "frames": [],
            "locator": None,
            "tool_box": None,
            "accepted_frame_count": 0,
            "max_consecutive_accepted_frames": 0,
        }

    seed = sampled[locator.frame_position]
    height, width = seed.image_bgr.shape[:2]
    tool_box = locate_held_punch_box(
        [item.image_bgr for item in sampled], locator,
    )
    if tool_box is None:
        return {
            "prompt": prompt_text,
            "prompt_kind": "box",
            "status": "tool_box_not_found",
            "automatic_gate_passed": False,
            "valid_frame_count": 0,
            "max_consecutive_valid_frames": 0,
            "frames": [],
            "locator": {
                "frame_position": locator.frame_position,
                "frame_index": seed.frame_index,
                "time_sec": seed.time_sec,
                "x": locator.x,
                "y": locator.y,
                "radius": locator.radius,
                "hole_count": locator.hole_count,
                "motion_ratio": locator.motion_ratio,
            },
            "tool_box": None,
            "accepted_frame_count": 0,
            "max_consecutive_accepted_frames": 0,
        }

    coordinates = tool_box.normalized(width, height)
    prompt = SegmentationPrompt(
        object_id="rubber_dam_punch",
        kind="box",
        frame_time_sec=seed.time_sec,
        coordinates=coordinates,
    )
    base_artifact_dir = output_dir / "rubber_dam_punch" / _support._slug(prompt_text)
    artifact_dir = base_artifact_dir / "scan"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    locator_overlay = seed.image_bgr.copy()
    cv2.circle(
        locator_overlay, (round(locator.x), round(locator.y)), round(locator.radius),
        (0, 255, 255), 2,
    )
    cv2.rectangle(
        locator_overlay,
        (tool_box.x1, tool_box.y1),
        (tool_box.x2, tool_box.y2),
        (255, 0, 255),
        2,
    )
    cv2.putText(
        locator_overlay,
        (
            f"holes={locator.hole_count} motion={locator.motion_ratio:.3f} "
            f"elongation={tool_box.elongation:.2f}"
        ),
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
        cv2.LINE_AA,
    )
    locator_overlay_path = artifact_dir / "locator_overlay.jpg"
    if not cv2.imwrite(str(locator_overlay_path), locator_overlay):
        raise OSError(f"could not write artifact: {locator_overlay_path}")

    tracked = list(backend.track(video_path, time_range, [prompt], sample_fps))
    frames = []
    accepted_flags = []
    accepted_masks = []
    for item in tracked:
        raw = read_frame_fn(video_path, item.frame_index)
        mask = np.asarray(
            item.masks.get(
                "rubber_dam_punch", np.zeros(raw.shape[:2], dtype=bool),
            ),
            dtype=bool,
        )
        measurement = measure_held_punch_mask(raw, mask, locator, tool_box)
        record = _support._write_artifacts(
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            frame_index=item.frame_index,
            frame_time_sec=item.frame_time_sec,
            object_id="rubber_dam_punch",
            prompt_text=prompt_text,
            raw=raw,
            mask=mask,
        )
        record["semantic_features"] = {
            "accepted": measurement.accepted,
            "reason": measurement.reason,
            "anchor_distance_radii": measurement.anchor_distance_radii,
            "inside_box_ratio": measurement.inside_box_ratio,
            "green_ratio": measurement.green_ratio,
            "elongation": measurement.elongation,
        }
        frames.append(record)
        accepted_flags.append(measurement.accepted)
        accepted_masks.append(mask if measurement.accepted else np.zeros_like(mask))

    metrics = _support.summarize_masks(
        accepted_masks, min_area_px=_support.MIN_MASK_AREA_PX,
    )
    max_consecutive = _support._max_true_run(accepted_flags)
    accepted_count = sum(accepted_flags)
    metrics.update(
        valid_frame_count=accepted_count,
        max_consecutive_valid_frames=max_consecutive,
        accepted_frame_count=accepted_count,
        max_consecutive_accepted_frames=max_consecutive,
        automatic_gate_passed=(
            max_consecutive >= 3
            and metrics["median_dominant_component_ratio"] >= 0.60
        ),
    )
    return {
        "prompt": prompt_text,
        "prompt_kind": "box",
        "box_coordinates": coordinates,
        "locator": {
            "frame_position": locator.frame_position,
            "frame_index": seed.frame_index,
            "time_sec": seed.time_sec,
            "x": locator.x,
            "y": locator.y,
            "radius": locator.radius,
            "hole_count": locator.hole_count,
            "motion_ratio": locator.motion_ratio,
            "overlay_path": locator_overlay_path.relative_to(
                output_dir
            ).as_posix(),
        },
        "tool_box": {
            "frame_position": tool_box.frame_position,
            "x1": tool_box.x1,
            "y1": tool_box.y1,
            "x2": tool_box.x2,
            "y2": tool_box.y2,
            "elongation": tool_box.elongation,
            "motion_ratio": tool_box.motion_ratio,
        },
        "frames": frames,
        **metrics,
    }


def main(argv=None):
    parser = _support.build_parser()
    parser.description = "SAM3 CP01/CP02 object gate only; no automatic grading."
    parser.add_argument("--checkpoint-id", choices=tuple(CATALOG), required=True)
    parser.add_argument("--sample-fps", type=float, default=2)
    args = parser.parse_args(argv)
    if args.sample_fps <= 0:
        parser.error("sample-fps must be positive")
    _support._prepare_output(args.output_dir)
    started = time.perf_counter()
    provenance = {
        "git_head": _support.git_output("rev-parse", "HEAD"),
        "git_status": _support.git_output("status", "--short"),
        "gate_script_sha256": _support.sha256_file(Path(__file__)),
        "sam3_revision": _support.git_output("-C", str(args.sam3_root), "rev-parse", "HEAD"),
        "checkpoint_sha256": _support.sha256_file(args.checkpoint),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "threshold": args.threshold, "grounding_batch_size": args.grounding_batch_size,
    }
    try:
        annotations = VideoAnnotations.model_validate_json(
            (args.annotations / f"{args.video_id}.json").read_text(encoding="utf-8")
        )
        ranges = [s.time_range for s in annotations.steps if s.checkpoint_id == args.checkpoint_id]
        if len(ranges) != 1:
            raise ValueError("exactly one checkpoint time range is required")
        window = TimeRange.model_validate(ranges[0].model_dump())
        _support._reset_cuda_peak_memory()
        result = collect_gate(
            _support._build_backend(args), args.videos / PRESETS[args.video_id],
            args.checkpoint_id, window, args.output_dir, sample_fps=args.sample_fps,
        )
        result.update(provenance=provenance, elapsed_seconds=time.perf_counter() - started,
                      cuda_peak_memory=_support.cuda_peak_memory())
        atomic_write_json(args.output_dir / "summary.json", result)
        print(json.dumps({"status": "completed", "visual_review_status": "pending"}), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - retain auditable failed runs
        atomic_write_json(args.output_dir / "summary.json", {
            "status": "failed", "error_type": type(exc).__name__, "error_message": str(exc),
            "provenance": provenance, "elapsed_seconds": time.perf_counter() - started,
        })
        print(f"gate failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
