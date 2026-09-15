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
from medical_evaluation.presets import PRESETS
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
                    sample_frames_fn=sample_frames,
                    appearance_validator=lambda raw, mask, object_id=object_id: (
                        visible_appearance_is_plausible(object_id, raw, mask)
                    ),
                )
            except Sam3AmbiguousTextResult as exc:
                result = {"status": "prompt_failed", "error_type": type(exc).__name__,
                          "error_message": str(exc), "automatic_gate_passed": False,
                          "valid_frame_count": 0, "max_consecutive_valid_frames": 0}
            objects[object_id].append({"prompt": text, **result})
    payload = {
        "status": "completed", "checkpoint_id": checkpoint_id,
        "time_range": time_range.model_dump(mode="json"),
        "model_version": backend.model_version, "objects": objects,
        "visual_review_status": "pending", "session_strategy": "independent_text_only",
        "score": None,
    }
    atomic_write_json(output_dir / "summary.json", payload)
    return payload


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
