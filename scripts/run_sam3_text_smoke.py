from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import SegmentationPrompt
from medical_evaluation.segmentation.sam3_backend import Sam3Backend
from medical_evaluation.storage import atomic_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded SAM3 text-prompt smoke test")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--sample-fps", type=float, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bpe-path", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-prob-threshold", type=float, default=0.2)
    parser.add_argument("--grounding-batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    backend = Sam3Backend(
        args.checkpoint,
        bpe_path=args.bpe_path,
        device=args.device,
        output_prob_threshold=args.output_prob_threshold,
        grounding_batch_size=args.grounding_batch_size,
    )
    prompt = SegmentationPrompt(
        object_id=args.object_id,
        kind="text",
        frame_time_sec=args.start_sec,
        text=args.text,
    )
    started = time.perf_counter()
    frame_indices: list[int] = []
    frame_times_sec: list[float] = []
    nonempty_area_ratios: list[float] = []
    for frame in backend.track(
        args.video,
        TimeRange(start_sec=args.start_sec, end_sec=args.end_sec),
        [prompt],
        sample_fps=args.sample_fps,
    ):
        mask = np.asarray(frame.masks.get(args.object_id, np.zeros((1, 1), dtype=bool)))
        destination = args.output_dir / f"{frame.frame_index:08d}.png"
        if not cv2.imwrite(str(destination), mask.astype(np.uint8) * 255):
            raise OSError(f"could not write mask: {destination}")
        frame_indices.append(frame.frame_index)
        frame_times_sec.append(frame.frame_time_sec)
        nonempty_area_ratios.append(float(mask.mean()))

    atomic_write_json(
        args.output_dir / "summary.json",
        {
            "video": str(args.video),
            "object_id": args.object_id,
            "text": args.text,
            "time_range": {"start_sec": args.start_sec, "end_sec": args.end_sec},
            "sample_fps": args.sample_fps,
            "output_prob_threshold": args.output_prob_threshold,
            "grounding_batch_size": args.grounding_batch_size,
            "frame_indices": frame_indices,
            "frame_times_sec": frame_times_sec,
            "nonempty_area_ratios": nonempty_area_ratios,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
