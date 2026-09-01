from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from itertools import product
from pathlib import Path

import numpy as np

from medical_evaluation.annotations import AnnotationStore
from medical_evaluation.features.marks import (
    cluster_stable_marks,
    detect_dark_mark_observations,
    select_punch_candidate,
)
from medical_evaluation.segmentation.prompts import prompts_for_object
from medical_evaluation.segmentation.sam2_backend import Sam2Backend
from medical_evaluation.storage import atomic_write_json
from medical_evaluation.video import read_frame
from scripts.smoke_sam2_cp01_cp11 import VIDEO_FILENAMES

CalibrationSample = tuple[np.ndarray, np.ndarray, int, float]


def sweep_cp01_detector(
    samples: Sequence[CalibrationSample],
    *,
    maximum_black_values: Sequence[int],
    maximum_faint_values: Sequence[int],
    maximum_faint_saturations: Sequence[int],
    minimum_area_ratios: Sequence[float],
    minimum_observed_frames: int,
    maximum_area_ratio: float = 0.01,
    maximum_local_distance: float = 0.04,
    corner_margin: float = 0.15,
) -> list[dict[str, object]]:
    """Return threshold combinations yielding exactly two selectable stable marks."""

    results: list[dict[str, object]] = []
    combinations = product(
        maximum_black_values,
        maximum_faint_values,
        maximum_faint_saturations,
        minimum_area_ratios,
    )
    for black_value, faint_value, faint_saturation, minimum_area in combinations:
        if black_value > faint_value:
            continue
        observations = []
        for frame, dam_mask, frame_index, time_sec in samples:
            observations.extend(
                detect_dark_mark_observations(
                    frame,
                    dam_mask,
                    frame_index=frame_index,
                    time_sec=time_sec,
                    min_area_ratio=minimum_area,
                    max_area_ratio=maximum_area_ratio,
                    maximum_black_value=black_value,
                    maximum_faint_value=faint_value,
                    maximum_faint_saturation=faint_saturation,
                )
            )
        candidates = cluster_stable_marks(
            observations,
            min_observed_frames=minimum_observed_frames,
            max_local_distance=maximum_local_distance,
        )
        selection = select_punch_candidate(candidates, corner_margin=corner_margin)
        if len(candidates) != 2 or selection.status != "selected":
            continue
        results.append(
            {
                "maximum_black_value": black_value,
                "maximum_faint_value": faint_value,
                "maximum_faint_saturation": faint_saturation,
                "minimum_area_ratio": minimum_area,
                "candidate_count": len(candidates),
                "selection_status": selection.status,
                "selection_reason": selection.reason,
                "punch_u": selection.punch.u if selection.punch else None,
                "punch_v": selection.punch.v if selection.punch else None,
                "candidates": [
                    {
                        "u": item.u,
                        "v": item.v,
                        "observed_frame_count": item.observed_frame_count,
                    }
                    for item in candidates
                ],
            }
        )
    return results


def _integers(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep CP01 dark-mark thresholds after one SAM2 tracking pass."
    )
    parser.add_argument("--video-id", choices=tuple(VIDEO_FILENAMES), default="success")
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-fps", default=2.0, type=float)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--maximum-black-values", default="40,55,70")
    parser.add_argument("--maximum-faint-values", default="120,140,160,180")
    parser.add_argument("--maximum-faint-saturations", default="60,90,120,150")
    parser.add_argument("--minimum-area-ratios", default="0.00005,0.0001,0.0002,0.0005")
    parser.add_argument("--minimum-observed-frames", default=3, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    annotations = AnnotationStore(data_dir / "annotations").load_segments(args.video_id)
    segment = next(item for item in annotations.steps if item.checkpoint_id == "cp_01")
    prompts = prompts_for_object(
        annotations,
        "rubber_dam",
        segment.time_range,
        checkpoint_id="cp_01",
    )
    backend = Sam2Backend(
        args.model_config,
        args.checkpoint_path.resolve(),
        device=args.device,
    )
    video_path = args.videos_dir.resolve() / VIDEO_FILENAMES[args.video_id]
    samples: list[CalibrationSample] = []
    for item in backend.track(
        video_path,
        segment.time_range,
        prompts,
        sample_fps=args.sample_fps,
    ):
        dam_mask = item.masks.get("rubber_dam")
        if dam_mask is None or not np.asarray(dam_mask, dtype=bool).any():
            continue
        samples.append(
            (
                read_frame(video_path, item.frame_index),
                np.asarray(dam_mask, dtype=bool),
                item.frame_index,
                item.frame_time_sec,
            )
        )

    results = sweep_cp01_detector(
        samples,
        maximum_black_values=_integers(args.maximum_black_values),
        maximum_faint_values=_integers(args.maximum_faint_values),
        maximum_faint_saturations=_integers(args.maximum_faint_saturations),
        minimum_area_ratios=_floats(args.minimum_area_ratios),
        minimum_observed_frames=args.minimum_observed_frames,
    )
    output = args.output or data_dir / "calibration/cp01_detector_sweep.json"
    atomic_write_json(
        output,
        {
            "video_id": args.video_id,
            "sample_count": len(samples),
            "valid_combination_count": len(results),
            "valid_combinations": results,
        },
    )
    print(f"samples: {len(samples)}")
    print(f"valid combinations: {len(results)}")
    print(f"output: {output}")
    for item in results[:20]:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
