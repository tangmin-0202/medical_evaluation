from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.annotations import AnnotationStore
from medical_evaluation.features.marks import (
    MarkCandidate,
    cluster_stable_marks,
    detect_dark_mark_observations,
    select_darkest_nearest_reference,
)
from medical_evaluation.segmentation.prompts import prompts_for_object
from medical_evaluation.segmentation.sam2_backend import Sam2Backend
from medical_evaluation.storage import atomic_write_json
from medical_evaluation.video import read_frame
from scripts.smoke_sam2_cp01_cp11 import VIDEO_FILENAMES

CalibrationSample = tuple[np.ndarray, np.ndarray, np.ndarray, int, float]


def sweep_cp01_detector(
    samples: Sequence[CalibrationSample],
    *,
    maximum_black_values: Sequence[int],
    maximum_faint_values: Sequence[int],
    maximum_faint_saturations: Sequence[int],
    minimum_area_ratios: Sequence[float],
    minimum_observed_frames_values: Sequence[int],
    minimum_pen_presence_frames_values: Sequence[int],
    reference_u: float,
    reference_v: float,
    output_dir: Path,
    maximum_area_ratio: float = 0.01,
    maximum_local_distance: float = 0.04,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    combinations = product(
        maximum_black_values,
        maximum_faint_values,
        maximum_faint_saturations,
        minimum_area_ratios,
        minimum_observed_frames_values,
        minimum_pen_presence_frames_values,
    )
    for index, combination in enumerate(combinations):
        (
            black_value,
            faint_value,
            faint_saturation,
            minimum_area,
            minimum_observed_frames,
            minimum_pen_presence_frames,
        ) = combination
        if black_value > faint_value:
            continue
        pen_valid_frame_count = sum(bool(pen_mask.any()) for _, _, pen_mask, _, _ in samples)
        pen_presence_detected = pen_valid_frame_count >= minimum_pen_presence_frames
        observations = []
        for frame, dam_mask, _, frame_index, time_sec in samples:
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
        selection = select_darkest_nearest_reference(
            candidates if pen_presence_detected else [],
            reference_u=reference_u,
            reference_v=reference_v,
        )
        overlay_path = output_dir / f"threshold-{index:04d}.jpg"
        _write_overlay(
            samples[-1][0],
            samples[-1][1],
            selection.ranked_candidates,
            reference_u,
            reference_v,
            overlay_path,
        )
        results.append(
            {
                "maximum_black_value": black_value,
                "maximum_faint_value": faint_value,
                "maximum_faint_saturation": faint_saturation,
                "minimum_area_ratio": minimum_area,
                "minimum_observed_frames": minimum_observed_frames,
                "minimum_pen_presence_frames": minimum_pen_presence_frames,
                "pen_valid_frame_count": pen_valid_frame_count,
                "pen_presence_detected": pen_presence_detected,
                "mark_candidate_count": len(candidates) if pen_presence_detected else 0,
                "selection_reason": selection.reason,
                "punch_u": selection.punch.u if selection.punch else None,
                "punch_v": selection.punch.v if selection.punch else None,
                "overlay_path": str(overlay_path),
            }
        )
    return results


def _write_overlay(
    frame: np.ndarray,
    dam_mask: np.ndarray,
    candidates: Sequence[MarkCandidate],
    reference_u: float,
    reference_v: float,
    output_path: Path,
) -> None:
    canvas = frame.copy()
    y_values, x_values = np.where(dam_mask)
    x_min, x_max = int(x_values.min()), int(x_values.max()) + 1
    y_min, y_max = int(y_values.min()), int(y_values.max()) + 1

    def point(item: MarkCandidate) -> tuple[int, int]:
        return (
            round(x_min + item.u * (x_max - x_min)),
            round(y_min + item.v * (y_max - y_min)),
        )

    for item in candidates:
        cv2.circle(canvas, point(item), 6, (0, 0, 255), -1)
    reference = (
        round(x_min + reference_u * (x_max - x_min)),
        round(y_min + reference_v * (y_max - y_min)),
    )
    cv2.circle(canvas, reference, 6, (0, 255, 0), -1)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"could not write CP01 calibration overlay: {output_path}")


def _integers(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep pen-presence CP01 thresholds after one SAM2 pass."
    )
    parser.add_argument("--video-id", choices=tuple(VIDEO_FILENAMES), default="success")
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-fps", default=2.0, type=float)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--maximum-black-values", default="40,55,70")
    parser.add_argument("--maximum-faint-values", default="120,140,160")
    parser.add_argument("--maximum-faint-saturations", default="60,90,120")
    parser.add_argument("--minimum-area-ratios", default="0.00005,0.0001,0.0002")
    parser.add_argument("--minimum-observed-frames-values", default="2,3")
    parser.add_argument("--minimum-pen-presence-frames-values", default="2,3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    annotations = AnnotationStore(data_dir / "annotations").load_segments(args.video_id)
    segment = next(item for item in annotations.steps if item.checkpoint_id == "cp_01")
    prompts = [
        *prompts_for_object(
            annotations,
            "rubber_dam",
            segment.time_range,
            checkpoint_id="cp_01",
        ),
        *prompts_for_object(
            annotations,
            "marking_pen",
            segment.time_range,
            checkpoint_id="cp_01",
        ),
    ]
    reference = json.loads(
        (data_dir / "calibration/cp01_reference.json").read_text(encoding="utf-8")
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
        dam = np.asarray(item.masks.get("rubber_dam"), dtype=bool)
        if not dam.any():
            continue
        pen_mask = item.masks.get("marking_pen")
        pen = (
            np.asarray(pen_mask, dtype=bool)
            if pen_mask is not None
            else np.zeros_like(dam)
        )
        samples.append(
            (
                read_frame(video_path, item.frame_index),
                dam,
                pen,
                item.frame_index,
                item.frame_time_sec,
            )
        )

    overlay_dir = data_dir / "calibration/cp01_detector_sweep_overlays"
    results = sweep_cp01_detector(
        samples,
        maximum_black_values=_integers(args.maximum_black_values),
        maximum_faint_values=_integers(args.maximum_faint_values),
        maximum_faint_saturations=_integers(args.maximum_faint_saturations),
        minimum_area_ratios=_floats(args.minimum_area_ratios),
        minimum_observed_frames_values=_integers(
            args.minimum_observed_frames_values
        ),
        minimum_pen_presence_frames_values=_integers(
            args.minimum_pen_presence_frames_values
        ),
        reference_u=float(reference["reference_u"]),
        reference_v=float(reference["reference_v"]),
        output_dir=overlay_dir,
    )
    output = args.output or data_dir / "calibration/cp01_detector_sweep.json"
    atomic_write_json(
        output,
        {
            "video_id": args.video_id,
            "sample_count": len(samples),
            "combination_count": len(results),
            "combinations": results,
        },
    )
    print(f"samples: {len(samples)}")
    print(f"combinations: {len(results)}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
