from __future__ import annotations

import argparse
import math
from pathlib import Path

from medical_evaluation.annotations import (
    AnnotationStore,
    BoxPrompt,
    PointPrompt,
    VideoAnnotations,
)
from medical_evaluation.domain import TimeRange
from medical_evaluation.storage import atomic_write_json


def calibrate_cp01_reference(
    annotations: VideoAnnotations,
    time_range: TimeRange,
) -> dict[str, float]:
    prompts = [
        item
        for item in annotations.prompts
        if time_range.start_sec <= item.frame_time_sec <= time_range.end_sec
    ]
    dam_boxes = [
        item
        for item in prompts
        if item.object_id == "rubber_dam" and isinstance(item, BoxPrompt)
    ]
    references = [
        item
        for item in prompts
        if item.object_id == "cp01_reference" and isinstance(item, PointPrompt)
    ]
    if len(dam_boxes) != 1:
        raise ValueError("cp_01 calibration requires exactly one rubber_dam box")
    if len(references) != 1:
        raise ValueError("cp_01 calibration requires exactly one cp01_reference point")
    dam_box = dam_boxes[0]
    reference = references[0]
    if not reference.positive:
        raise ValueError("cp01_reference must be a positive point")
    if not math.isclose(
        dam_box.frame_time_sec,
        reference.frame_time_sec,
        abs_tol=1e-3,
    ):
        raise ValueError("rubber_dam and cp01_reference must share one frame")
    if not (
        dam_box.x1 <= reference.x <= dam_box.x2
        and dam_box.y1 <= reference.y <= dam_box.y2
    ):
        raise ValueError("cp01_reference must be inside rubber_dam box")
    return {
        "reference_u": (reference.x - dam_box.x1) / (dam_box.x2 - dam_box.x1),
        "reference_v": (reference.y - dam_box.y1) / (dam_box.y2 - dam_box.y1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save the one-time CP01 36-tooth reference in dam-local coordinates."
    )
    parser.add_argument("--video-id", default="success")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    annotations = AnnotationStore(data_dir / "annotations").load_segments(args.video_id)
    segment = next(item for item in annotations.steps if item.checkpoint_id == "cp_01")
    values = calibrate_cp01_reference(annotations, segment.time_range)
    output = args.output or data_dir / "calibration/cp01_reference.json"
    atomic_write_json(
        output,
        {
            "checkpoint_id": "cp_01",
            "source_video_id": annotations.video_id,
            **values,
        },
    )
    print(output)


if __name__ == "__main__":
    main()
