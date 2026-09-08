from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.features.mannequin_registration import (
    RegistrationLimits,
    estimate_similarity_registration,
    registration_is_continuous,
    transform_points,
)
from medical_evaluation.storage import atomic_write_json


def _slug(text: str) -> str:
    return "-".join(part for part in text.lower().replace("_", "-").split() if part)


def validate_template(payload: Mapping[str, object]) -> dict[str, object]:
    required = {"schema_version", "template_id", "head_prompt", "reference", "demo_only"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"template is missing: {', '.join(missing)}")
    if payload["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if payload["demo_only"] is not True:
        raise ValueError("registration gate template must declare demo_only=true")
    reference = payload["reference"]
    if not isinstance(reference, Mapping) or not {
        "video_id",
        "stage",
        "frame_index",
    }.issubset(reference):
        raise ValueError("reference must include video_id, stage, and frame_index")
    polygon = payload.get("nose_polygon_normalized")
    if not isinstance(polygon, Sequence) or isinstance(polygon, (str, bytes)):
        raise TypeError("nose_polygon_normalized is required")
    if len(polygon) < 3:
        raise ValueError("nose_polygon_normalized requires at least three points")
    normalized: list[list[float]] = []
    for point in polygon:
        if not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError("nose_polygon_normalized points must be [x, y]")
        x, y = float(point[0]), float(point[1])
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError("nose_polygon_normalized values must be within [0, 1]")
        normalized.append([x, y])
    return dict(payload) | {"nose_polygon_normalized": normalized}


def has_consecutive_acceptance(
    rows: Sequence[Mapping[str, object]], minimum: int = 3
) -> bool:
    run = 0
    previous_position: int | None = None
    for row in sorted(rows, key=lambda item: int(item["sample_position"])):
        position = int(row["sample_position"])
        adjacent = previous_position is None or position == previous_position + 1
        if row.get("accepted") is True and row.get("continuous") is True and adjacent:
            run += 1
        elif row.get("accepted") is True and row.get("continuous") is True:
            run = 1
        else:
            run = 0
        previous_position = position
        if run >= minimum:
            return True
    return False


def _image_path(root: Path, kind: str, frame_index: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = root / kind / f"{frame_index:08d}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing {kind} artifact for frame {frame_index} under {root}")


def _trial_root(source: Path, video_id: str, prompt: str) -> Path:
    return source / "overlays" / video_id / "head" / _slug(prompt)


def _read_pair(root: Path, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    image_path = _image_path(root, "raw", frame_index)
    mask_path = _image_path(root, "masks", frame_index)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f"could not read frame artifacts for {frame_index}")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"raw and mask dimensions differ for frame {frame_index}")
    return image, mask > 0


def _polygon_pixels(template: Mapping[str, object], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return np.asarray(
        [[float(x) * width, float(y) * height] for x, y in template["nose_polygon_normalized"]],
        dtype=np.float64,
    )


def _write_overlay(
    output_path: Path,
    image: np.ndarray,
    target_mask: np.ndarray,
    mapped_nose: np.ndarray,
    *,
    accepted: bool,
    reason: str | None,
    mask_iou: float,
    residual_px: float,
) -> None:
    canvas = image.copy()
    contours, _hierarchy = cv2.findContours(
        target_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 3)
    cv2.polylines(canvas, [np.rint(mapped_nose).astype(np.int32)], True, (0, 220, 255), 4)
    label = (
        f"accepted iou={mask_iou:.3f} residual={residual_px:.2f}px"
        if accepted
        else f"rejected {reason} iou={mask_iou:.3f}"
    )
    cv2.putText(canvas, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"could not write overlay: {output_path}")


def run_gate(
    source: Path,
    output: Path,
    template_payload: Mapping[str, object],
    *,
    minimum_consecutive: int = 3,
    limits: RegistrationLimits | None = None,
) -> int:
    if output.exists():
        return 2
    output.mkdir(parents=True)
    try:
        template = validate_template(template_payload)
        summary_path = source / "summary.json"
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        prompt = str(template["head_prompt"])
        candidates = source_summary["rows_by_candidate"]
        if prompt not in candidates:
            raise ValueError(f"head prompt is absent from source summary: {prompt}")
        reference = template["reference"]
        assert isinstance(reference, Mapping)
        reference_root = _trial_root(source, str(reference["video_id"]), prompt)
        reference_image, reference_mask = _read_pair(
            reference_root, int(reference["frame_index"])
        )
        nose_polygon = _polygon_pixels(template, reference_mask.shape)
        rows: list[dict[str, object]] = []
        previous_by_sample: dict[str, np.ndarray] = {}
        for source_row in candidates[prompt]:
            if source_row.get("valid") is not True:
                continue
            sample_key = str(source_row["sample_key"])
            video_id = str(source_row["video_id"])
            frame_index = int(source_row["frame_index"])
            target_root = _trial_root(source, video_id, prompt)
            target_image, target_mask = _read_pair(target_root, frame_index)
            result = estimate_similarity_registration(
                reference_image,
                reference_mask,
                target_image,
                target_mask,
                limits,
            )
            previous = previous_by_sample.get(sample_key)
            continuous = result.accepted and (
                previous is None
                or registration_is_continuous(
                    previous,
                    result.matrix,
                    image_shape=target_mask.shape,
                    max_angle_delta_deg=6.0,
                    max_scale_delta=0.10,
                    max_translation_diagonal_ratio=0.08,
                )
            )
            if result.accepted:
                previous_by_sample[sample_key] = result.matrix
            mapped_nose = transform_points(nose_polygon, result.matrix)
            overlay_path = (
                output
                / "overlays"
                / sample_key.replace(":", "-")
                / f"{frame_index:08d}.jpg"
            ).resolve()
            _write_overlay(
                overlay_path,
                target_image,
                target_mask,
                mapped_nose,
                accepted=result.accepted,
                reason=result.reason,
                mask_iou=result.mask_iou,
                residual_px=result.residual_px,
            )
            rows.append(
                {
                    "sample_key": sample_key,
                    "video_id": video_id,
                    "stage": source_row["stage"],
                    "frame_index": frame_index,
                    "sample_position": int(source_row["sample_position"]),
                    "source_time_sec": float(source_row["source_time_sec"]),
                    "accepted": result.accepted,
                    "continuous": continuous,
                    "reason": result.reason,
                    "matrix": result.matrix.tolist(),
                    "match_count": result.match_count,
                    "inlier_count": result.inlier_count,
                    "inlier_ratio": result.inlier_ratio,
                    "mask_iou": result.mask_iou,
                    "residual_px": result.residual_px,
                    "scale": result.scale,
                    "angle_deg": result.angle_deg,
                    "mapped_nose_polygon": mapped_nose.tolist(),
                    "overlay_path": str(overlay_path),
                }
            )
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["sample_key"])].append(row)
        samples = {
            key: {
                "accepted": has_consecutive_acceptance(value, minimum_consecutive),
                "frame_count": len(value),
                "accepted_frame_count": sum(row["accepted"] is True for row in value),
                "continuous_frame_count": sum(row["continuous"] is True for row in value),
            }
            for key, value in grouped.items()
        }
        accepted = bool(samples) and all(item["accepted"] for item in samples.values())
        atomic_write_json(
            output / "summary.json",
            {
                "accepted": accepted,
                "source_git_revision": source_summary.get("git_revision"),
                "source_sam3_revision": source_summary.get("sam3_source_revision"),
                "source_model_version": source_summary.get("model_version"),
                "template": template,
                "minimum_consecutive": minimum_consecutive,
                "samples": samples,
                "rows": rows,
            },
        )
        return 0 if accepted else 2
    except Exception as exc:  # noqa: BLE001 - failed gates must retain an audit record
        atomic_write_json(
            output / "summary.json",
            {
                "accepted": False,
                "failure": {"exception_type": type(exc).__name__, "message": str(exc)},
            },
        )
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map a versioned mannequin nose region using saved SAM3 head artifacts."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--minimum-consecutive", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(args.template.read_text(encoding="utf-8"))
    return run_gate(
        args.source,
        args.output,
        payload,
        minimum_consecutive=args.minimum_consecutive,
    )


if __name__ == "__main__":
    raise SystemExit(main())
