from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import (
    FrameMasks,
    SegmentationPrompt,
    checkpoint_digest,
    normalize_masks,
    normalized_to_pixels,
)
from medical_evaluation.video import probe_video


class Sam2Backend:
    def __init__(
        self,
        config_name: str,
        checkpoint: Path,
        *,
        device: str = "cuda",
        predictor: Any | None = None,
    ) -> None:
        self.config_name = config_name
        self.checkpoint = checkpoint
        self.device = device
        if predictor is None:
            try:
                from sam2.build_sam import build_sam2_video_predictor
            except ImportError as exc:
                raise RuntimeError(
                    "SAM2 is not installed; install Meta's official sam2 package on the GPU server"
                ) from exc
            predictor = build_sam2_video_predictor(
                config_file=config_name,
                ckpt_path=str(checkpoint),
                device=device,
            )
        self.predictor = predictor

    @property
    def model_version(self) -> str:
        return f"sam2.1:{self.config_name}:{checkpoint_digest(self.checkpoint)}"

    def track(
        self,
        video_path: Path,
        time_range: TimeRange,
        prompts: list[SegmentationPrompt],
        sample_fps: float,
    ) -> Iterator[FrameMasks]:
        if any(prompt.kind == "text" for prompt in prompts):
            raise ValueError("SAM2 does not support text-only prompts; use point/box/mask or SAM3")
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        metadata = probe_video(video_path)
        state = self.predictor.init_state(video_path=str(video_path))
        object_numbers = {
            object_id: number
            for number, object_id in enumerate(dict.fromkeys(p.object_id for p in prompts), start=1)
        }
        reverse_ids = {number: object_id for object_id, number in object_numbers.items()}
        for prompt in prompts:
            frame_index = round(prompt.frame_time_sec * metadata.fps)
            if prompt.kind == "mask":
                self.predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=frame_index,
                    obj_id=object_numbers[prompt.object_id],
                    mask=prompt.mask,
                )
                continue
            assert prompt.coordinates is not None
            pixels = normalized_to_pixels(prompt.coordinates, metadata.width, metadata.height)
            arguments: dict[str, object] = {
                "inference_state": state,
                "frame_idx": frame_index,
                "obj_id": object_numbers[prompt.object_id],
            }
            if prompt.kind == "point":
                arguments.update(
                    points=pixels.reshape(1, 2),
                    labels=np.asarray([int(prompt.positive)], dtype=np.int32),
                )
            else:
                arguments["box"] = pixels
            self.predictor.add_new_points_or_box(**arguments)

        stride = max(1, round(metadata.fps / sample_fps))
        try:
            for frame_index, object_ids, logits in self.predictor.propagate_in_video(state):
                frame_time = frame_index / metadata.fps
                if not time_range.start_sec <= frame_time <= time_range.end_sec:
                    continue
                if frame_index % stride:
                    continue
                raw = {
                    "frame": frame_index,
                    "objects": {
                        reverse_ids[int(object_id)]: mask
                        for object_id, mask in zip(object_ids, logits, strict=True)
                    },
                }
                yield normalize_masks(raw, threshold=0, frame_time_sec=frame_time)
        finally:
            if hasattr(self.predictor, "reset_state"):
                self.predictor.reset_state(state)
