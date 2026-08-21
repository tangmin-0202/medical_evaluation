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
)
from medical_evaluation.video import probe_video


class Sam3Backend:
    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str = "cuda",
        predictor: Any | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        if predictor is None:
            try:
                from sam3.model_builder import build_sam3_video_predictor
            except ImportError as exc:
                raise RuntimeError(
                    "SAM3 is not installed; install Meta's official sam3 package on the GPU server"
                ) from exc
            predictor = build_sam3_video_predictor(
                checkpoint_path=str(checkpoint),
                device=device,
            )
        self.predictor = predictor

    @property
    def model_version(self) -> str:
        return f"sam3.1:{checkpoint_digest(self.checkpoint)}"

    def track(
        self,
        video_path: Path,
        time_range: TimeRange,
        prompts: list[SegmentationPrompt],
        sample_fps: float,
    ) -> Iterator[FrameMasks]:
        if any(prompt.kind == "mask" for prompt in prompts):
            raise ValueError("SAM3 adapter accepts text, point, and box prompts; use SAM2 for masks")
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        metadata = probe_video(video_path)
        response = self.predictor.handle_request(
            {"type": "start_session", "resource_path": str(video_path)}
        )
        session_id = response["session_id"]
        object_numbers = {
            object_id: number
            for number, object_id in enumerate(dict.fromkeys(p.object_id for p in prompts), start=1)
        }
        reverse_ids = {number: object_id for object_id, number in object_numbers.items()}
        try:
            for prompt in prompts:
                request: dict[str, object] = {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": round(prompt.frame_time_sec * metadata.fps),
                    "obj_id": object_numbers[prompt.object_id],
                }
                if prompt.kind == "text":
                    request["text"] = prompt.text
                elif prompt.kind == "point":
                    request["points"] = np.asarray([prompt.coordinates], dtype=np.float32)
                    request["point_labels"] = np.asarray([int(prompt.positive)], dtype=np.int32)
                else:
                    assert prompt.coordinates is not None
                    x1, y1, x2, y2 = prompt.coordinates
                    request["bounding_boxes"] = np.asarray(
                        [[x1, y1, x2 - x1, y2 - y1]],
                        dtype=np.float32,
                    )
                    request["bounding_box_labels"] = np.asarray([1], dtype=np.int32)
                self.predictor.handle_request(request)

            stride = max(1, round(metadata.fps / sample_fps))
            stream = self.predictor.handle_stream_request(
                {"type": "propagate_in_video", "session_id": session_id}
            )
            for item in stream:
                frame_index = int(item["frame_index"])
                frame_time = frame_index / metadata.fps
                if not time_range.start_sec <= frame_time <= time_range.end_sec:
                    continue
                if frame_index % stride:
                    continue
                outputs = item["outputs"]
                raw = {
                    "frame": frame_index,
                    "objects": {
                        reverse_ids.get(int(object_id), f"detected_{int(object_id)}"): mask
                        for object_id, mask in zip(
                            outputs["out_obj_ids"],
                            outputs["out_binary_masks"],
                            strict=True,
                        )
                    },
                }
                yield normalize_masks(raw, threshold=0.5, frame_time_sec=frame_time)
        finally:
            self.predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
