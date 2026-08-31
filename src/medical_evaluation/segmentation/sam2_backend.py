from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
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
from medical_evaluation.video import (
    SampledFrameSequence,
    write_sampled_frame_sequence,
)


@dataclass
class _PromptGroup:
    object_id: str
    frame_index: int
    points: list[np.ndarray]
    labels: list[int]
    box: np.ndarray | None = None


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
        if not prompts:
            raise ValueError("SAM2 requires at least one point, box, or mask prompt")
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        required_times = [prompt.frame_time_sec for prompt in prompts]
        with (
            TemporaryDirectory(prefix="medical-evaluation-sam2-") as temporary_root,
            self._autocast_context(),
        ):
            sequence = write_sampled_frame_sequence(
                video_path,
                Path(temporary_root) / "frames",
                time_range=time_range,
                sample_fps=sample_fps,
                required_times_sec=required_times,
            )
            state = self.predictor.init_state(video_path=str(sequence.directory))
            try:
                yield from self._track_initialized(state, sequence, prompts)
            finally:
                if hasattr(self.predictor, "reset_state"):
                    self.predictor.reset_state(state)

    def _autocast_context(self) -> Any:
        if self.device.split(":", maxsplit=1)[0] != "cuda":
            return nullcontext()
        try:
            import torch
        except ImportError:
            return nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def _track_initialized(
        self,
        state: Any,
        sequence: SampledFrameSequence,
        prompts: list[SegmentationPrompt],
    ) -> Iterator[FrameMasks]:
        metadata = sequence.metadata
        object_numbers = {
            object_id: number
            for number, object_id in enumerate(dict.fromkeys(p.object_id for p in prompts), start=1)
        }
        reverse_ids = {number: object_id for object_id, number in object_numbers.items()}
        conditioning_frames: list[int] = []
        for prompt in (item for item in prompts if item.kind == "mask"):
            source_index = round(prompt.frame_time_sec * metadata.fps)
            local_index = sequence.source_to_local[source_index]
            conditioning_frames.append(local_index)
            self.predictor.add_new_mask(
                inference_state=state,
                frame_idx=local_index,
                obj_id=object_numbers[prompt.object_id],
                mask=prompt.mask,
            )
        for group in _group_prompts(prompts, sequence):
            conditioning_frames.append(group.frame_index)
            points = np.asarray(group.points, dtype=np.float32).reshape(-1, 2)
            labels = np.asarray(group.labels, dtype=np.int32)
            self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=group.frame_index,
                obj_id=object_numbers[group.object_id],
                points=points if len(points) else None,
                labels=labels if len(labels) else None,
                box=group.box,
                clear_old_points=True,
                normalize_coords=True,
            )

        results: dict[int, FrameMasks] = {}
        first_frame = 0
        last_frame = len(sequence.entries) - 1
        forward_start = min(conditioning_frames)
        reverse_start = max(conditioning_frames)
        directions: list[tuple[int, bool, int]] = []
        if forward_start <= last_frame:
            directions.append((forward_start, False, last_frame - forward_start + 1))
        if reverse_start >= first_frame:
            directions.append((reverse_start, True, reverse_start - first_frame + 1))
        for start_frame_idx, reverse, maximum_frames in directions:
            propagation = self.predictor.propagate_in_video(
                state,
                start_frame_idx=start_frame_idx,
                reverse=reverse,
                max_frame_num_to_track=maximum_frames,
            )
            for local_index, object_ids, logits in propagation:
                timeline = sequence.entries[local_index]
                raw = {
                    "frame": timeline.source_frame_index,
                    "objects": {
                        reverse_ids[int(object_id)]: mask
                        for object_id, mask in zip(object_ids, logits, strict=True)
                    },
                }
                results[timeline.source_frame_index] = normalize_masks(
                    raw,
                    threshold=0,
                    frame_time_sec=timeline.source_time_sec,
                )
        for frame_index in sorted(results):
            yield results[frame_index]


def _group_prompts(
    prompts: list[SegmentationPrompt],
    sequence: SampledFrameSequence,
) -> list[_PromptGroup]:
    metadata = sequence.metadata
    grouped: dict[tuple[str, int], _PromptGroup] = {}
    for prompt in prompts:
        if prompt.kind in {"mask", "text"}:
            continue
        source_index = round(prompt.frame_time_sec * metadata.fps)
        local_index = sequence.source_to_local[source_index]
        key = prompt.object_id, local_index
        group = grouped.setdefault(
            key,
            _PromptGroup(prompt.object_id, local_index, [], []),
        )
        assert prompt.coordinates is not None
        pixels = normalized_to_pixels(prompt.coordinates, metadata.width, metadata.height)
        if prompt.kind == "point":
            group.points.append(pixels)
            group.labels.append(int(prompt.positive))
        elif group.box is not None:
            raise ValueError(f"multiple boxes for {prompt.object_id} on frame {local_index}")
        else:
            group.box = pixels
    return [grouped[key] for key in sorted(grouped, key=lambda item: (item[1], item[0]))]
