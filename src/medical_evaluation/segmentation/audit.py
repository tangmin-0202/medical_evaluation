from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import FrameMasks, SegmentationPrompt, VideoSegmenter
from medical_evaluation.storage import atomic_write_json


class AuditedVideoSegmenter:
    def __init__(
        self,
        delegate: VideoSegmenter,
        *,
        output_path: Path,
        static_metadata: dict[str, Any],
    ) -> None:
        self.delegate = delegate
        self.output_path = output_path
        self.static_metadata = dict(static_metadata)
        self.tracks: list[dict[str, Any]] = []

    @property
    def model_version(self) -> str:
        return self.delegate.model_version

    def track(
        self,
        video_path: Path,
        time_range: TimeRange,
        prompts: list[SegmentationPrompt],
        sample_fps: float,
    ) -> Iterator[FrameMasks]:
        started = time.perf_counter()
        frame_count = 0
        status = "completed"
        error_type: str | None = None
        try:
            for frame in self.delegate.track(
                video_path,
                time_range,
                prompts,
                sample_fps,
            ):
                frame_count += 1
                yield frame
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__
            raise
        finally:
            record = {
                "status": status,
                "error_type": error_type,
                "elapsed_seconds": time.perf_counter() - started,
                "frame_count": frame_count,
                "sample_fps": sample_fps,
                "time_range": {
                    "start_sec": time_range.start_sec,
                    "end_sec": time_range.end_sec,
                },
                "prompts": [
                    {
                        "kind": prompt.kind,
                        "object_id": prompt.object_id,
                        "frame_time_sec": prompt.frame_time_sec,
                        "text": prompt.text,
                    }
                    for prompt in prompts
                ],
                "prompt_texts": [prompt.text for prompt in prompts if prompt.text],
                "process_peak_allocated_mib": _cuda_peak_allocated_mib(self.delegate),
            }
            self.tracks.append(record)
            atomic_write_json(
                self.output_path,
                {**self.static_metadata, "tracks": self.tracks},
            )


def _cuda_peak_allocated_mib(delegate: VideoSegmenter) -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device = getattr(delegate, "device", None)
        return float(torch.cuda.max_memory_allocated(device) / 1024**2)
    except (ImportError, RuntimeError, ValueError):
        return None
