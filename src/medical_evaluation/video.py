from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


class VideoMetadata(BaseModel):
    path: Path
    duration_sec: float = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class SampledFrame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    time_sec: float = Field(ge=0)
    frame_index: int = Field(ge=0)
    image_bgr: np.ndarray


def probe_video(path: Path) -> VideoMetadata:
    _validate_video_path(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("video has invalid metadata")
        return VideoMetadata(
            path=path.resolve(),
            duration_sec=frame_count / fps,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
    finally:
        capture.release()


def sample_frames(
    path: Path,
    *,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
) -> Iterator[SampledFrame]:
    metadata = probe_video(path)
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")
    if start_sec < 0 or end_sec <= start_sec or end_sec > metadata.duration_sec + 1e-6:
        raise ValueError("sample range is outside video duration")

    capture = cv2.VideoCapture(str(path))
    try:
        sample_interval = 1.0 / sample_fps
        time_sec = start_sec
        previous_frame_index = -1
        while time_sec < end_sec - 1e-9:
            frame_index = min(round(time_sec * metadata.fps), metadata.frame_count - 1)
            if frame_index != previous_frame_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, image = capture.read()
                if not success or image is None:
                    raise ValueError(f"failed to decode frame {frame_index}")
                yield SampledFrame(
                    time_sec=frame_index / metadata.fps,
                    frame_index=frame_index,
                    image_bgr=image.copy(),
                )
                previous_frame_index = frame_index
            time_sec += sample_interval
    finally:
        capture.release()


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    metadata = probe_video(path)
    if frame_index < 0 or frame_index >= metadata.frame_count:
        raise ValueError(f"frame index {frame_index} is outside video")
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, image = capture.read()
        if not success or image is None:
            raise ValueError(f"failed to decode frame {frame_index}")
        return image.copy()
    finally:
        capture.release()


def _validate_video_path(path: Path) -> None:
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported video extension: {path.suffix}")
    if not path.is_file():
        raise ValueError(f"video file does not exist: {path}")
