from pathlib import Path

import cv2
import numpy as np


def make_test_video(
    path: Path,
    *,
    fps: int,
    seconds: int,
    size: tuple[int, int],
) -> Path:
    width, height = size
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("test MP4 writer is unavailable")
    try:
        for frame_index in range(fps * seconds):
            value = frame_index % 255
            frame = np.full((height, width, 3), value, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path
