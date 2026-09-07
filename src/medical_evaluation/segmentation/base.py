from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from medical_evaluation.domain import TimeRange


class SegmentationPrompt(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    object_id: str = Field(min_length=1)
    kind: Literal["point", "box", "mask", "text"]
    frame_time_sec: float = Field(default=0, ge=0)
    coordinates: list[float] | None = None
    positive: bool = True
    text: str | None = None
    mask: np.ndarray | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> SegmentationPrompt:
        if self.kind == "point":
            if self.coordinates is None or len(self.coordinates) != 2:
                raise ValueError("point prompt requires x, y coordinates")
            _validate_normalized(self.coordinates)
        elif self.kind == "box":
            if self.coordinates is None or len(self.coordinates) != 4:
                raise ValueError("box prompt requires x1, y1, x2, y2 coordinates")
            _validate_normalized(self.coordinates)
            x1, y1, x2, y2 = self.coordinates
            if x2 <= x1 or y2 <= y1:
                raise ValueError("box corners must be ordered")
        elif self.kind == "text":
            if not self.text or not self.text.strip():
                raise ValueError("text prompt requires non-empty text")
        elif self.mask is None or self.mask.ndim != 2:
            raise ValueError("mask prompt requires a two-dimensional mask")
        return self


class FrameMasks(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_index: int = Field(ge=0)
    frame_time_sec: float = Field(ge=0)
    masks: dict[str, np.ndarray]
    scores: dict[str, float] = Field(default_factory=dict)
    score_sources: dict[str, str] = Field(default_factory=dict)
    sample_position: int | None = Field(default=None, ge=0)


class VideoSegmenter(Protocol):
    @property
    def model_version(self) -> str: ...

    def track(
        self,
        video_path: Path,
        time_range: TimeRange,
        prompts: list[SegmentationPrompt],
        sample_fps: float,
    ) -> Iterator[FrameMasks]: ...


def normalize_masks(
    raw: dict[str, object],
    *,
    threshold: float,
    frame_time_sec: float,
) -> FrameMasks:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    frame_index = int(raw["frame"])
    objects = raw.get("objects")
    if not isinstance(objects, dict):
        raise TypeError("raw masks must contain an objects mapping")
    masks = {
        str(object_id): _boolean_mask(mask, threshold)
        for object_id, mask in objects.items()
    }
    return FrameMasks(frame_index=frame_index, frame_time_sec=frame_time_sec, masks=masks)


def normalized_to_pixels(coordinates: list[float], width: int, height: int) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float32).copy()
    values[0::2] *= width
    values[1::2] *= height
    return values


def checkpoint_digest(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def _validate_normalized(coordinates: list[float]) -> None:
    if any(not 0 <= value <= 1 for value in coordinates):
        raise ValueError("prompt coordinates must be normalized between zero and one")


def _to_numpy(value: object) -> np.ndarray:
    detached = value.detach() if hasattr(value, "detach") else value
    cpu_value = detached.cpu() if hasattr(detached, "cpu") else detached
    return np.asarray(cpu_value)


def _boolean_mask(value: object, threshold: float) -> np.ndarray:
    array = _to_numpy(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("each object mask must resolve to two dimensions")
    return array.astype(np.float32) > threshold
