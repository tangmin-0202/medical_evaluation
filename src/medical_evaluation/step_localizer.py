from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field

AnchorName = Literal[
    "punch_contact",
    "clamp_appears",
    "mouth_entry",
    "floss_appears",
    "frame_expansion",
]


class EventAnchor(BaseModel):
    name: AnchorName
    time_sec: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class AnchorEvidence(EventAnchor):
    accepted: bool
    reason: str


class LocalizedSegment(BaseModel):
    step_index: int = Field(ge=0)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    raw_start_sec: float = Field(ge=0)
    raw_end_sec: float = Field(ge=0)
    missing: bool = False
    mean_cost: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    anchor_evidence: list[AnchorEvidence] = Field(default_factory=list)


class LocalizationResult(BaseModel):
    segments: list[LocalizedSegment]
    state_path: list[int]
    total_cost: float = Field(ge=0)


class EmbeddingEncoder(Protocol):
    @property
    def model_version(self) -> str: ...

    def encode(self, frames: Sequence[np.ndarray]) -> np.ndarray: ...


class OpenClipEncoder:
    """Optional, lazily imported OpenCLIP frame encoder."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model: object | None = None
        self._preprocess: object | None = None

    @property
    def model_version(self) -> str:
        return f"open_clip:{self.model_name}:{self.pretrained}"

    def encode(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        if self._model is None:
            self._load()
        import torch
        from PIL import Image

        assert self._preprocess is not None
        tensors = [self._preprocess(Image.fromarray(frame[..., ::-1])) for frame in frames]
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            encoded = self._model.encode_image(batch)  # type: ignore[union-attr]
        return encoded.float().cpu().numpy()

    def _load(self) -> None:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess


def cosine_cost(sequence: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    sequence = _two_dimensional(sequence, "sequence")
    prototypes = _two_dimensional(prototypes, "prototypes")
    if sequence.shape[1] != prototypes.shape[1]:
        raise ValueError("sequence and prototype dimensions must match")
    normalized_sequence = sequence / np.clip(
        np.linalg.norm(sequence, axis=1, keepdims=True),
        1e-8,
        None,
    )
    normalized_prototypes = prototypes / np.clip(
        np.linalg.norm(prototypes, axis=1, keepdims=True),
        1e-8,
        None,
    )
    return np.clip(1.0 - normalized_sequence @ normalized_prototypes.T, 0, 2)


def align_ordered_steps(
    sequence: np.ndarray,
    prototypes: np.ndarray,
    frame_times: np.ndarray,
    *,
    skip_penalty: float = 0.5,
    confidence_scale: float = 1.0,
) -> LocalizationResult:
    if skip_penalty < 0:
        raise ValueError("skip_penalty must be non-negative")
    if confidence_scale <= 0:
        raise ValueError("confidence_scale must be positive")
    costs = cosine_cost(sequence, prototypes)
    times = np.asarray(frame_times, dtype=np.float64)
    if times.ndim != 1 or len(times) != len(sequence) or len(times) == 0:
        raise ValueError("frame_times must contain one value per sequence frame")
    if np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError("frame_times must be non-negative and strictly increasing")

    frame_count, step_count = costs.shape
    dynamic = np.full((frame_count, step_count), np.inf, dtype=np.float64)
    previous = np.full((frame_count, step_count), -1, dtype=np.int32)
    dynamic[0] = costs[0] + skip_penalty * np.arange(step_count)

    for frame_index in range(1, frame_count):
        for step_index in range(step_count):
            prior_steps = np.arange(step_index + 1)
            skipped = np.maximum(step_index - prior_steps - 1, 0)
            candidates = dynamic[frame_index - 1, : step_index + 1] + skip_penalty * skipped
            best_prior = int(np.argmin(candidates))
            dynamic[frame_index, step_index] = candidates[best_prior] + costs[
                frame_index, step_index
            ]
            previous[frame_index, step_index] = best_prior

    trailing = skip_penalty * np.arange(step_count - 1, -1, -1)
    final_step = int(np.argmin(dynamic[-1] + trailing))
    total_cost = float(dynamic[-1, final_step] + trailing[final_step])
    state_path = [final_step]
    for frame_index in range(frame_count - 1, 0, -1):
        state_path.append(int(previous[frame_index, state_path[-1]]))
    state_path.reverse()

    interval = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    segments: list[LocalizedSegment] = []
    path_array = np.asarray(state_path)
    for step_index in range(step_count):
        assigned = np.flatnonzero(path_array == step_index)
        if assigned.size:
            start = float(times[assigned[0]])
            end = float(times[assigned[-1]] + interval)
            mean_cost = float(np.mean(costs[assigned, step_index]))
            confidence = float(np.clip(1 - mean_cost / confidence_scale, 0, 1))
            missing = False
        else:
            later = np.flatnonzero(path_array > step_index)
            boundary_index = int(later[0]) if later.size else len(times) - 1
            start = end = float(times[boundary_index])
            mean_cost = float(confidence_scale)
            confidence = 0.0
            missing = True
        segments.append(
            LocalizedSegment(
                step_index=step_index,
                start_sec=start,
                end_sec=end,
                raw_start_sec=start,
                raw_end_sec=end,
                missing=missing,
                mean_cost=mean_cost,
                confidence=confidence,
            )
        )
    return LocalizationResult(segments=segments, state_path=state_path, total_cost=total_cost)


def refine_with_anchors(
    segment: LocalizedSegment,
    anchors: Sequence[EventAnchor],
    *,
    minimum_confidence: float = 0.5,
) -> LocalizedSegment:
    start = segment.start_sec
    end = segment.end_sec
    evidence = list(segment.anchor_evidence)
    for anchor in anchors:
        reason = "accepted"
        accepted = True
        if anchor.confidence < minimum_confidence:
            accepted = False
            reason = "confidence below threshold"
        elif not segment.raw_start_sec <= anchor.time_sec <= segment.raw_end_sec:
            accepted = False
            reason = "anchor outside raw segment"
        elif anchor.name == "punch_contact" and anchor.time_sec <= start:
            accepted = False
            reason = "end anchor must follow start"
        elif anchor.name != "punch_contact" and anchor.time_sec >= end:
            accepted = False
            reason = "start anchor must precede end"
        elif anchor.name == "punch_contact":
            end = anchor.time_sec
        else:
            start = anchor.time_sec
        evidence.append(
            AnchorEvidence(
                **anchor.model_dump(),
                accepted=accepted,
                reason=reason,
            )
        )
    return segment.model_copy(
        update={"start_sec": start, "end_sec": end, "anchor_evidence": evidence}
    )


def localized_segment(
    *,
    start_sec: float,
    end_sec: float,
    step_index: int = 0,
) -> LocalizedSegment:
    return LocalizedSegment(
        step_index=step_index,
        start_sec=start_sec,
        end_sec=end_sec,
        raw_start_sec=start_sec,
        raw_end_sec=end_sec,
        mean_cost=0,
        confidence=1,
    )


def event_anchor(name: AnchorName, time_sec: float, *, confidence: float) -> EventAnchor:
    return EventAnchor(name=name, time_sec=time_sec, confidence=confidence)


def _two_dimensional(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    return array
