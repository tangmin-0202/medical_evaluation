from __future__ import annotations

import inspect
from collections.abc import Iterator
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
)
from medical_evaluation.video import write_sampled_frame_sequence


class Sam3AmbiguousTextResult(RuntimeError):
    """Raised when a text prompt returns candidates without ranking information."""


class Sam3Backend:
    def __init__(
        self,
        checkpoint: Path,
        *,
        bpe_path: Path | None = None,
        device: str = "cuda:0",
        output_prob_threshold: float = 0.5,
        grounding_batch_size: int = 4,
        predictor: Any | None = None,
    ) -> None:
        if not 0 <= output_prob_threshold <= 1:
            raise ValueError("output_prob_threshold must be between zero and one")
        if grounding_batch_size <= 0:
            raise ValueError("grounding_batch_size must be positive")
        self.checkpoint = checkpoint
        self.bpe_path = bpe_path
        self.device = device
        self.output_prob_threshold = output_prob_threshold
        self.grounding_batch_size = grounding_batch_size
        if predictor is None:
            try:
                import torch
                from sam3.model_builder import build_sam3_multiplex_video_predictor
            except ImportError as exc:
                raise RuntimeError(
                    "SAM3 is not installed; install Meta's official sam3 package on the GPU server"
                ) from exc

            torch.cuda.set_device(device)
            predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=str(checkpoint),
                bpe_path=str(bpe_path) if bpe_path else None,
                use_fa3=False,
                use_rope_real=False,
                compile=False,
                warm_up=False,
                async_loading_frames=False,
            )
        self.predictor = predictor
        _limit_grounding_batch_size(self.predictor, grounding_batch_size)
        _filter_unsupported_init_state_kwargs(self.predictor)

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
        if len(prompts) != 1 or prompts[0].kind != "text" or not prompts[0].text:
            raise ValueError("SAM3 text mode requires exactly one text prompt")
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")

        prompt = prompts[0]
        with TemporaryDirectory(prefix="medical-evaluation-sam3-") as temporary:
            sequence = write_sampled_frame_sequence(
                video_path,
                Path(temporary) / "frames",
                time_range=time_range,
                sample_fps=sample_fps,
                required_times_sec=[prompt.frame_time_sec],
            )
            response = self.predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(sequence.directory),
                    "offload_video_to_cpu": True,
                }
            )
            session_id = str(response["session_id"])
            try:
                selected_id: int | None = None
                selected_prompt_score: float | None = None
                prompt_local_index = 0
                for candidate_index in range(len(sequence.entries)):
                    prompt_result = self.predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": candidate_index,
                            "text": prompt.text,
                            "output_prob_thresh": self.output_prob_threshold,
                        }
                    )
                    selected_id = _select_candidate_id(prompt_result)
                    if selected_id is not None:
                        selected_prompt_score = _candidate_score(
                            prompt_result["outputs"], selected_id
                        )
                        prompt_local_index = candidate_index
                        break
                    self.predictor.handle_request(
                        {"type": "reset_session", "session_id": session_id}
                    )
                if selected_id is None:
                    return

                stream = self.predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "both",
                        "start_frame_index": prompt_local_index,
                        "output_prob_thresh": self.output_prob_threshold,
                    }
                )
                tracked: dict[int, FrameMasks] = {}
                for item in stream:
                    local_index = int(item["frame_index"])
                    if local_index < 0 or local_index >= len(sequence.entries):
                        raise ValueError(f"SAM3 returned invalid local frame index {local_index}")
                    entry = sequence.entries[local_index]
                    outputs = item["outputs"]
                    object_ids = _as_array(outputs["out_obj_ids"]).reshape(-1)
                    masks = list(outputs["out_binary_masks"])
                    selected_masks = [
                        mask
                        for object_id, mask in zip(object_ids, masks, strict=True)
                        if int(object_id) == selected_id
                    ]
                    selected_mask = (
                        selected_masks[0]
                        if selected_masks
                        else np.zeros(
                            (sequence.metadata.height, sequence.metadata.width),
                            dtype=bool,
                        )
                    )
                    raw = {
                        "frame": entry.source_frame_index,
                        "objects": {prompt.object_id: selected_mask},
                    }
                    frame_masks = normalize_masks(
                        raw,
                        threshold=self.output_prob_threshold,
                        frame_time_sec=entry.source_time_sec,
                    )
                    if not frame_masks.masks[prompt.object_id].any():
                        tracked[local_index] = frame_masks
                        continue
                    score = _candidate_score(outputs, selected_id)
                    source = "propagation"
                    if score is None:
                        # Official propagation output omits scores on some revisions.
                        score = selected_prompt_score
                        source = "discovery"
                    if score is not None:
                        frame_masks.scores[prompt.object_id] = score
                        frame_masks.score_sources[prompt.object_id] = source
                    tracked[local_index] = frame_masks
                for local_index in sorted(tracked):
                    yield tracked[local_index]
            finally:
                self.predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )


def _select_candidate_id(prompt_result: dict[str, object]) -> int | None:
    outputs = prompt_result.get("outputs")
    if not isinstance(outputs, dict):
        raise TypeError("SAM3 add_prompt response has no outputs mapping")
    object_ids = _as_array(outputs.get("out_obj_ids", [])).reshape(-1)
    if object_ids.size == 0:
        return None
    if object_ids.size == 1:
        return int(object_ids[0])
    if "out_scores" not in outputs:
        raise Sam3AmbiguousTextResult(
            "SAM3 text prompt returned multiple candidates without scores"
        )
    scores = _as_array(outputs["out_scores"]).reshape(-1)
    if scores.size != object_ids.size:
        raise ValueError("SAM3 candidate scores do not match object IDs")
    return int(object_ids[int(np.argmax(scores))])


def _candidate_score(outputs: object, object_id: int) -> float | None:
    if not isinstance(outputs, dict) or "out_scores" not in outputs:
        return None
    object_ids = _as_array(outputs.get("out_obj_ids", [])).reshape(-1)
    scores = _as_array(outputs["out_scores"]).reshape(-1)
    if scores.size != object_ids.size:
        raise ValueError("SAM3 candidate scores do not match object IDs")
    matches = np.flatnonzero(object_ids == object_id)
    if matches.size != 1:
        return None
    score = float(scores[int(matches[0])])
    return score if np.isfinite(score) else None


def _as_array(value: object) -> np.ndarray:
    detached = value.detach() if hasattr(value, "detach") else value
    cpu_value = detached.cpu() if hasattr(detached, "cpu") else detached
    return np.asarray(cpu_value)


def _filter_unsupported_init_state_kwargs(predictor: Any) -> None:
    """Work around facebookresearch/sam3#543 without editing vendored source."""
    model = getattr(predictor, "model", None)
    original = getattr(model, "init_state", None)
    if not callable(original) or getattr(original, "_medical_eval_filters_kwargs", False):
        return
    parameters = inspect.signature(original).parameters
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return
    accepted = set(parameters)

    def compatible_init_state(*args: object, **kwargs: object) -> object:
        filtered = {key: value for key, value in kwargs.items() if key in accepted}
        return original(*args, **filtered)

    compatible_init_state._medical_eval_filters_kwargs = True  # type: ignore[attr-defined]
    model.init_state = compatible_init_state


def _limit_grounding_batch_size(predictor: Any, batch_size: int) -> None:
    model = getattr(predictor, "model", None)
    if model is None:
        return
    if not hasattr(model, "batched_grounding_batch_size"):
        raise RuntimeError("SAM3 multiplex model has no grounding batch size setting")
    model.batched_grounding_batch_size = batch_size
