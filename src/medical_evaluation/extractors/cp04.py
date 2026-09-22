"""CP04 extraction for a clamp displayed on a white-gloved palm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.domain import TimeRange
from medical_evaluation.features.cp04_clamp import (
    ClampDisplayMeasurement,
    ClampMatchMeasurement,
    compare_clamp_mask,
    extract_metal_candidates_from_glove,
    measure_display_candidate,
    normalize_clamp_mask,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.segmentation.base import FrameMasks, SegmentationPrompt, VideoSegmenter
from medical_evaluation.segmentation.sam3_backend import Sam3AmbiguousTextResult
from medical_evaluation.storage import atomic_write_json, safe_child
from medical_evaluation.video import read_frame

HAND_PROMPT = "open white gloved palm holding a small shiny metal clip"
OPENCV_CANDIDATE_SOURCE = "opencv:compact-metal-object-on-glove"
BOX_REFINEMENT_SOURCE = "sam3:opencv-local-box-refinement"


def _shape_evidence_consistent(
    similarities: list[float], min_similarity: float
) -> bool | None:
    if len(similarities) < 2:
        return None
    matching_count = sum(value >= min_similarity for value in similarities)
    return matching_count == 0 or matching_count >= 2


@dataclass(frozen=True)
class _ClampObservation:
    item: FrameMasks
    frame: np.ndarray
    hand_mask: np.ndarray | None
    clamp_mask: np.ndarray
    display: ClampDisplayMeasurement
    match: ClampMatchMeasurement | None
    prompt_text: str


class Cp04FeatureExtractor:
    sample_fps = 2.0
    maximum_evidence_frames = 3

    def __init__(
        self,
        *,
        segmenter: VideoSegmenter,
        evidence_root: Path,
        reference_dir: Path,
        min_similarity: float = 0.8,
        enable_local_box_refinement: bool = False,
    ) -> None:
        self.segmenter = segmenter
        self.evidence_root = evidence_root
        self.reference_dir = reference_dir
        self.min_similarity = min_similarity
        self.enable_local_box_refinement = enable_local_box_refinement

    @property
    def model_version(self) -> str:
        return f"{self.segmenter.model_version}+cp04-glove-opencv-box-v1+cp04-reference-v1"

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id != "cp_04":
            raise ValueError("Cp04FeatureExtractor is CP04-only")
        if dense_fps <= 0 or analysis_width <= 0:
            raise ValueError("sampling and analysis dimensions must be positive")

        hand_items = self._track_text(
            video_path,
            time_range,
            object_id="cp04_gloved_hand",
            prompt_text=HAND_PROMPT,
        )
        references = self._load_reference_masks()
        chosen = self._measure_hand_items(video_path, hand_items, references)
        if (
            self.enable_local_box_refinement
            and chosen
            and not self._has_reference_match(chosen)
        ):
            refined = self._refine_from_local_box(
                video_path, time_range, hand_items, chosen, references
            )
            if refined:
                chosen = refined
        observed_any = bool(chosen)

        comparable = [
            item
            for item in chosen
            if item.display.reliable and item.match is not None and item.match.reliable
        ]
        comparable.sort(key=self._quality_score, reverse=True)
        similarities = [
            float(item.match.similarity)
            for item in comparable
            if item.match is not None and item.match.similarity is not None
        ]
        matching_count = sum(value >= self.min_similarity for value in similarities)
        # Two independent matches tolerate occasional noisy OpenCV candidates.
        # Zero matches is a consistent mismatch; exactly one remains ambiguous.
        evidence_consistent = _shape_evidence_consistent(
            similarities, self.min_similarity
        )
        features: dict[str, float | bool | None] = {
            "clamp_observed": observed_any,
            "shape_evidence_reliable": bool(comparable),
            "clear_frame_count": float(len(comparable)),
            "matching_frame_count": float(matching_count),
            "clamp_reference_similarity": (
                None if not similarities else float(np.median(similarities))
            ),
            "evidence_consistent": evidence_consistent,
        }
        evidence = self._write_evidence(
            chosen,
            comparable[: self.maximum_evidence_frames],
            references,
        )
        return ExtractedEvidence(features=features, evidence=evidence)

    def build_reference(
        self,
        video_path: Path,
        time_range: TimeRange,
        *,
        video_id: str,
        replace: bool,
    ) -> Path:
        if video_id != "success":
            raise ValueError("CP04 reference must be built from the configured success video")
        manifest_path = self.reference_dir / "manifest.json"
        if manifest_path.exists() and not replace:
            raise FileExistsError(
                f"CP04 reference already exists at {manifest_path}; use --replace"
            )

        hand_items = self._track_text(
            video_path,
            time_range,
            object_id="cp04_gloved_hand",
            prompt_text=HAND_PROMPT,
        )
        observations = self._measure_hand_items(video_path, hand_items, [])
        if self.enable_local_box_refinement:
            refined = self._refine_from_local_box(
                video_path, time_range, hand_items, observations, []
            )
            if refined:
                observations = refined
        reliable = [item for item in observations if item.display.reliable]
        reliable.sort(key=self._quality_score, reverse=True)
        selected_observations = reliable[: self.maximum_evidence_frames]
        if len(selected_observations) < 2:
            raise RuntimeError("fewer than two reliable CP04 success reference frames")

        self.reference_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = self.reference_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, float | int | str]] = []
        for observation in selected_observations:
            name = f"{observation.item.frame_index:08d}.png"
            path = mask_dir / name
            if not cv2.imwrite(str(path), observation.clamp_mask.astype(np.uint8) * 255):
                raise OSError(f"could not write CP04 reference mask: {path}")
            rows.append(
                {
                    "frame_index": observation.item.frame_index,
                    "time_sec": observation.item.frame_time_sec,
                    "mask_path": f"masks/{name}",
                }
            )
        atomic_write_json(
            manifest_path,
            {
                "reference_version": "cp04-reference-v1",
                "video_id": video_id,
                "video_filename": video_path.name,
                "candidate_source": selected_observations[0].prompt_text,
                "model_version": self.model_version,
                "frames": rows,
            },
        )
        self._write_evidence(selected_observations, selected_observations, [])
        return manifest_path

    def _track_text(
        self,
        video_path: Path,
        time_range: TimeRange,
        *,
        object_id: str,
        prompt_text: str,
    ) -> list[FrameMasks]:
        prompt = SegmentationPrompt(
            object_id=object_id,
            kind="text",
            frame_time_sec=time_range.start_sec,
            text=prompt_text,
        )
        try:
            return list(
                self.segmenter.track(
                    video_path,
                    time_range,
                    [prompt],
                    sample_fps=self.sample_fps,
                )
            )
        except Sam3AmbiguousTextResult:
            return []

    def _measure_hand_items(
        self,
        video_path: Path,
        items: list[FrameMasks],
        references: list[np.ndarray],
    ) -> list[_ClampObservation]:
        observations: list[_ClampObservation] = []
        for item in items:
            raw_hand = self._mask(item, "cp04_gloved_hand")
            if raw_hand is None:
                continue
            frame = read_frame(video_path, item.frame_index)
            hand = self._resize_mask(raw_hand, frame.shape[:2])
            candidates = extract_metal_candidates_from_glove(frame, hand)
            if not candidates:
                continue
            score_floor = candidates[0].score - 0.20
            eligible = [candidate for candidate in candidates if candidate.score >= score_floor]
            candidate = max(eligible, key=lambda value: int(value.object_mask.sum()))
            display = measure_display_candidate(frame, candidate.hand_mask, candidate.object_mask)
            match = (
                compare_clamp_mask(candidate.object_mask, references)
                if display.reliable
                else None
            )
            observations.append(
                _ClampObservation(
                    item,
                    frame,
                    candidate.hand_mask,
                    candidate.object_mask,
                    display,
                    match,
                    OPENCV_CANDIDATE_SOURCE,
                )
            )
        return observations

    def _has_reference_match(self, observations: list[_ClampObservation]) -> bool:
        return any(
            item.display.reliable
            and item.match is not None
            and item.match.reliable
            and item.match.similarity is not None
            and item.match.similarity >= self.min_similarity
            for item in observations
        )

    def _refine_from_local_box(
        self,
        video_path: Path,
        time_range: TimeRange,
        hand_items: list[FrameMasks],
        observations: list[_ClampObservation],
        references: list[np.ndarray],
    ) -> list[_ClampObservation]:
        reliable = [item for item in observations if item.display.reliable]
        if not reliable:
            return []
        seed = max(reliable, key=lambda item: int(item.clamp_mask.sum()))
        ys, xs = np.nonzero(seed.clamp_mask)
        if len(xs) == 0:
            return []
        height, width = seed.frame.shape[:2]
        object_width = int(xs.max() - xs.min() + 1)
        object_height = int(ys.max() - ys.min() + 1)
        margin = max(
            8,
            round(min(height, width) * 0.05),
            round(max(object_width, object_height) * 0.5),
        )
        coordinates = [
            max(0, int(xs.min()) - margin) / width,
            max(0, int(ys.min()) - margin) / height,
            min(width, int(xs.max()) + margin + 1) / width,
            min(height, int(ys.max()) + margin + 1) / height,
        ]
        prompt = SegmentationPrompt(
            object_id="cp04_clamp_refined",
            kind="box",
            frame_time_sec=seed.item.frame_time_sec,
            coordinates=coordinates,
        )
        try:
            refined_items = list(
                self.segmenter.track(
                    video_path,
                    time_range,
                    [prompt],
                    sample_fps=self.sample_fps,
                )
            )
        except Sam3AmbiguousTextResult:
            return []

        hands_by_frame = {
            item.frame_index: self._mask(item, "cp04_gloved_hand") for item in hand_items
        }
        refined: list[_ClampObservation] = []
        for item in refined_items:
            raw_clamp = self._mask(item, "cp04_clamp_refined")
            raw_hand = hands_by_frame.get(item.frame_index)
            if raw_clamp is None or raw_hand is None:
                continue
            frame = read_frame(video_path, item.frame_index)
            hand = self._resize_mask(raw_hand, frame.shape[:2])
            clamp = self._resize_mask(raw_clamp, frame.shape[:2])
            display = measure_display_candidate(frame, hand, clamp)
            if not display.reliable:
                continue
            match = compare_clamp_mask(clamp, references) if references else None
            refined.append(
                _ClampObservation(
                    item,
                    frame,
                    hand,
                    clamp,
                    display,
                    match,
                    BOX_REFINEMENT_SOURCE,
                )
            )
        return refined

    def _load_reference_masks(self) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        paths = sorted((self.reference_dir / "masks").glob("*.png"))
        manifest_path = self.reference_dir / "manifest.json"
        if manifest_path.is_file():
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths = [self.reference_dir / row["mask_path"] for row in manifest["frames"]]
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None and np.any(image > 0):
                masks.append(image > 0)
        return masks

    def _write_evidence(
        self,
        observations: list[_ClampObservation],
        selected: list[_ClampObservation],
        references: list[np.ndarray],
    ) -> list[EvidenceItem]:
        selected_ids = {id(item) for item in selected}
        evidence: list[EvidenceItem] = []
        rows: list[dict[str, float | str | bool | None]] = []
        for observation in observations:
            stem = f"{observation.item.frame_index:08d}"
            paths = {
                "original": safe_child(self.evidence_root, f"cp_04/originals/{stem}.jpg"),
                "hand": safe_child(self.evidence_root, f"cp_04/masks/hand/{stem}.png"),
                "clamp": safe_child(self.evidence_root, f"cp_04/masks/clamp/{stem}.png"),
                "roi": safe_child(self.evidence_root, f"cp_04/rois/{stem}.jpg"),
                "normalized": safe_child(self.evidence_root, f"cp_04/normalized/{stem}.png"),
                "comparison": safe_child(self.evidence_root, f"cp_04/comparisons/{stem}.png"),
                "overlay": safe_child(self.evidence_root, f"cp_04/overlays/{stem}.jpg"),
            }
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            normalized = normalize_clamp_mask(observation.clamp_mask)
            images = {
                "original": observation.frame,
                "hand": self._mask_image(observation.hand_mask, observation.frame.shape[:2]),
                "clamp": observation.clamp_mask.astype(np.uint8) * 255,
                "roi": self._roi(observation),
                "normalized": normalized.astype(np.uint8) * 255,
                "comparison": self._comparison_panel(normalized, references),
                "overlay": self._overlay(observation),
            }
            for name, image in images.items():
                if not cv2.imwrite(str(paths[name]), image):
                    raise OSError(f"could not write CP04 evidence: {paths[name]}")
            similarity = None if observation.match is None else observation.match.similarity
            rows.append(
                {
                    "frame_index": float(observation.item.frame_index),
                    "time_sec": observation.item.frame_time_sec,
                    "prompt": observation.prompt_text,
                    "display_reliable": observation.display.reliable,
                    "display_reason": observation.display.reason,
                    "relative_area": observation.display.relative_area,
                    "hand_proximity_ratio": observation.display.hand_proximity_ratio,
                    "sharpness": observation.display.sharpness,
                    "elongation_ratio": observation.display.elongation_ratio,
                    "similarity": similarity,
                    "matches_reference": (
                        None if similarity is None else similarity >= self.min_similarity
                    ),
                }
            )
            if id(observation) in selected_ids:
                evidence.append(
                    EvidenceItem(
                        time_sec=observation.item.frame_time_sec,
                        overlay_path=paths["overlay"].relative_to(self.evidence_root).as_posix(),
                        rule="cp04_success_reference_shape",
                    )
                )
        atomic_write_json(
            safe_child(self.evidence_root, "cp_04/frame_analysis.json"),
            {"frames": rows},
        )
        return evidence

    @staticmethod
    def _mask(item: FrameMasks, object_id: str) -> np.ndarray | None:
        raw = item.masks.get(object_id)
        if raw is None or not np.asarray(raw, dtype=bool).any():
            return None
        return np.asarray(raw, dtype=bool)

    @staticmethod
    def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        if mask.shape == shape:
            return mask.astype(bool)
        return cv2.resize(
            mask.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    @staticmethod
    def _quality_score(observation: _ClampObservation) -> tuple[float, float]:
        return (
            float(observation.display.sharpness or 0.0),
            float(observation.display.relative_area or 0.0),
        )

    @staticmethod
    def _mask_image(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
        return (
            np.zeros(shape, dtype=np.uint8)
            if mask is None
            else mask.astype(np.uint8) * 255
        )

    @staticmethod
    def _overlay(observation: _ClampObservation) -> np.ndarray:
        canvas = observation.frame.copy()
        if observation.hand_mask is not None:
            contours, _ = cv2.findContours(
                observation.hand_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
        contours, _ = cv2.findContours(
            observation.clamp_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(canvas, contours, -1, (255, 255, 0), 2)
        return canvas

    @staticmethod
    def _roi(observation: _ClampObservation) -> np.ndarray:
        ys, xs = np.nonzero(observation.clamp_mask)
        if len(xs) == 0:
            return observation.frame.copy()
        margin = 16
        x1, x2 = max(0, xs.min() - margin), min(observation.frame.shape[1], xs.max() + margin + 1)
        y1, y2 = max(0, ys.min() - margin), min(observation.frame.shape[0], ys.max() + margin + 1)
        return observation.frame[y1:y2, x1:x2].copy()

    @staticmethod
    def _comparison_panel(
        normalized: np.ndarray, references: list[np.ndarray]
    ) -> np.ndarray:
        candidate = normalized.astype(np.uint8) * 255
        if not references:
            return cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR)
        reference = normalize_clamp_mask(references[0]).astype(np.uint8) * 255
        overlap = np.zeros((candidate.shape[0], candidate.shape[1], 3), np.uint8)
        overlap[:, :, 1] = reference
        overlap[:, :, 2] = candidate
        return np.hstack(
            (
                cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR),
                overlap,
            )
        )
