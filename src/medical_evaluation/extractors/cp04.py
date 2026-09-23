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
    select_metal_candidate_near_seed,
)
from medical_evaluation.features.cp04_display import (
    DisplayFrameCandidate,
    HandObjectCrop,
    build_hand_object_crop,
    display_candidate_is_clipped,
    select_stable_display_frames,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.segmentation.base import FrameMasks, SegmentationPrompt, VideoSegmenter
from medical_evaluation.segmentation.sam3_backend import Sam3AmbiguousTextResult
from medical_evaluation.storage import atomic_write_json, safe_child
from medical_evaluation.video import read_frame

HAND_PROMPT = "open white gloved palm holding a small shiny metal clip"
LOCAL_CLAMP_PROMPT = "metal clip"
LOCAL_CLAMP_OBJECT_ID = "cp04_clamp_local"
LOCAL_CLAMP_SOURCE = "sam3:local-hand-crop"
LOCAL_ROI_LOCATOR_PROMPT = "small curved metal piece"
LOCAL_ROI_CLAMP_PROMPT = "small metal object"
LOCAL_ROI_SOURCE = "sam3:lossless-clamp-roi"
EXEMPLAR_SOURCE = "sam3:success-exemplar-roi"
LOCAL_OPENCV_SOURCE = "opencv:isolated-hand-crop"


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
    crop_clip_position: int | None = None


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
    ) -> None:
        self.segmenter = segmenter
        self.evidence_root = evidence_root
        self.reference_dir = reference_dir
        self.min_similarity = min_similarity

    @property
    def model_version(self) -> str:
        return f"{self.segmenter.model_version}+cp04-local-hand-crop-v1+cp04-reference-v1"

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
        lossless_single = (
            len(comparable) == 1
            and comparable[0].prompt_text in {LOCAL_ROI_SOURCE, EXEMPLAR_SOURCE}
        )
        if lossless_single:
            evidence_consistent = True
        features: dict[str, float | bool | None] = {
            "clamp_observed": observed_any,
            "shape_evidence_reliable": len(comparable) >= 2 or lossless_single,
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
        observations = self._measure_hand_items(
            video_path, hand_items, [], allow_local_fallback=False
        )
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
                    "crop_clip_position": int(observation.crop_clip_position or 0),
                    "mask_path": f"masks/{name}",
                }
            )
        exemplar_dir = self.reference_dir / "exemplars"
        exemplar_dir.mkdir(parents=True, exist_ok=True)
        exemplar_image, exemplar_box = self._build_exemplar_tile(
            selected_observations[0].frame,
            selected_observations[0].clamp_mask,
        )
        exemplar_name = f"{selected_observations[0].item.frame_index:08d}.png"
        if not cv2.imwrite(str(exemplar_dir / exemplar_name), exemplar_image):
            raise OSError("could not write CP04 success exemplar image")
        atomic_write_json(
            manifest_path,
            {
                "reference_version": "cp04-reference-v1",
                "video_id": video_id,
                "video_filename": video_path.name,
                "candidate_source": selected_observations[0].prompt_text,
                "model_version": self.model_version,
                "frames": rows,
                "exemplar": {
                    "image_path": f"exemplars/{exemplar_name}",
                    "box_xyxy": list(exemplar_box),
                },
            },
        )
        self._write_evidence(selected_observations, selected_observations, [])
        return manifest_path

    @staticmethod
    def _build_exemplar_tile(
        frame: np.ndarray,
        object_mask: np.ndarray,
        *,
        tile_size: int = 512,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        ys, xs = np.nonzero(object_mask)
        if not len(xs):
            raise ValueError("cannot build an exemplar from an empty clamp mask")
        object_width = float(xs.max() - xs.min() + 1)
        object_height = float(ys.max() - ys.min() + 1)
        side = max(object_width, object_height) * 2.0
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        height, width = frame.shape[:2]
        x1 = max(0, round(center_x - side / 2))
        y1 = max(0, round(center_y - side / 2))
        x2 = min(width, round(center_x + side / 2))
        y2 = min(height, round(center_y + side / 2))
        crop = frame[y1:y2, x1:x2]
        local_mask = object_mask[y1:y2, x1:x2]
        content_size = tile_size - 48
        scale = min(content_size / crop.shape[1], content_size / crop.shape[0])
        resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        tile = np.full((tile_size, tile_size, 3), 127, dtype=np.uint8)
        left = (tile_size - resized.shape[1]) // 2
        top = (tile_size - resized.shape[0]) // 2
        tile[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        local_y, local_x = np.nonzero(local_mask)
        box = (
            left + round(float(local_x.min()) * scale),
            top + round(float(local_y.min()) * scale),
            left + round(float(local_x.max() + 1) * scale),
            top + round(float(local_y.max() + 1) * scale),
        )
        return tile, box

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
        *,
        allow_local_fallback: bool = True,
    ) -> list[_ClampObservation]:
        candidates: list[DisplayFrameCandidate] = []
        item_by_frame: dict[int, FrameMasks] = {}
        for item in items:
            raw_hand = self._mask(item, "cp04_gloved_hand")
            if raw_hand is None:
                continue
            frame = read_frame(video_path, item.frame_index)
            hand = self._resize_mask(raw_hand, frame.shape[:2])
            metal_candidates = extract_metal_candidates_from_glove(frame, hand)
            if not metal_candidates:
                continue
            score_floor = metal_candidates[0].score - 0.20
            eligible = [
                candidate
                for candidate in metal_candidates
                if candidate.score >= score_floor
            ]
            candidate = max(eligible, key=lambda value: int(value.object_mask.sum()))
            seed_display = measure_display_candidate(
                frame, candidate.hand_mask, candidate.object_mask
            )
            hand_component = np.asarray(candidate.hand_mask, dtype=bool)
            boundary_clipped = display_candidate_is_clipped(
                hand_component, candidate.object_mask
            )
            candidates.append(
                DisplayFrameCandidate(
                    frame_index=item.frame_index,
                    frame_time_sec=item.frame_time_sec,
                    frame_bgr=frame,
                    hand_mask=hand_component,
                    seed_mask=np.asarray(candidate.object_mask, dtype=bool),
                    sharpness=float(seed_display.sharpness or 0.0),
                    boundary_clipped=boundary_clipped,
                    score=float(candidate.score),
                )
            )
            item_by_frame[item.frame_index] = item

        selected = select_stable_display_frames(
            candidates, maximum_frames=5, minimum_stable_frames=2
        )
        if len(selected) < 2:
            return []
        crops = [
            build_hand_object_crop(
                item.frame_bgr, item.hand_mask, item.seed_mask, output_size=640
            )
            for item in selected
        ]
        clip_path = safe_child(
            self.evidence_root, "cp_04/local_input/display_clip.mp4"
        )
        self._write_local_clip(clip_path, crops)
        local_items = self._track_local_clip(clip_path, len(crops))
        observations: list[_ClampObservation] = []
        for local_item in local_items:
            position = self._local_position(local_item, len(crops))
            if position is None:
                continue
            raw_clamp = self._mask(local_item, LOCAL_CLAMP_OBJECT_ID)
            if raw_clamp is None:
                continue
            crop = crops[position]
            local_mask = self._resize_mask(raw_clamp, crop.image_bgr.shape[:2])
            if self._touches_boundary(local_mask):
                continue
            clamp = crop.restore_mask(local_mask)
            support_overlap = float(
                np.logical_and(clamp, crop.support_mask).sum() / max(int(clamp.sum()), 1)
            )
            if support_overlap < 0.8:
                continue
            source = selected[position]
            display = measure_display_candidate(
                source.frame_bgr, source.hand_mask, clamp
            )
            match = (
                compare_clamp_mask(clamp, references)
                if display.reliable and references
                else None
            )
            observations.append(
                _ClampObservation(
                    item=item_by_frame[source.frame_index],
                    frame=source.frame_bgr,
                    hand_mask=source.hand_mask,
                    clamp_mask=clamp,
                    display=display,
                    match=match,
                    prompt_text=LOCAL_CLAMP_SOURCE,
                    crop_clip_position=position,
                )
            )
        reliable_count = sum(item.display.reliable for item in observations)
        if reliable_count >= 2 or not allow_local_fallback:
            return observations
        roi_observations = self._measure_lossless_roi_candidates(
            crops, selected, item_by_frame, references
        )
        if any(item.display.reliable for item in roi_observations):
            return roi_observations
        return self._measure_isolated_crop_candidates(
            crops, selected, item_by_frame, references
        )

    def _measure_lossless_roi_candidates(
        self,
        crops: list[HandObjectCrop],
        selected: list[DisplayFrameCandidate],
        item_by_frame: dict[int, FrameMasks],
        references: list[np.ndarray],
    ) -> list[_ClampObservation]:
        segment_image = getattr(self.segmenter, "segment_image", None)
        if segment_image is None:
            return []
        locator_prompt = SegmentationPrompt(
            object_id=LOCAL_CLAMP_OBJECT_ID,
            kind="text",
            text=LOCAL_ROI_LOCATOR_PROMPT,
        )
        locator_mask: np.ndarray | None = None
        for crop in reversed(crops):
            item = segment_image(crop.image_bgr, locator_prompt)
            if item is None:
                continue
            locator_mask = self._mask(item, LOCAL_CLAMP_OBJECT_ID)
            if locator_mask is not None and locator_mask.any():
                break
        if locator_mask is None or not locator_mask.any():
            return []
        ys, xs = np.nonzero(locator_mask)
        object_width = float(xs.max() - xs.min() + 1)
        object_height = float(ys.max() - ys.min() + 1)
        side = max(object_width, object_height) * 2.5
        center_x = float(xs.mean())
        center_y = float(ys.mean()) + 0.65 * object_height
        height, width = crops[0].image_bgr.shape[:2]
        x1 = max(0, round(center_x - side / 2))
        y1 = max(0, round(center_y - side / 2))
        x2 = min(width, round(center_x + side / 2))
        y2 = min(height, round(center_y + side / 2))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return []
        exemplar_observations = self._measure_exemplar_candidates(
            crops,
            selected,
            item_by_frame,
            references,
            (x1, y1, x2, y2),
        )
        if any(item.display.reliable for item in exemplar_observations):
            return exemplar_observations
        clamp_prompt = SegmentationPrompt(
            object_id=LOCAL_CLAMP_OBJECT_ID,
            kind="text",
            text=LOCAL_ROI_CLAMP_PROMPT,
        )
        observations: list[_ClampObservation] = []
        for position, (crop, source) in enumerate(zip(crops, selected, strict=True)):
            roi = cv2.resize(
                crop.image_bgr[y1:y2, x1:x2],
                (640, 640),
                interpolation=cv2.INTER_CUBIC,
            )
            item = segment_image(roi, clamp_prompt)
            if item is None:
                continue
            roi_mask = self._mask(item, LOCAL_CLAMP_OBJECT_ID)
            if roi_mask is None or not roi_mask.any():
                continue
            local_mask = np.zeros(crop.image_bgr.shape[:2], dtype=bool)
            local_mask[y1:y2, x1:x2] = cv2.resize(
                roi_mask.astype(np.uint8),
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            clamp = crop.restore_mask(local_mask)
            display = measure_display_candidate(source.frame_bgr, source.hand_mask, clamp)
            match = compare_clamp_mask(clamp, references) if display.reliable and references else None
            observations.append(
                _ClampObservation(
                    item=item_by_frame[source.frame_index],
                    frame=source.frame_bgr,
                    hand_mask=source.hand_mask,
                    clamp_mask=clamp,
                    display=display,
                    match=match,
                    prompt_text=LOCAL_ROI_SOURCE,
                    crop_clip_position=position,
                )
            )
        return observations

    def _measure_exemplar_candidates(
        self,
        crops: list[HandObjectCrop],
        selected: list[DisplayFrameCandidate],
        item_by_frame: dict[int, FrameMasks],
        references: list[np.ndarray],
        target_box: tuple[int, int, int, int],
    ) -> list[_ClampObservation]:
        segment_exemplar = getattr(self.segmenter, "segment_image_exemplar", None)
        exemplar = self._load_exemplar()
        if segment_exemplar is None or exemplar is None:
            return []
        exemplar_image, exemplar_box = exemplar
        tile_size = 512
        exemplar_tile = cv2.resize(
            exemplar_image,
            (tile_size, tile_size),
            interpolation=cv2.INTER_CUBIC,
        )
        scale_x = tile_size / exemplar_image.shape[1]
        scale_y = tile_size / exemplar_image.shape[0]
        bx1, by1, bx2, by2 = exemplar_box
        prompt = SegmentationPrompt(
            object_id=LOCAL_CLAMP_OBJECT_ID,
            kind="box",
            coordinates=[
                bx1 * scale_x / (tile_size * 2),
                by1 * scale_y / tile_size,
                bx2 * scale_x / (tile_size * 2),
                by2 * scale_y / tile_size,
            ],
        )
        x1, y1, x2, y2 = target_box
        observations: list[_ClampObservation] = []
        for position, (crop, source) in enumerate(zip(crops, selected, strict=True)):
            target_tile = cv2.resize(
                crop.image_bgr[y1:y2, x1:x2],
                (tile_size, tile_size),
                interpolation=cv2.INTER_CUBIC,
            )
            composite = np.concatenate([exemplar_tile, target_tile], axis=1)
            item = segment_exemplar(composite, prompt)
            if item is None:
                continue
            composite_mask = self._mask(item, LOCAL_CLAMP_OBJECT_ID)
            if composite_mask is None or composite_mask.shape != composite.shape[:2]:
                continue
            target_mask = composite_mask[:, tile_size:]
            if not target_mask.any():
                continue
            local_mask = np.zeros(crop.image_bgr.shape[:2], dtype=bool)
            local_mask[y1:y2, x1:x2] = cv2.resize(
                target_mask.astype(np.uint8),
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            clamp = crop.restore_mask(local_mask)
            display = measure_display_candidate(source.frame_bgr, source.hand_mask, clamp)
            match = compare_clamp_mask(clamp, references) if display.reliable and references else None
            observations.append(
                _ClampObservation(
                    item=item_by_frame[source.frame_index],
                    frame=source.frame_bgr,
                    hand_mask=source.hand_mask,
                    clamp_mask=clamp,
                    display=display,
                    match=match,
                    prompt_text=EXEMPLAR_SOURCE,
                    crop_clip_position=position,
                )
            )
        return observations

    def _measure_isolated_crop_candidates(
        self,
        crops: list[HandObjectCrop],
        selected: list[DisplayFrameCandidate],
        item_by_frame: dict[int, FrameMasks],
        references: list[np.ndarray],
    ) -> list[_ClampObservation]:
        observations: list[_ClampObservation] = []
        for position, (crop, source) in enumerate(zip(crops, selected, strict=True)):
            local_hand = crop.project_mask(source.hand_mask)
            candidates = extract_metal_candidates_from_glove(
                crop.image_bgr, local_hand
            )
            candidate = select_metal_candidate_near_seed(
                candidates,
                crop.project_mask(source.seed_mask),
            )
            if candidate is None:
                continue
            clamp = crop.restore_mask(candidate.object_mask)
            display = measure_display_candidate(
                source.frame_bgr, source.hand_mask, clamp
            )
            match = (
                compare_clamp_mask(clamp, references)
                if display.reliable and references
                else None
            )
            observations.append(
                _ClampObservation(
                    item=item_by_frame[source.frame_index],
                    frame=source.frame_bgr,
                    hand_mask=source.hand_mask,
                    clamp_mask=clamp,
                    display=display,
                    match=match,
                    prompt_text=LOCAL_OPENCV_SOURCE,
                    crop_clip_position=position,
                )
            )
        return observations

    def _track_local_clip(
        self, clip_path: Path, frame_count: int
    ) -> list[FrameMasks]:
        prompt = SegmentationPrompt(
            object_id=LOCAL_CLAMP_OBJECT_ID,
            kind="text",
            frame_time_sec=0.0,
            text=LOCAL_CLAMP_PROMPT,
        )
        try:
            return list(
                self.segmenter.track(
                    clip_path,
                    TimeRange(start_sec=0.0, end_sec=frame_count / self.sample_fps),
                    [prompt],
                    sample_fps=self.sample_fps,
                )
            )
        except Sam3AmbiguousTextResult:
            return []

    @staticmethod
    def _write_local_clip(path: Path, crops: list[HandObjectCrop]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = crops[0].image_bgr.shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height)
        )
        if not writer.isOpened():
            raise OSError(f"could not create CP04 local display clip: {path}")
        try:
            for crop in crops:
                writer.write(crop.image_bgr)
        finally:
            writer.release()

    @staticmethod
    def _local_position(item: FrameMasks, frame_count: int) -> int | None:
        candidates = [item.sample_position, item.frame_index]
        for value in candidates:
            if value is not None and 0 <= value < frame_count:
                return int(value)
        derived = round(item.frame_time_sec * 2.0)
        return derived if 0 <= derived < frame_count else None

    @staticmethod
    def _touches_boundary(mask: np.ndarray) -> bool:
        value = np.asarray(mask, dtype=bool)
        return bool(
            value[0].any()
            or value[-1].any()
            or value[:, 0].any()
            or value[:, -1].any()
        )

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

    def _load_exemplar(
        self,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
        manifest_path = self.reference_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        exemplar = manifest.get("exemplar")
        if not isinstance(exemplar, dict):
            return None
        image_path = exemplar.get("image_path")
        box = exemplar.get("box_xyxy")
        if not isinstance(image_path, str) or not isinstance(box, list) or len(box) != 4:
            return None
        image = cv2.imread(str(self.reference_dir / image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        values = tuple(float(value) for value in box)
        return image, values

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
