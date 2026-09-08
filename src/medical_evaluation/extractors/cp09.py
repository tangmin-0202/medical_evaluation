from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.annotations import BoxPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.features.frame_reference import FrameReference, FrameReferenceStore
from medical_evaluation.features.geometry import (
    relative_bbox_center_offset_to_box,
    write_reference_overlay,
)
from medical_evaluation.features.mannequin_registration import (
    estimate_similarity_registration,
    invert_similarity,
    warp_mask,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.segmentation.base import FrameMasks, VideoSegmenter
from medical_evaluation.segmentation.prompt_policy import (
    AnnotationPromptPolicy,
    Cp09Cp11PromptPolicy,
    TextPromptPolicy,
)
from medical_evaluation.storage import safe_child
from medical_evaluation.video import read_frame


class Cp09FeatureExtractor:
    minimum_valid_frames = 3
    prompt_boundary_tolerance_sec = 0.5

    def __init__(
        self,
        *,
        segmenter: VideoSegmenter,
        annotations: VideoAnnotations,
        evidence_root: Path,
        prompt_policy: Cp09Cp11PromptPolicy | None = None,
        template: Mapping[str, object] | None = None,
    ) -> None:
        self.segmenter = segmenter
        self.annotations = annotations
        self.evidence_root = evidence_root
        self.prompt_policy = prompt_policy or AnnotationPromptPolicy()
        self.template = dict(template or {})
        self.reference_store = FrameReferenceStore(evidence_root)

    @property
    def model_version(self) -> str:
        return self.segmenter.model_version

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id != "cp_09":
            raise ValueError("Cp09FeatureExtractor is CP09-only")
        if analysis_width <= 0:
            raise ValueError("analysis_width must be positive")

        if isinstance(self.prompt_policy, TextPromptPolicy):
            return self._extract_head_relative(
                video_path, checkpoint_id, time_range, dense_fps=dense_fps
            )

        frame_prompts = self.prompt_policy.frame_prompts(
            self.annotations,
            time_range,
            checkpoint_id=checkpoint_id,
            boundary_tolerance_sec=self.prompt_boundary_tolerance_sec,
        )
        oral_reference = self._oral_reference_box(time_range, checkpoint_id)
        tracking_range = TimeRange(
            start_sec=min(time_range.start_sec, *(item.frame_time_sec for item in frame_prompts)),
            end_sec=max(time_range.end_sec, *(item.frame_time_sec for item in frame_prompts)),
        )
        tracked = self.segmenter.track(
            video_path,
            tracking_range,
            frame_prompts,
            sample_fps=dense_fps,
        )

        frame_valid_count = 0
        valid_offsets: list[tuple[FrameMasks, np.ndarray, float]] = []
        for frame_masks in tracked:
            if not time_range.start_sec <= frame_masks.frame_time_sec <= time_range.end_sec:
                continue

            frame_mask = frame_masks.masks.get("rubber_dam_frame")
            frame_is_valid = frame_mask is not None and bool(np.asarray(frame_mask).any())
            frame_valid_count += int(frame_is_valid)

            if not frame_is_valid:
                continue
            assert frame_mask is not None
            offset = relative_bbox_center_offset_to_box(frame_mask, oral_reference)
            if offset is not None:
                valid_offsets.append((frame_masks, frame_mask, offset))

        evidence: list[EvidenceItem] = []
        selected = (
            sorted({0, len(valid_offsets) // 2, len(valid_offsets) - 1})
            if valid_offsets
            else []
        )
        for position in selected:
            frame_masks, frame_mask, _offset = valid_offsets[position]
            frame = read_frame(video_path, frame_masks.frame_index)
            mask_path = safe_child(
                self.evidence_root,
                f"cp_09/masks/rubber_dam_frame/{frame_masks.frame_index:08d}.png",
            )
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), frame_mask.astype(np.uint8) * 255):
                raise OSError(f"could not write mask: {mask_path}")

            overlay = write_reference_overlay(
                frame,
                frame_mask,
                oral_reference,
                self.evidence_root,
                f"cp_09/overlays/{frame_masks.frame_index:08d}.jpg",
            )
            evidence.append(
                EvidenceItem(
                    time_sec=frame_masks.frame_time_sec,
                    overlay_path=overlay.relative_to(self.evidence_root).as_posix(),
                    rule="rubber_dam_frame_relative_to_oral_reference",
                )
            )

        feature = (
            float(np.median([item[2] for item in valid_offsets]))
            if len(valid_offsets) >= self.minimum_valid_frames
            else None
        )
        return ExtractedEvidence(
            features={
                "frame_oral_center_offset": feature,
                "frame_valid_count": float(frame_valid_count),
                "oral_reference_count": 1.0,
                "relative_offset_valid_count": float(len(valid_offsets)),
            },
            evidence=evidence,
        )

    def _extract_head_relative(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
    ) -> ExtractedEvidence:
        head_frames = list(
            self.segmenter.track(
                video_path,
                time_range,
                self.prompt_policy.head_prompts(
                    self.annotations, time_range, checkpoint_id=checkpoint_id
                ),
                sample_fps=dense_fps,
            )
        )
        frame_frames = list(
            self.segmenter.track(
                video_path,
                time_range,
                self.prompt_policy.frame_prompts(
                    self.annotations, time_range, checkpoint_id=checkpoint_id
                ),
                sample_fps=dense_fps,
            )
        )
        heads = [
            item
            for item in head_frames
            if (mask := item.masks.get("mannequin_head")) is not None
            and np.asarray(mask, dtype=bool).any()
        ]
        empty: dict[str, float | bool | None] = {
            "head_registration_reliable": False,
            "head_valid_count": float(len(heads)),
            "frame_presence_ratio": 0.0,
            "frame_valid_count": 0.0,
            "frame_present_at_end": False,
            "frame_stable_duration_sec": 0.0,
            "frame_center_x_ratio": None,
            "frame_center_y_ratio": None,
            "frame_scale_ratio": None,
            "frame_angle_deg": None,
            "head_template_compatible": False,
        }
        if len(heads) < self.minimum_valid_frames:
            return ExtractedEvidence(features=empty, evidence=[])

        anchor_item = heads[-1]
        anchor_head = np.asarray(anchor_item.masks["mannequin_head"], dtype=bool)
        anchor_image = read_frame(video_path, anchor_item.frame_index)
        frames_by_index = {item.frame_index: item for item in frame_frames}
        mapped: list[tuple[FrameMasks, np.ndarray]] = []
        reliable_count = 0
        for head_item in heads:
            head_mask = np.asarray(head_item.masks["mannequin_head"], dtype=bool)
            image = read_frame(video_path, head_item.frame_index)
            if head_item.frame_index == anchor_item.frame_index:
                registration_ok = True
                matrix = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=float)
            else:
                registration = estimate_similarity_registration(
                    anchor_image, anchor_head, image, head_mask
                )
                registration_ok = registration.accepted
                matrix = registration.matrix
            if not registration_ok:
                continue
            reliable_count += 1
            frame_item = frames_by_index.get(head_item.frame_index)
            frame_mask = (
                frame_item.masks.get("rubber_dam_frame")
                if frame_item is not None
                else None
            )
            if frame_mask is None or not np.asarray(frame_mask, dtype=bool).any():
                continue
            mapped_mask = warp_mask(
                np.asarray(frame_mask, dtype=bool),
                invert_similarity(matrix),
                output_shape=anchor_head.shape,
            )
            if mapped_mask.any():
                mapped.append((head_item, mapped_mask))

        head_reliable = reliable_count >= self.minimum_valid_frames
        presence_ratio = float(len(mapped) / reliable_count) if reliable_count else 0.0
        final_cutoff = time_range.end_sec - 2.0
        final_head_indices = {
            item.frame_index for item in heads if item.frame_time_sec >= final_cutoff
        }
        final_mapped = [item for item in mapped if item[0].frame_index in final_head_indices]
        last_head_index = max(heads, key=lambda item: item.frame_time_sec).frame_index
        present_at_end = any(
            item.frame_index == last_head_index for item, _mask in final_mapped
        )
        stable = self._stable_tail(
            final_mapped or mapped,
            anchor_head,
            max_gap_sec=1.5 / dense_fps,
        )
        template_reference = self.template.get("reference", {})
        template_compatible = bool(
            isinstance(template_reference, Mapping)
            and template_reference.get("video_id") == self.annotations.video_id
        )
        features = dict(empty)
        features.update(
            {
                "head_registration_reliable": head_reliable,
                "head_valid_count": float(reliable_count),
                "frame_presence_ratio": presence_ratio,
                "frame_valid_count": float(len(mapped)),
                "frame_present_at_end": present_at_end,
                "head_template_compatible": template_compatible,
            }
        )
        evidence: list[EvidenceItem] = []
        if stable is not None:
            stable_items, median_mask, metrics = stable
            duration = (
                stable_items[-1][0].frame_time_sec
                - stable_items[0][0].frame_time_sec
                + 1.0 / dense_fps
            )
            features.update(
                {
                    "frame_stable_duration_sec": float(max(0.0, duration)),
                    **metrics,
                }
            )
            reference = FrameReference(
                schema_version=1,
                video_id=self.annotations.video_id,
                anchor_frame_index=anchor_item.frame_index,
                anchor_time_sec=anchor_item.frame_time_sec,
                anchor_image_path="cp_09/reference/anchor.jpg",
                anchor_head_mask_path="cp_09/reference/head.png",
                frame_mask_path="cp_09/reference/frame.png",
                frame_center_x_ratio=float(metrics["frame_center_x_ratio"]),
                frame_center_y_ratio=float(metrics["frame_center_y_ratio"]),
                frame_scale_ratio=float(metrics["frame_scale_ratio"]),
                frame_angle_deg=float(metrics["frame_angle_deg"]),
                stable_duration_sec=float(max(0.0, duration)),
                template_id=str(self.template.get("template_id", "unconfigured")),
                template_compatible=template_compatible,
            )
            self.reference_store.save(
                reference,
                anchor_image=anchor_image,
                anchor_head_mask=anchor_head,
                frame_mask=median_mask,
            )
            evidence = self._write_head_relative_evidence(
                video_path, stable_items, anchor_head
            )
        return ExtractedEvidence(features=features, evidence=evidence)

    @staticmethod
    def _frame_geometry(mask: np.ndarray, head_mask: np.ndarray) -> dict[str, float]:
        ys, xs = np.nonzero(mask)
        head_ys, head_xs = np.nonzero(head_mask)
        if not len(xs) or not len(head_xs):
            raise ValueError("frame and head masks must be non-empty")
        points = np.column_stack((xs, ys)).astype(np.float32)
        (_cx, _cy), (width, height), angle = cv2.minAreaRect(points)
        if width < height:
            angle += 90.0
        head_width = max(1.0, float(head_xs.max() - head_xs.min() + 1))
        head_height = max(1.0, float(head_ys.max() - head_ys.min() + 1))
        return {
            "frame_center_x_ratio": float((np.median(xs) - head_xs.min()) / head_width),
            "frame_center_y_ratio": float((np.median(ys) - head_ys.min()) / head_height),
            "frame_scale_ratio": float(np.sqrt(mask.sum() / max(1, head_mask.sum()))),
            "frame_angle_deg": float(angle),
        }

    def _stable_tail(
        self,
        values: list[tuple[FrameMasks, np.ndarray]],
        head_mask: np.ndarray,
        *,
        max_gap_sec: float,
    ) -> tuple[list[tuple[FrameMasks, np.ndarray]], np.ndarray, dict[str, float]] | None:
        if len(values) < self.minimum_valid_frames:
            return None
        ordered = sorted(values, key=lambda value: value[0].frame_time_sec)
        contiguous = [ordered[-1]]
        for value in reversed(ordered[:-1]):
            if contiguous[0][0].frame_time_sec - value[0].frame_time_sec > max_gap_sec:
                break
            contiguous.insert(0, value)
        geometries = [
            self._frame_geometry(mask, head_mask) for _item, mask in contiguous
        ]
        final_geometry = geometries[-1]
        stable_values = [contiguous[-1]]
        for value, geometry in reversed(
            list(zip(contiguous[:-1], geometries[:-1], strict=True))
        ):
            center_delta = np.hypot(
                geometry["frame_center_x_ratio"]
                - final_geometry["frame_center_x_ratio"],
                geometry["frame_center_y_ratio"]
                - final_geometry["frame_center_y_ratio"],
            )
            if (
                center_delta > 0.05
                or abs(
                    geometry["frame_scale_ratio"]
                    - final_geometry["frame_scale_ratio"]
                )
                > 0.12
                or abs(
                    geometry["frame_angle_deg"] - final_geometry["frame_angle_deg"]
                )
                > 12.0
            ):
                break
            stable_values.insert(0, value)
        if len(stable_values) < self.minimum_valid_frames:
            return None
        median_mask = np.mean(
            [mask.astype(np.float32) for _item, mask in stable_values], axis=0
        ) >= 0.5
        return stable_values, median_mask, self._frame_geometry(median_mask, head_mask)

    def _write_head_relative_evidence(
        self,
        video_path: Path,
        values: list[tuple[FrameMasks, np.ndarray]],
        anchor_head: np.ndarray,
    ) -> list[EvidenceItem]:
        selected = sorted({0, len(values) // 2, len(values) - 1})
        evidence: list[EvidenceItem] = []
        for position in selected:
            item, mapped_frame = values[position]
            image = read_frame(video_path, item.frame_index)
            if image.shape[:2] != anchor_head.shape:
                continue
            canvas = image.copy()
            for mask, color, thickness in (
                (anchor_head, (0, 255, 0), 2),
                (mapped_frame, (255, 255, 255), 3),
            ):
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(canvas, contours, -1, color, thickness)
            output = safe_child(
                self.evidence_root, f"cp_09/overlays/{item.frame_index:08d}.jpg"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), canvas):
                raise OSError(f"could not write evidence overlay: {output}")
            evidence.append(
                EvidenceItem(
                    time_sec=item.frame_time_sec,
                    overlay_path=output.relative_to(self.evidence_root).as_posix(),
                    rule="frame_installed_stably_relative_to_mannequin",
                )
            )
        return evidence

    def _oral_reference_box(
        self,
        time_range: TimeRange,
        checkpoint_id: str,
    ) -> tuple[float, float, float, float]:
        allowed_start = time_range.start_sec - self.prompt_boundary_tolerance_sec
        allowed_end = time_range.end_sec + self.prompt_boundary_tolerance_sec
        candidates = [
            prompt
            for prompt in self.annotations.prompts
            if prompt.object_id == "oral_region"
            and allowed_start <= prompt.frame_time_sec <= allowed_end
        ]
        if any(not isinstance(prompt, BoxPrompt) for prompt in candidates):
            raise ValueError(f"{checkpoint_id} oral_region must be a box")
        if len(candidates) != 1:
            raise ValueError(f"{checkpoint_id} requires exactly one oral_region box")
        reference = candidates[0]
        assert isinstance(reference, BoxPrompt)
        return reference.x1, reference.y1, reference.x2, reference.y2
