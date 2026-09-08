from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.storage import atomic_write_json, safe_child


@dataclass(frozen=True)
class FrameReference:
    schema_version: int
    video_id: str
    anchor_frame_index: int
    anchor_time_sec: float
    anchor_image_path: str
    anchor_head_mask_path: str
    frame_mask_path: str
    frame_center_x_ratio: float
    frame_center_y_ratio: float
    frame_scale_ratio: float
    frame_angle_deg: float
    stable_duration_sec: float
    template_id: str
    template_compatible: bool


class FrameReferenceStore:
    def __init__(self, evidence_root: Path) -> None:
        self.evidence_root = evidence_root
        self.path = safe_child(evidence_root, "cp_09/frame_reference.json")

    def save(
        self,
        reference: FrameReference,
        *,
        anchor_image: np.ndarray,
        anchor_head_mask: np.ndarray,
        frame_mask: np.ndarray,
    ) -> None:
        artifacts = {
            reference.anchor_image_path: anchor_image,
            reference.anchor_head_mask_path: np.asarray(anchor_head_mask, dtype=np.uint8) * 255,
            reference.frame_mask_path: np.asarray(frame_mask, dtype=np.uint8) * 255,
        }
        for relative, value in artifacts.items():
            path = safe_child(self.evidence_root, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), value):
                raise OSError(f"could not write frame reference artifact: {path}")
        atomic_write_json(self.path, asdict(reference))

    def load(self) -> FrameReference | None:
        if not self.path.is_file():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported frame reference schema")
        return FrameReference(**payload)

    def read_artifacts(
        self, reference: FrameReference
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = cv2.imread(
            str(safe_child(self.evidence_root, reference.anchor_image_path)),
            cv2.IMREAD_COLOR,
        )
        head = cv2.imread(
            str(safe_child(self.evidence_root, reference.anchor_head_mask_path)),
            cv2.IMREAD_GRAYSCALE,
        )
        frame = cv2.imread(
            str(safe_child(self.evidence_root, reference.frame_mask_path)),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None or head is None or frame is None:
            raise ValueError("frame reference artifacts are missing or unreadable")
        if image.shape[:2] != head.shape or head.shape != frame.shape:
            raise ValueError("frame reference artifact dimensions differ")
        return image, head > 0, frame > 0
