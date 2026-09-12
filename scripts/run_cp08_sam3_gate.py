from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medical_evaluation.annotations import VideoAnnotations


PROMPT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "blunt_instrument": (
        "blunt dental instrument",
        "dental instrument with a rounded blunt working tip",
        "metal dental instrument with a curved shaft and rounded working tip",
    ),
    "sharp_probe": (
        "sharp dental explorer probe",
        "pointed dental probe with a needle-like tip",
    ),
    "target_tooth": (
        "isolated white tooth inside the rubber dam clamp",
        "white molar enclosed by the metal rubber dam clamp",
    ),
    "rubber_dam_clamp": (
        "metal rubber dam clamp around the tooth",
        "complete stainless steel rubber dam clamp with two side wings",
    ),
    "rubber_dam": ("green dental rubber dam",),
}


@dataclass(frozen=True)
class GateWindows:
    cp08: tuple[float, float]
    cp09_tail: tuple[float, float]


def load_gate_windows(annotation_path: Path) -> GateWindows:
    annotations = VideoAnnotations.model_validate_json(
        annotation_path.read_text(encoding="utf-8")
    )
    ranges = {step.checkpoint_id: step.time_range for step in annotations.steps}
    for checkpoint_id in ("cp_08", "cp_09"):
        if checkpoint_id not in ranges:
            raise ValueError(f"Missing {checkpoint_id} annotation range")
    cp08 = ranges["cp_08"]
    cp09 = ranges["cp_09"]
    return GateWindows(
        cp08=(cp08.start_sec, cp08.end_sec),
        cp09_tail=(max(cp09.start_sec, cp09.end_sec - 3.0), cp09.end_sec),
    )

