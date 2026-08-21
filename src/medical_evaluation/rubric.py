from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from medical_evaluation.domain import TimeRange


class CheckpointRule(BaseModel):
    id: str = Field(pattern=r"^cp_\d{2}$")
    name: str = Field(min_length=1)
    criteria: list[str] = Field(min_length=1)
    weight: float = Field(gt=0, le=1)
    judge_type: str = Field(min_length=1)
    required_objects: list[str]
    reference_time: TimeRange
    thresholds: dict[str, float] = Field(default_factory=dict)


class Rubric(BaseModel):
    version: str = Field(min_length=1)
    checkpoints: list[CheckpointRule]

    @model_validator(mode="after")
    def validate_checkpoints(self) -> Rubric:
        if len(self.checkpoints) != 11:
            raise ValueError("rubric must contain 11 checkpoints")
        expected_ids = [f"cp_{number:02d}" for number in range(1, 12)]
        if [item.id for item in self.checkpoints] != expected_ids:
            raise ValueError("checkpoint IDs must be ordered cp_01 through cp_11")
        if abs(sum(item.weight for item in self.checkpoints) - 1.0) > 1e-9:
            raise ValueError("checkpoint weights must sum to 1")
        return self


def load_rubric(path: Path) -> Rubric:
    with path.open("r", encoding="utf-8") as handle:
        return Rubric.model_validate(yaml.safe_load(handle))


def parse_time_range(value: str) -> TimeRange:
    compact = value.strip().replace(" ", "")
    parts = compact.split("-", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"invalid time range: {value!r}")
    return TimeRange(start_sec=_parse_timestamp(parts[0]), end_sec=_parse_timestamp(parts[1]))


def split_criteria(value: str) -> list[str]:
    text = value.strip()
    if not text:
        raise ValueError("criterion text cannot be empty")
    matches = list(re.finditer(r"(?:^|[，,；;\n])\s*\d+[、.]\s*", text))
    if not matches:
        return [text]
    items: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        item = text[start:end].strip(" \t\r\n；;")
        if item:
            items.append(item)
    if not items:
        raise ValueError(f"could not split criterion text: {value!r}")
    return items


def _parse_timestamp(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid timestamp: {value!r}")
    hours, minutes, seconds = (float(part) for part in parts)
    return hours * 3600 + minutes * 60 + seconds
