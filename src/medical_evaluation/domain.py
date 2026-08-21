from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class CheckpointStatus(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    INCOMPLETE = "incomplete"
    NEEDS_REVIEW = "needs_review"


class TimeRange(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "TimeRange":
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        return self
