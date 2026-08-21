from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

Point = tuple[float, float]


def displacement_sequence(points: Sequence[Point]) -> list[Point]:
    return [
        (current[0] - previous[0], current[1] - previous[1])
        for previous, current in pairwise(points)
    ]


def velocity_sequence(points: Sequence[Point], times: Sequence[float]) -> list[Point]:
    if len(points) != len(times):
        raise ValueError("points and times must have equal length")
    deltas = displacement_sequence(points)
    velocities: list[Point] = []
    for index, displacement in enumerate(deltas):
        elapsed = times[index + 1] - times[index]
        if elapsed <= 0:
            raise ValueError("times must be strictly increasing")
        velocities.append((displacement[0] / elapsed, displacement[1] / elapsed))
    return velocities


def trajectory_length(points: Sequence[Point]) -> float:
    return float(sum(math.hypot(dx, dy) for dx, dy in displacement_sequence(points)))


def direction_change_count(
    displacements: Sequence[Point],
    *,
    minimum_angle_degrees: float = 90,
) -> int:
    if not 0 <= minimum_angle_degrees <= 180:
        raise ValueError("minimum angle must be between zero and 180 degrees")
    count = 0
    for first, second in pairwise(displacements):
        first_length = math.hypot(*first)
        second_length = math.hypot(*second)
        if first_length == 0 or second_length == 0:
            continue
        cosine = (first[0] * second[0] + first[1] * second[1]) / (
            first_length * second_length
        )
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if angle >= minimum_angle_degrees:
            count += 1
    return count
