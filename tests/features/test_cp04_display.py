from __future__ import annotations

import numpy as np

from medical_evaluation.features.cp04_display import (
    DisplayFrameCandidate,
    build_hand_object_crop,
    select_stable_display_frames,
)


def test_hand_object_crop_keeps_seed_and_removes_outside_plier() -> None:
    background = np.array((170, 100, 40), dtype=np.uint8)
    glove = np.array((225, 225, 225), dtype=np.uint8)
    metal = np.array((35, 40, 45), dtype=np.uint8)
    plier = np.array((10, 20, 30), dtype=np.uint8)
    frame = np.full((120, 160, 3), background, dtype=np.uint8)
    hand = np.zeros((120, 160), dtype=bool)
    hand[25:105, 30:115] = True
    hand[50:68, 60:82] = False
    frame[hand] = glove
    frame[50:68, 60:82] = metal
    frame[12:42, 128:150] = plier
    seed = np.zeros_like(hand)
    seed[53:65, 63:79] = True

    crop = build_hand_object_crop(frame, hand, seed, output_size=320)

    assert np.any(np.all(crop.image_bgr == metal, axis=2))
    assert not np.any(np.all(crop.image_bgr == plier, axis=2))
    projected = crop.project_mask(seed)
    restored = crop.restore_mask(projected)
    assert restored.shape == hand.shape
    assert restored[59, 70]
    assert not restored[20, 138]


def test_hand_object_crop_uses_uniform_background_outside_support() -> None:
    frame = np.full((100, 140, 3), (190, 120, 50), dtype=np.uint8)
    hand = np.zeros((100, 140), dtype=bool)
    hand[20:90, 25:100] = True
    frame[hand] = (230, 230, 230)
    seed = np.zeros_like(hand)
    seed[45:58, 55:70] = True
    frame[seed] = (40, 45, 50)

    crop = build_hand_object_crop(frame, hand, seed, output_size=200)

    outside = ~crop.project_mask(crop.support_mask)
    assert outside.any()
    assert np.all(crop.image_bgr[outside] == np.asarray(crop.background_bgr))


def _candidate(index: int, center_x: int, *, boundary: bool = False) -> DisplayFrameCandidate:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    hand = np.zeros((100, 120), dtype=bool)
    hand[15:90, 15:105] = True
    seed = np.zeros_like(hand)
    seed[45:53, center_x - 4 : center_x + 4] = True
    return DisplayFrameCandidate(
        frame_index=index,
        frame_time_sec=index / 2,
        frame_bgr=frame,
        hand_mask=hand,
        seed_mask=seed,
        sharpness=120.0,
        boundary_clipped=boundary,
        score=0.9,
    )


def test_select_stable_display_frames_excludes_distant_and_clipped_candidates() -> None:
    candidates = [
        _candidate(0, 55),
        _candidate(1, 57),
        _candidate(2, 54),
        _candidate(3, 91),
        _candidate(4, 56, boundary=True),
    ]

    selected = select_stable_display_frames(
        candidates, maximum_frames=5, minimum_stable_frames=2
    )

    assert [item.frame_index for item in selected] == [0, 1, 2]


def test_select_stable_display_frames_rejects_isolated_candidate() -> None:
    selected = select_stable_display_frames(
        [_candidate(0, 55)], maximum_frames=5, minimum_stable_frames=2
    )

    assert selected == []
