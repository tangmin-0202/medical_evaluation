from pathlib import Path

import numpy as np

from medical_evaluation.features.frame_reference import FrameReference, FrameReferenceStore


def test_frame_reference_round_trip_with_artifacts(tmp_path: Path) -> None:
    store = FrameReferenceStore(tmp_path)
    reference = FrameReference(
        schema_version=1,
        video_id="success",
        anchor_frame_index=10,
        anchor_time_sec=1.0,
        anchor_image_path="cp_09/reference/anchor.jpg",
        anchor_head_mask_path="cp_09/reference/head.png",
        frame_mask_path="cp_09/reference/frame.png",
        frame_center_x_ratio=0.5,
        frame_center_y_ratio=0.6,
        frame_scale_ratio=0.2,
        frame_angle_deg=3.0,
        stable_duration_sec=2.5,
        template_id="template-v1",
        template_compatible=True,
    )
    image = np.full((20, 30, 3), 80, np.uint8)
    head = np.zeros((20, 30), bool)
    head[2:18, 3:27] = True
    frame = np.zeros((20, 30), bool)
    frame[8:14, 10:20] = True

    store.save(reference, anchor_image=image, anchor_head_mask=head, frame_mask=frame)

    assert store.load() == reference
    loaded_image, loaded_head, loaded_frame = store.read_artifacts(reference)
    assert loaded_image.shape == image.shape
    assert np.array_equal(loaded_head, head)
    assert np.array_equal(loaded_frame, frame)
