from pathlib import Path

import pytest

from tests.fixtures.make_test_video import make_test_video


def test_probe_and_sparse_sampling(tmp_path: Path) -> None:
    from medical_evaluation.video import probe_video, sample_frames

    path = make_test_video(tmp_path / "sample.mp4", fps=10, seconds=2, size=(160, 120))

    metadata = probe_video(path)
    frames = list(sample_frames(path, start_sec=0, end_sec=2, sample_fps=2))

    assert metadata.duration_sec == pytest.approx(2.0, abs=0.05)
    assert (metadata.width, metadata.height, round(metadata.fps)) == (160, 120, 10)
    assert len(frames) == 4
    assert [round(item.time_sec, 1) for item in frames] == [0.0, 0.5, 1.0, 1.5]
    assert all(item.image_bgr.shape == (120, 160, 3) for item in frames)


def test_probe_rejects_unsupported_extension(tmp_path: Path) -> None:
    from medical_evaluation.video import probe_video

    path = tmp_path / "not-video.txt"
    path.write_text("not a video", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported video extension"):
        probe_video(path)


def test_sampling_rejects_range_beyond_video(tmp_path: Path) -> None:
    from medical_evaluation.video import sample_frames

    path = make_test_video(tmp_path / "sample.mp4", fps=10, seconds=2, size=(160, 120))

    with pytest.raises(ValueError, match="outside video duration"):
        list(sample_frames(path, start_sec=1, end_sec=3, sample_fps=2))
