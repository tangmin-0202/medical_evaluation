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


def test_write_sampled_frame_sequence_includes_required_prompt_frame(
    tmp_path: Path,
) -> None:
    from medical_evaluation.domain import TimeRange
    from medical_evaluation.video import write_sampled_frame_sequence

    video = make_test_video(
        tmp_path / "sample.mp4",
        fps=10,
        seconds=3,
        size=(160, 120),
    )
    output = tmp_path / "frames"

    sequence = write_sampled_frame_sequence(
        video,
        output,
        time_range=TimeRange(start_sec=1.0, end_sec=2.5),
        sample_fps=2,
        required_times_sec=[1.3, 2.0],
    )

    assert [item.source_frame_index for item in sequence.entries] == [10, 13, 15, 20]
    assert [item.local_frame_index for item in sequence.entries] == [0, 1, 2, 3]
    assert [path.name for path in sorted(output.glob("*.jpg"))] == [
        "00000.jpg",
        "00001.jpg",
        "00002.jpg",
        "00003.jpg",
    ]
    assert sequence.source_to_local[13] == 1
    assert sequence.entries[1].source_time_sec == pytest.approx(1.3)


def test_lower_fps_writes_fewer_non_prompt_frames(tmp_path: Path) -> None:
    from medical_evaluation.domain import TimeRange
    from medical_evaluation.video import write_sampled_frame_sequence

    video = make_test_video(
        tmp_path / "sample.mp4",
        fps=10,
        seconds=4,
        size=(160, 120),
    )
    interval = TimeRange(start_sec=1, end_sec=3)
    two_fps = write_sampled_frame_sequence(
        video,
        tmp_path / "two",
        time_range=interval,
        sample_fps=2,
        required_times_sec=[1.3],
    )
    one_fps = write_sampled_frame_sequence(
        video,
        tmp_path / "one",
        time_range=interval,
        sample_fps=1,
        required_times_sec=[1.3],
    )

    assert len(one_fps.entries) < len(two_fps.entries)
    assert 13 in one_fps.source_to_local
    assert 13 in two_fps.source_to_local


def test_windowed_sequence_rejects_prompt_outside_interval(tmp_path: Path) -> None:
    from medical_evaluation.domain import TimeRange
    from medical_evaluation.video import write_sampled_frame_sequence

    video = make_test_video(
        tmp_path / "sample.mp4",
        fps=10,
        seconds=3,
        size=(160, 120),
    )

    with pytest.raises(ValueError, match="prompt time"):
        write_sampled_frame_sequence(
            video,
            tmp_path / "frames",
            time_range=TimeRange(start_sec=1, end_sec=2),
            sample_fps=2,
            required_times_sec=[2.5],
        )
