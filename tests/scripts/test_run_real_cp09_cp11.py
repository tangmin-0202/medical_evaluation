from scripts.run_real_cp09_cp11 import VIDEO_FILENAMES, build_parser


def test_real_runner_has_exact_three_demo_inputs() -> None:
    assert VIDEO_FILENAMES == {
        "success": "橡皮障完整.mp4",
        "failure": "橡皮障失败.mp4",
        "clamp_failure": "橡皮障夹子飞了.mp4",
    }
    args = build_parser().parse_args(["--video-id", "success", "--no-commentary"])
    assert args.video_id == "success"
    assert args.no_commentary is True
