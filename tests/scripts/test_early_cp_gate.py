import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from medical_evaluation.domain import TimeRange
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.segmentation.base import FrameMasks


def load_gate():
    scripts = Path(__file__).parents[2] / "scripts"
    spec = importlib.util.spec_from_file_location("early_gate", scripts / "run_early_cp_sam3_gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cp01_pen_only_uses_production_gate_without_other_objects(
    tmp_path, monkeypatch,
):
    gate = load_gate()
    observed = {}

    class Extractor:
        model_version = "test+cp01-pen-gate-v1"

        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def extract(self, video_path, checkpoint_id, time_range, **kwargs):
            observed["extract"] = (
                video_path, checkpoint_id, time_range, kwargs,
            )
            return ExtractedEvidence(
                features={
                    "stage_scan_reliable": True,
                    "pen_presence_detected": False,
                }
            )

    monkeypatch.setattr(gate, "Cp01PenGateExtractor", Extractor)
    result = gate.collect_cp01_pen_gate(
        Path("video.mp4"),
        TimeRange(start_sec=0, end_sec=14),
        tmp_path,
    )

    assert observed["init"] == {"evidence_root": tmp_path}
    assert observed["extract"][1] == "cp_01"
    assert result["features"]["pen_presence_detected"] is False
    assert result["objects"] == {
        "marking_pen": "opencv_dark_elongated_near_green_v2"
    }


def test_gate_objects_are_independent_and_text_only(tmp_path):
    gate = load_gate()

    class Backend:
        model_version = "test"

        def track(self, video_path, time_range, prompts, sample_fps):
            assert len(prompts) == 1 and prompts[0].kind == "text"
            yield FrameMasks(frame_index=0, frame_time_sec=1,
                             masks={prompts[0].object_id: np.ones((16, 16), bool)})

    result = gate.collect_gate(
        Backend(), Path("unused.mp4"), "cp_01", TimeRange(start_sec=0, end_sec=2),
        tmp_path, sample_fps=2,
        read_frame_fn=lambda *_: np.zeros((16, 16, 3), np.uint8),
    )
    assert set(result["objects"]) == {"template_board", "rubber_dam", "marking_pen"}
    assert result["visual_review_status"] == "pending"
    paths = list(tmp_path.rglob("*.png"))
    assert len(paths) >= 3
    assert all(cv2.imread(str(path)) is not None for path in paths)


def test_cp02_gate_catalog_includes_punch_cleanup_and_dam():
    gate = load_gate()
    assert set(gate.CATALOG["cp_02"]) == {"rubber_dam_punch", "cleaning_instrument", "rubber_dam"}


def test_cp02_gate_window_includes_last_adjustment_between_stages():
    from medical_evaluation.annotations import SegmentAnnotation, VideoAnnotations

    annotations = VideoAnnotations(video_id="failure", steps=[
        SegmentAnnotation(checkpoint_id="cp_02", time_range=TimeRange(start_sec=19, end_sec=50),
                          label="needs_review", reason="test"),
        SegmentAnnotation(checkpoint_id="cp_03", time_range=TimeRange(start_sec=55, end_sec=71),
                          label="needs_review", reason="test"),
    ])

    assert load_gate().resolve_gate_window(
        annotations, "cp_02", include_prepunch_gap=True,
    ) == TimeRange(start_sec=19, end_sec=55)


def test_extended_cp02_tracking_uses_original_stage_for_auto_box_seed(tmp_path, monkeypatch):
    gate = load_gate()
    observed = {}

    def fake_auto_box(**kwargs):
        observed.update(kwargs)
        return {"automatic_gate_passed": False}

    monkeypatch.setattr(gate, "_run_automatic_punch_box", fake_auto_box)
    original = TimeRange(start_sec=19, end_sec=50)
    extended = TimeRange(start_sec=19, end_sec=55)

    gate.collect_gate(
        SimpleNamespace(model_version="test"), Path("unused.mp4"), "cp_02", extended, tmp_path,
        automatic_punch_only=True, localization_time_range=original,
    )

    assert observed["time_range"] == extended
    assert observed["localization_time_range"] == original


def test_cp03_gate_needs_only_dam():
    assert set(load_gate().CATALOG["cp_03"]) == {"rubber_dam"}


def test_nonempty_output_is_not_overwritten(tmp_path):
    gate = load_gate()
    (tmp_path / "user.txt").write_text("keep")
    import pytest
    with pytest.raises(FileExistsError):
        gate.collect_gate(None, Path("unused"), "cp_01", TimeRange(start_sec=0, end_sec=2),
                          tmp_path, sample_fps=2)
    assert (tmp_path / "user.txt").read_text() == "keep"


def test_ambiguous_prompt_is_audited_without_aborting_other_objects(tmp_path):
    gate = load_gate()

    class Backend:
        model_version = "test"

        def track(self, video_path, time_range, prompts, sample_fps):
            if prompts[0].text == gate.CATALOG["cp_01"]["template_board"][1]:
                raise gate.Sam3AmbiguousTextResult("multiple candidates")
            yield FrameMasks(frame_index=0, frame_time_sec=1,
                             masks={prompts[0].object_id: np.ones((16, 16), bool)})

    result = gate.collect_gate(
        Backend(), Path("unused.mp4"), "cp_01", TimeRange(start_sec=0, end_sec=2),
        tmp_path, sample_fps=2,
        read_frame_fn=lambda *_: np.zeros((16, 16, 3), np.uint8),
    )
    failed = result["objects"]["template_board"][1]
    assert failed["status"] == "prompt_failed"
    assert failed["error_type"] == "Sam3AmbiguousTextResult"
    assert result["objects"]["marking_pen"]


def test_cp01_dam_catalog_includes_flat_sheet_appearance():
    prompts = load_gate().CATALOG["cp_01"]["rubber_dam"]
    assert "flat green rubber sheet" in prompts
    assert all("dental" not in prompt and "rubber dam" not in prompt for prompt in prompts)


def test_cp02_cp03_dam_prompts_describe_visible_green_sheet():
    catalog = load_gate().CATALOG
    assert "large green sheet" in catalog["cp_02"]["rubber_dam"]
    assert "large green sheet with a small hole" in catalog["cp_03"]["rubber_dam"]


def test_cp01_prompts_describe_visible_template_and_pen_appearance():
    catalog = load_gate().CATALOG["cp_01"]
    assert "white rectangular card with a black cross and rows of black dots" in (
        catalog["template_board"]
    )
    assert "thin black pen touching the green sheet" in catalog["marking_pen"]


def test_cp02_punch_prompt_distinguishes_held_tool_with_hole_disk():
    prompts = load_gate().CATALOG["cp_02"]["rubber_dam_punch"]
    assert "metal pliers held in a hand with a round disk containing several holes" in prompts
    assert "small round metal disk with several holes held in a gloved hand" in prompts


def test_visible_appearance_rejects_green_sheet_as_white_board():
    gate = load_gate()
    mask = np.ones((20, 20), dtype=bool)
    green = np.full((20, 20, 3), (30, 180, 30), dtype=np.uint8)
    white = np.full((20, 20, 3), (220, 220, 220), dtype=np.uint8)

    assert gate.visible_appearance_is_plausible("template_board", green, mask) is False
    assert gate.visible_appearance_is_plausible("template_board", white, mask) is True


def test_visible_appearance_rejects_bright_metal_as_black_pen():
    gate = load_gate()
    mask = np.ones((20, 20), dtype=bool)
    metal = np.full((20, 20, 3), (190, 190, 190), dtype=np.uint8)
    black = np.full((20, 20, 3), (35, 35, 35), dtype=np.uint8)

    assert gate.visible_appearance_is_plausible("marking_pen", metal, mask) is False
    assert gate.visible_appearance_is_plausible("marking_pen", black, mask) is True


def test_cp02_uses_automatic_box_when_text_prompts_miss(tmp_path):
    gate = load_gate()
    frames = []
    for index in range(5):
        image = np.full((360, 640, 3), (170, 120, 70), np.uint8)
        center = (120 + 35 * index, 250)
        cv2.line(
            image,
            (center[0] + 15, center[1] + 5),
            (center[0] + 115, center[1] + 65),
            (190, 190, 190),
            18,
        )
        cv2.circle(image, center, 30, (185, 185, 185), -1)
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            point = (center[0] + int(16 * np.cos(angle)),
                     center[1] + int(16 * np.sin(angle)))
            cv2.circle(image, point, 4, (20, 20, 20), -1)
        frames.append(SimpleNamespace(frame_index=index, time_sec=float(index), image_bgr=image))

    class Backend:
        model_version = "test"

        def __init__(self):
            self.box_prompt = None

        def track(self, video_path, time_range, prompts, sample_fps):
            if prompts[0].kind == "text":
                return iter(())
            self.box_prompt = prompts[0]
            mask = np.zeros((360, 640), bool)
            frame_index = round(prompts[0].frame_time_sec)
            center = (120 + 35 * frame_index, 250)
            cv2.line(
                mask.view(np.uint8),
                (center[0] + 15, center[1] + 5),
                (center[0] + 115, center[1] + 65),
                1,
                18,
            )
            cv2.circle(mask.view(np.uint8), center, 30, 1, -1)
            return iter([FrameMasks(
                frame_index=frame_index,
                frame_time_sec=prompts[0].frame_time_sec,
                masks={
                "rubber_dam_punch": mask,
                },
            )])

    backend = Backend()
    result = gate.collect_gate(
        backend, Path("unused.mp4"), "cp_02", TimeRange(start_sec=0, end_sec=4),
        tmp_path, sample_fps=1,
        read_frame_fn=lambda _path, index: frames[index].image_bgr,
        sample_frames_fn=lambda *args, **kwargs: iter(frames),
    )

    automatic = result["objects"]["rubber_dam_punch"][-1]
    assert automatic["prompt_kind"] == "box"
    assert automatic["prompt"] == "automatic held punch box"
    assert automatic["locator"]["motion_ratio"] >= 0.15
    assert automatic["locator"]["surface_contrast"] > 30
    assert automatic["tool_box"] is not None
    assert automatic["accepted_frame_count"] == 1
    assert automatic["frames"][0]["semantic_features"]["reason"] == (
        "criteria_satisfied"
    )
    assert backend.box_prompt is not None
    assert backend.box_prompt.kind == "box"


def test_cp02_rejects_nonempty_green_sam3_mask(tmp_path):
    gate = load_gate()
    frames = []
    for index in range(5):
        image = np.full((360, 640, 3), (30, 170, 30), np.uint8)
        center = (120 + 35 * index, 250)
        cv2.line(
            image,
            (center[0] + 15, center[1] + 5),
            (center[0] + 115, center[1] + 65),
            (190, 190, 190),
            18,
        )
        cv2.circle(image, center, 30, (185, 185, 185), -1)
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            point = (center[0] + int(16 * np.cos(angle)),
                     center[1] + int(16 * np.sin(angle)))
            cv2.circle(image, point, 4, (20, 20, 20), -1)
        frames.append(SimpleNamespace(frame_index=index, time_sec=float(index), image_bgr=image))

    class Backend:
        def track(self, _video_path, _time_range, prompts, _sample_fps):
            box = prompts[0].coordinates
            mask = np.zeros((360, 640), bool)
            left, top = round(box[0] * 640), round(box[1] * 360)
            right, bottom = round(box[2] * 640), round(box[3] * 360)
            mask[top:bottom, left:right] = True
            frame_index = round(prompts[0].frame_time_sec)
            return iter([FrameMasks(
                frame_index=frame_index,
                frame_time_sec=prompts[0].frame_time_sec,
                masks={"rubber_dam_punch": mask},
            )])

    backend = Backend()
    result = gate._run_automatic_punch_box(
        backend=backend,
        video_path=Path("unused.mp4"),
        time_range=TimeRange(start_sec=0, end_sec=4),
        output_dir=tmp_path,
        sample_fps=1,
        read_frame_fn=lambda _path, index: frames[index].image_bgr,
        sample_frames_fn=lambda *args, **kwargs: iter(frames),
    )

    assert result["accepted_frame_count"] == 0
    assert result["frames"][0]["mask_area_px"] > 0
    assert result["frames"][0]["semantic_features"]["reason"] == "green_sheet_mask"
    assert result["automatic_gate_passed"] is False


def test_cp02_gate_accepts_two_moved_disk_frames(tmp_path, monkeypatch):
    gate = load_gate()
    from medical_evaluation.features.cp02_punch import HeldPunchBox, MovingDisk

    images = []
    masks = []
    for index in range(4):
        image = np.full((360, 640, 3), (170, 120, 70), np.uint8)
        center = (120 + 65 * index, 220)
        cv2.circle(image, center, 32, (190, 190, 190), -1)
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            point = (center[0] + int(17 * np.cos(angle)),
                     center[1] + int(17 * np.sin(angle)))
            cv2.circle(image, point, 4, (20, 20, 20), -1)
        mask = np.zeros((360, 640), bool)
        if index >= 2:
            cv2.circle(mask.view(np.uint8), center, 32, 1, -1)
        images.append(SimpleNamespace(frame_index=index, time_sec=float(index), image_bgr=image))
        masks.append(mask)

    monkeypatch.setattr(
        gate, "locate_moving_multihole_disk",
        lambda _frames: MovingDisk(0, 120, 220, 32, 6, 0.6),
    )
    monkeypatch.setattr(
        gate, "locate_held_punch_box",
        lambda _frames, _disk: HeldPunchBox(0, 88, 188, 190, 285, 1.6, 0.6),
    )

    class Backend:
        def track(self, _video_path, _time_range, _prompts, _sample_fps):
            return iter(FrameMasks(
                frame_index=index, frame_time_sec=float(index),
                masks={"rubber_dam_punch": masks[index]},
            ) for index in range(4))

    result = gate._run_automatic_punch_box(
        backend=Backend(), video_path=Path("unused.mp4"),
        time_range=TimeRange(start_sec=0, end_sec=3), output_dir=tmp_path,
        sample_fps=1,
        read_frame_fn=lambda _path, index: images[index].image_bgr,
        sample_frames_fn=lambda *args, **kwargs: iter(images),
    )

    assert result["accepted_frame_count"] == 2
    assert result["max_consecutive_accepted_frames"] == 2
    assert result["automatic_gate_passed"] is True


def test_cp02_automatic_only_mode_skips_all_text_prompts(tmp_path):
    gate = load_gate()
    frames = []
    for index in range(5):
        image = np.full((360, 640, 3), (170, 120, 70), np.uint8)
        center = (120 + 35 * index, 250)
        cv2.line(
            image,
            (center[0] + 15, center[1] + 5),
            (center[0] + 115, center[1] + 65),
            (190, 190, 190),
            18,
        )
        cv2.circle(image, center, 30, (185, 185, 185), -1)
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            point = (
                center[0] + int(16 * np.cos(angle)),
                center[1] + int(16 * np.sin(angle)),
            )
            cv2.circle(image, point, 4, (20, 20, 20), -1)
        frames.append(
            SimpleNamespace(frame_index=index, time_sec=float(index), image_bgr=image)
        )

    class Backend:
        model_version = "test"

        def __init__(self):
            self.prompt_kinds = []

        def track(self, _video_path, _time_range, prompts, _sample_fps):
            self.prompt_kinds.append(prompts[0].kind)
            mask = np.zeros((360, 640), bool)
            frame_index = round(prompts[0].frame_time_sec)
            center = (120 + 35 * frame_index, 250)
            cv2.line(
                mask.view(np.uint8),
                (center[0] + 15, center[1] + 5),
                (center[0] + 115, center[1] + 65),
                1,
                18,
            )
            cv2.circle(mask.view(np.uint8), center, 30, 1, -1)
            return iter([FrameMasks(
                frame_index=frame_index,
                frame_time_sec=prompts[0].frame_time_sec,
                masks={"rubber_dam_punch": mask},
            )])

    backend = Backend()
    result = gate.collect_gate(
        backend,
        Path("unused.mp4"),
        "cp_02",
        TimeRange(start_sec=0, end_sec=4),
        tmp_path,
        sample_fps=1,
        read_frame_fn=lambda _path, index: frames[index].image_bgr,
        sample_frames_fn=lambda *args, **kwargs: iter(frames),
        automatic_punch_only=True,
    )

    assert backend.prompt_kinds == ["box"]
    assert set(result["objects"]) == {"rubber_dam_punch"}
    assert len(result["objects"]["rubber_dam_punch"]) == 1
