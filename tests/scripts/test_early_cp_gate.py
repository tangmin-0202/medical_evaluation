import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import FrameMasks


def load_gate():
    scripts = Path(__file__).parents[2] / "scripts"
    spec = importlib.util.spec_from_file_location("early_gate", scripts / "run_early_cp_sam3_gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
