from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from medical_evaluation.annotations import SegmentAnnotation, VideoAnnotations
from medical_evaluation.domain import CheckpointStatus, TimeRange
from medical_evaluation.segmentation.base import FrameMasks

SCRIPT = Path(__file__).parents[2] / "scripts" / "run_sam3_prompt_matrix.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_sam3_prompt_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mask(*, dominant: int = 12, fragmented: int = 0) -> np.ndarray:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:5] = True  # 12 connected pixels
    for index in range(fragmented):
        mask[6, index] = True
    if dominant == 0:
        mask[:] = False
    return mask


def _row(module, *, sample: str, valid: bool, score: float | None = None) -> dict[str, object]:
    return {
        "sample_key": sample,
        "video_id": sample.split(":")[0],
        "stage": sample.split(":")[1],
        "valid": valid,
        "score": score,
    }


def test_dominant_component_validation_requires_nonempty_major_component() -> None:
    module = _load_script()

    assert module.mask_measurements(_mask()) == (12 / 100, 1.0, True)
    assert module.mask_measurements(_mask(fragmented=9)) == (21 / 100, 12 / 21, False)
    assert module.mask_measurements(_mask(dominant=0)) == (0.0, 0.0, False)


def test_candidate_passes_only_for_three_consecutive_valid_rows() -> None:
    module = _load_script()

    assert module.candidate_passes([_row(module, sample="success:cp_09", valid=True)] * 3)
    assert not module.candidate_passes(
        [
            _row(module, sample="success:cp_09", valid=True),
            _row(module, sample="success:cp_09", valid=False),
            _row(module, sample="success:cp_09", valid=True),
            _row(module, sample="success:cp_09", valid=True),
        ]
    )


def test_short_annotated_range_requests_the_end_frame_for_three_source_samples() -> None:
    module = _load_script()

    assert module.required_prompt_time(TimeRange(start_sec=303, end_sec=304)) == 304


def test_help_documents_the_server_matrix_command() -> None:
    module = _load_script()

    help_text = module.build_parser().format_help()

    assert "PYTHONPATH=\"$PWD/src\"" in help_text
    assert "--video-ids success failure clamp_failure" in help_text
    assert "--probe-multiplex" in help_text


def test_rank_candidates_uses_sample_continuity_then_median_score_then_order() -> None:
    module = _load_script()
    candidates = ("first", "second", "third")
    rows = {
        "first": [
            *[_row(module, sample="success:cp_09", valid=True, score=0.2) for _ in range(3)],
            *[_row(module, sample="success:cp_11", valid=True, score=0.2) for _ in range(3)],
        ],
        "second": [
            *[_row(module, sample="success:cp_09", valid=True, score=0.9) for _ in range(3)],
            *[_row(module, sample="success:cp_11", valid=True, score=0.9) for _ in range(3)],
        ],
        "third": [
            *[_row(module, sample="success:cp_09", valid=True, score=0.99) for _ in range(3)],
        ],
    }

    assert module.rank_candidates(rows, candidates) == ["second", "first", "third"]


def test_frame_score_adapter_reads_optional_object_score_metadata() -> None:
    module = _load_script()
    frame = SimpleNamespace(metadata={"scores": {"head": 0.82}})

    assert module.extract_frame_score(frame, "head") == 0.82


def test_required_samples_remain_global_when_cli_omits_video_ids() -> None:
    module = _load_script()

    required, missing = module.required_samples(("success",))
    success_only = {
        "head": [
            _row(module, sample=sample, valid=True)
            for sample in ("success:cp_09", "success:cp_11")
            for _ in range(3)
        ],
        "nose": [
            _row(module, sample="success:cp_11", valid=True) for _ in range(3)
        ],
    }

    assert missing == ("failure", "clamp_failure")
    assert "failure:cp_09" in required["head"]
    assert not module.select_prompts(
        success_only, required, ("head",), ("nose",)
    )["accepted"]


def test_acceptance_requires_every_video_and_required_stage() -> None:
    module = _load_script()
    required = {
        "head": (
            "success:cp_09", "success:cp_11", "failure:cp_09", "failure:cp_11",
            "clamp_failure:cp_09", "clamp_failure:cp_11",
        ),
        "nose": ("success:cp_11", "failure:cp_11", "clamp_failure:cp_11"),
    }
    complete = {
        "head-good": [
            _row(module, sample=sample, valid=True) for sample in required["head"] for _ in range(3)
        ],
        "nose-good": [
            _row(module, sample=sample, valid=True) for sample in required["nose"] for _ in range(3)
        ],
    }

    selected = module.select_prompts(complete, required, ("head-good",), ("nose-good",))
    assert selected["accepted"] is True
    assert selected["head_prompt"] == "head-good"
    assert selected["nose_prompt"] == "nose-good"

    complete["head-good"] = complete["head-good"][:-3]
    assert not module.select_prompts(complete, required, ("head-good",), ("nose-good",))["accepted"]


def test_run_writes_artifacts_and_returns_zero_for_accepted_matrix(tmp_path: Path) -> None:
    module = _load_script()
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for video_id in ("success", "failure", "clamp_failure"):
        (annotations / f"{video_id}.json").write_text(
            VideoAnnotations(
                video_id=video_id,
                steps=[
                    SegmentAnnotation(checkpoint_id="cp_09", time_range=TimeRange(start_sec=1, end_sec=3), label=CheckpointStatus.CORRECT, reason="ok"),
                    SegmentAnnotation(checkpoint_id="cp_11", time_range=TimeRange(start_sec=4, end_sec=8), label=CheckpointStatus.CORRECT, reason="ok"),
                ],
            ).model_dump_json(), encoding="utf-8"
        )
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in module.PRESETS.values():
        (videos / name).touch()

    class FakeBackend:
        model_version = "sam3.1:test"

        def track(self, _path, time_range, prompts, sample_fps):
            object_id = prompts[0].object_id
            for index in range(3):
                yield FrameMasks(frame_index=index + 10, frame_time_sec=time_range.start_sec + index / sample_fps, masks={object_id: _mask()})

    output = tmp_path / "output"
    result = module.run_gate(
        annotations_dir=annotations,
        videos_dir=videos,
        video_ids=("success", "failure", "clamp_failure"),
        output_dir=output,
        sample_fps=2,
        output_threshold=0.2,
        backend_factory=lambda: FakeBackend(),
        read_frame_fn=lambda _path, _index: np.zeros((10, 10, 3), dtype=np.uint8),
        git_revision_fn=lambda: "test-revision",
        cuda_peak_mib_fn=lambda: 12.5,
    )

    assert result == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    selected = json.loads((output / "selected_prompts.json").read_text(encoding="utf-8"))
    assert summary["model_version"] == "sam3.1:test"
    assert summary["git_revision"] == "test-revision"
    assert summary["cuda_peak_allocated_mib"] == 12.5
    assert selected["accepted"] is True
    artifact_dir = output / "overlays" / "success" / "head" / "dental-training-mannequin-head"
    assert (artifact_dir / "raw" / "00000010.jpg").is_file()
    assert (artifact_dir / "masks" / "00000010.png").is_file()
    assert (artifact_dir / "overlays" / "00000010.png").is_file()


def test_run_returns_two_when_required_prompt_is_not_continuous(tmp_path: Path) -> None:
    module = _load_script()
    rows = {candidate: [] for candidate in (*module.HEAD_PROMPTS, *module.NOSE_PROMPTS)}
    assert module.exit_status(module.select_prompts(rows, {"head": ("success:cp_09",), "nose": ("success:cp_11",)}, module.HEAD_PROMPTS, module.NOSE_PROMPTS)) == 2


def test_multiplex_probe_marks_exception_unsupported() -> None:
    module = _load_script()

    result = module.handle_multiplex_probe(lambda: (_ for _ in ()).throw(RuntimeError("unsupported signature")))

    assert result == {"supported": False, "result": "RuntimeError: unsupported signature"}


def test_multiplex_probe_marks_three_frame_distinct_identity_persistence_supported() -> None:
    module = _load_script()

    result = module.handle_multiplex_probe(
        lambda: {
            "source_frames": [
                {"source_frame_index": 0, "object_ids": {1, 2}},
                {"source_frame_index": 1, "object_ids": {1, 2}},
                {"source_frame_index": 2, "object_ids": {1, 2}},
            ]
        }
    )

    assert result == {"supported": True, "result": "two distinct object IDs persisted for 3 consecutive source frames"}


def test_multiplex_probe_rejects_duplicate_skipped_changing_merged_and_missing_ids() -> None:
    module = _load_script()
    invalid_sequences = (
        [
            {"source_frame_index": 0, "object_ids": {1, 2}},
            {"source_frame_index": 0, "object_ids": {1, 2}},
            {"source_frame_index": 1, "object_ids": {1, 2}},
        ],
        [
            {"source_frame_index": 0, "object_ids": {1, 2}},
            {"source_frame_index": 2, "object_ids": {1, 2}},
            {"source_frame_index": 3, "object_ids": {1, 2}},
        ],
        [
            {"source_frame_index": 0, "object_ids": {1, 2}},
            {"source_frame_index": 1, "object_ids": {1, 3}},
            {"source_frame_index": 2, "object_ids": {1, 2}},
        ],
        [
            {"source_frame_index": 0, "object_ids": {1, 2}},
            {"source_frame_index": 1, "object_ids": {1}},
            {"source_frame_index": 2, "object_ids": {1, 2}},
        ],
        [
            {"source_frame_index": 0, "object_ids": set()},
            {"source_frame_index": 1, "object_ids": {1, 2}},
            {"source_frame_index": 2, "object_ids": {1, 2}},
        ],
    )

    for source_frames in invalid_sequences:
        result = module.handle_multiplex_probe(
            lambda source_frames=source_frames: {"source_frames": source_frames}
        )
        assert result == {"supported": False, "result": "merged or missing object identities"}
