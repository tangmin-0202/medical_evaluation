import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from medical_evaluation.annotations import (
    BoxPrompt,
    PointPrompt,
    SegmentAnnotation,
    VideoAnnotations,
)
from medical_evaluation.domain import CheckpointStatus, TimeRange
from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor
from medical_evaluation.features.cp08_positioning import (
    ClampWingSplit,
    InstrumentShapeMeasurement,
    WingHoleColorMeasurement,
)
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.segmentation.base import FrameMasks

INSTRUMENT_PROMPT = "metal dental instrument with a long handle and curved working shaft"
TOOTH_PROMPT = "tooth"
CLAMP_PROMPT = "metal rubber dam clamp around the tooth"
DAM_PROMPT = "large green sheet covering the mouth area"


def _annotations() -> VideoAnnotations:
    return VideoAnnotations(
        video_id="sample",
        steps=[
            SegmentAnnotation(
                checkpoint_id="cp_08",
                time_range=TimeRange(start_sec=0, end_sec=10),
                label=CheckpointStatus.NEEDS_REVIEW,
                reason="test",
            ),
            SegmentAnnotation(
                checkpoint_id="cp_09",
                time_range=TimeRange(start_sec=16, end_sec=20),
                label=CheckpointStatus.NEEDS_REVIEW,
                reason="test",
            ),
        ],
        prompts=[
            PointPrompt(
                video_id="sample",
                frame_time_sec=4,
                object_id="instrument",
                x=0.2,
                y=0.3,
            ),
            BoxPrompt(
                video_id="sample",
                frame_time_sec=18,
                object_id="clamp",
                x1=0.1,
                y1=0.1,
                x2=0.9,
                y2=0.9,
            ),
        ],
    )


def _mask(kind: str) -> np.ndarray:
    mask = np.zeros((48, 64), dtype=bool)
    if kind == "instrument":
        mask[20:25, 5:50] = True
        mask[14:23, 45:54] = True
        mask[35:40, 5:10] = True
    elif kind == "tooth":
        mask[18:31, 27:37] = True
    elif kind == "clamp":
        mask[17:32, 10:54] = True
        mask[20:29, 25:39] = False
    elif kind == "dam":
        mask[8:43, 5:59] = True
    return mask


def test_text_prompts_use_the_requested_nonzero_stage_start(tmp_path: Path) -> None:
    annotations = VideoAnnotations(
        video_id="sample",
        steps=[
            SegmentAnnotation(
                checkpoint_id="cp_08",
                time_range=TimeRange(start_sec=10, end_sec=20),
                label=CheckpointStatus.NEEDS_REVIEW,
                reason="test",
            ),
            SegmentAnnotation(
                checkpoint_id="cp_09",
                time_range=TimeRange(start_sec=30, end_sec=40),
                label=CheckpointStatus.NEEDS_REVIEW,
                reason="test",
            ),
        ],
        prompts=[],
    )
    segmenter = RangeCheckingSegmenter()
    extractor = Cp08FeatureExtractor(
        segmenter=segmenter,
        annotations=annotations,
        evidence_root=tmp_path,
    )

    extractor.extract(
        tmp_path / "unused.mp4",
        "cp_08",
        TimeRange(start_sec=10, end_sec=20),
        dense_fps=5,
        analysis_width=1280,
    )

    assert [(call[0].start_sec, call[1]) for call in segmenter.calls] == [
        (10, 10),
        (37, 37),
        (37, 37),
        (37, 37),
    ]


class FakeSegmenter:
    model_version = "fake-sam3"

    def __init__(self, *, candidate: bool = True) -> None:
        self.candidate = candidate
        self.calls: list[tuple[TimeRange, str, str, float]] = []

    def track(self, _video_path, time_range, prompts, sample_fps):
        assert len(prompts) == 1
        prompt = prompts[0]
        assert prompt.kind == "text"
        assert prompt.coordinates is None
        self.calls.append((time_range, prompt.object_id, prompt.text or "", sample_fps))
        if prompt.object_id == "cp08_instrument":
            if sample_fps == 1.0:
                for index, time_sec in enumerate((0.0, 5.0, 10.0)):
                    value = _mask("instrument") if self.candidate and time_sec == 5.0 else _mask("")
                    yield FrameMasks(
                        frame_index=index * 50,
                        frame_time_sec=time_sec,
                        masks={"cp08_instrument": value},
                    )
            elif self.candidate:
                for index, time_sec in enumerate((4.8, 5.0, 5.2), start=48):
                    yield FrameMasks(
                        frame_index=index,
                        frame_time_sec=time_sec,
                        masks={"cp08_instrument": _mask("instrument")},
                    )
            return
        kind = {
            "cp08_target_tooth": "tooth",
            "cp08_full_clamp": "clamp",
            "cp08_rubber_dam": "dam",
        }[prompt.object_id]
        for index, time_sec in enumerate((17.5, 18.0, 18.5), start=175):
            yield FrameMasks(
                frame_index=index,
                frame_time_sec=time_sec,
                masks={prompt.object_id: _mask(kind)},
            )


class RangeCheckingSegmenter:
    model_version = "range-checking-sam3"

    def __init__(self) -> None:
        self.calls: list[tuple[TimeRange, float]] = []

    def track(self, _video_path, time_range, prompts, sample_fps):
        assert sample_fps > 0
        assert len(prompts) == 1
        prompt_time = prompts[0].frame_time_sec
        assert time_range.start_sec <= prompt_time <= time_range.end_sec
        self.calls.append((time_range, prompt_time))
        return iter(())


class ShapeSelectionSegmenter(FakeSegmenter):
    def __init__(self, *, include_plausible: bool = True) -> None:
        super().__init__()
        self.include_plausible = include_plausible

    def track(self, video_path, time_range, prompts, sample_fps):
        prompt = prompts[0]
        if prompt.object_id != "cp08_instrument":
            yield from super().track(video_path, time_range, prompts, sample_fps)
            return
        self.calls.append((time_range, prompt.object_id, prompt.text or "", sample_fps))
        if sample_fps == 1.0:
            compact = np.zeros((48, 64), dtype=bool)
            compact[4:9, 4:9] = True
            yield FrameMasks(
                frame_index=20, frame_time_sec=2.0,
                masks={"cp08_instrument": compact},
            )
            if self.include_plausible:
                yield FrameMasks(
                    frame_index=70, frame_time_sec=7.0,
                    masks={"cp08_instrument": _mask("instrument")},
                )
            return
        for index, time_sec in enumerate((6.8, 7.0, 7.2), start=68):
            yield FrameMasks(
                frame_index=index, frame_time_sec=time_sec,
                masks={"cp08_instrument": _mask("instrument")},
            )


class StraightCandidateSegmenter(ShapeSelectionSegmenter):
    pass


class DamConsensusSegmenter(FakeSegmenter):
    def __init__(self, dam_present_count: int, *, total_count: int = 10) -> None:
        super().__init__()
        self.dam_present_count = dam_present_count
        self.total_count = total_count

    def track(self, video_path, time_range, prompts, sample_fps):
        prompt = prompts[0]
        if prompt.object_id not in {
            "cp08_target_tooth", "cp08_full_clamp", "cp08_rubber_dam"
        }:
            yield from super().track(video_path, time_range, prompts, sample_fps)
            return
        self.calls.append((time_range, prompt.object_id, prompt.text or "", sample_fps))
        kind = {
            "cp08_target_tooth": "tooth",
            "cp08_full_clamp": "clamp",
            "cp08_rubber_dam": "dam",
        }[prompt.object_id]
        for offset in range(self.total_count):
            value = _mask(kind)
            if prompt.object_id == "cp08_rubber_dam" and offset >= self.dam_present_count:
                value = _mask("")
            yield FrameMasks(
                frame_index=200 + offset,
                frame_time_sec=17.0 + offset * 0.25,
                masks={prompt.object_id: value},
            )


class MultiCandidateSegmenter(FakeSegmenter):
    def __init__(self, *, later_matches: bool) -> None:
        super().__init__()
        self.later_matches = later_matches

    @staticmethod
    def straight() -> np.ndarray:
        mask = np.zeros((48, 64), dtype=bool)
        mask[20:24, 5:50] = True
        return mask

    def track(self, video_path, time_range, prompts, sample_fps):
        prompt = prompts[0]
        if prompt.object_id != "cp08_instrument":
            yield from super().track(video_path, time_range, prompts, sample_fps)
            return
        self.calls.append((time_range, prompt.object_id, prompt.text or "", sample_fps))
        if sample_fps == 1.0:
            yield FrameMasks(
                frame_index=20, frame_time_sec=2.0,
                masks={"cp08_instrument": self.straight()},
            )
            yield FrameMasks(
                frame_index=70, frame_time_sec=7.0,
                masks={
                    "cp08_instrument": (
                        _mask("instrument") if self.later_matches else self.straight()
                    )
                },
            )
            return
        value = _mask("instrument") if time_range.start_sec > 4 and self.later_matches else self.straight()
        center = 7.0 if time_range.start_sec > 4 else 2.0
        for offset, time_sec in enumerate((center - 0.2, center, center + 0.2)):
            yield FrameMasks(
                frame_index=round(time_sec * 10) + offset,
                frame_time_sec=time_sec,
                masks={"cp08_instrument": value},
            )


class CollisionSegmenter(FakeSegmenter):
    def track(self, video_path, time_range, prompts, sample_fps):
        prompt = prompts[0]
        if prompt.object_id != "cp08_instrument":
            yield from super().track(video_path, time_range, prompts, sample_fps)
            return
        self.calls.append((time_range, prompt.object_id, prompt.text or "", sample_fps))
        if sample_fps == 1.0:
            yield FrameMasks(
                frame_index=42, frame_time_sec=5.0,
                masks={"cp08_instrument": _mask("instrument")},
            )
            return
        for index, time_sec in ((42, 4.8), (43, 5.0), (44, 5.2)):
            dense = _mask("instrument").copy()
            dense[10:14, 10:14] = True
            yield FrameMasks(
                frame_index=index, frame_time_sec=time_sec,
                masks={"cp08_instrument": dense},
            )


class BoundaryCandidateSegmenter(FakeSegmenter):
    @staticmethod
    def straight() -> np.ndarray:
        mask = np.zeros((48, 64), dtype=bool)
        mask[20:24, 5:50] = True
        return mask

    def track(self, video_path, time_range, prompts, sample_fps):
        prompt = prompts[0]
        if prompt.object_id != "cp08_instrument":
            yield from super().track(video_path, time_range, prompts, sample_fps)
            return
        self.calls.append((time_range, prompt.object_id, prompt.text or "", sample_fps))
        if sample_fps == 1.0:
            yield FrameMasks(
                frame_index=20, frame_time_sec=2.0,
                masks={"cp08_instrument": self.straight()},
            )
            yield FrameMasks(
                frame_index=30, frame_time_sec=3.0,
                masks={"cp08_instrument": _mask("instrument")},
            )
            return
        matches = time_range.start_sec >= 2.0
        value = _mask("instrument") if matches else self.straight()
        center = 3.0 if matches else 2.0
        for offset, time_sec in enumerate((center - 0.2, center, center + 0.2)):
            yield FrameMasks(
                frame_index=round(time_sec * 10) + offset,
                frame_time_sec=time_sec,
                masks={"cp08_instrument": value},
            )


def _patch_measurements(monkeypatch) -> None:
    from medical_evaluation.extractors import cp08

    def instrument_measurement(mask):
        selected = mask.copy()
        selected[35:40, 5:10] = False
        return InstrumentShapeMeasurement(
            True, True, True, "shape_proxy_matched", 2, int(selected.sum()), 8.0, 2.0,
            25.0, selected,
        )

    monkeypatch.setattr(
        cp08,
        "measure_instrument_shape",
        instrument_measurement,
    )
    wing = np.zeros((48, 64), dtype=bool)
    wing[15:35, 8:24] = True
    monkeypatch.setattr(
        cp08,
        "split_clamp_wings",
        lambda _tooth, _clamp: ClampWingSplit(
            True, "wings_split", wing, np.fliplr(wing), (1.0, 0.0), (32.0, 24.0),
            True, True, 0.9, 0.9,
        ),
    )
    monkeypatch.setattr(
        cp08,
        "measure_wing_hole_color",
        lambda _frame, wing_mask, _dam: WingHoleColorMeasurement(
            True, True, "hole_color_measured", 0.95, 0.05, 30, 15, wing_mask, wing_mask,
        ),
    )
    monkeypatch.setattr(
        cp08,
        "read_frame",
        lambda _path, index: np.full((48, 64, 3), (30, 160, 60), np.uint8),
    )


def test_cross_stage_extractor_uses_text_only_sparse_dense_and_independent_tail_sessions(
    monkeypatch, tmp_path: Path,
) -> None:
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)
    segmenter = FakeSegmenter()
    result = Cp08FeatureExtractor(
        segmenter=segmenter,
        annotations=_annotations(),
        evidence_root=tmp_path / "evidence",
    ).extract(
        tmp_path / "unused.mp4",
        "cp_08",
        TimeRange(start_sec=0, end_sec=10),
        dense_fps=5,
        analysis_width=1280,
    )

    assert [(c[0].start_sec, c[0].end_sec, c[1], c[2], c[3]) for c in segmenter.calls] == [
        (0.0, 10.0, "cp08_instrument", INSTRUMENT_PROMPT, 1.0),
        (4.0, 6.0, "cp08_instrument", INSTRUMENT_PROMPT, 5),
        (17.0, 20.0, "cp08_target_tooth", TOOTH_PROMPT, 5),
        (17.0, 20.0, "cp08_full_clamp", CLAMP_PROMPT, 5),
        (17.0, 20.0, "cp08_rubber_dam", DAM_PROMPT, 5),
    ]
    assert all("blunt" not in call[2].lower() and "sharp" not in call[2].lower()
               for call in segmenter.calls)
    assert result.features == {
        "instrument_observed": True,
        "instrument_shape_reliable": True,
        "instrument_shape_match": True,
        "instrument_valid_frame_count": 3.0,
        "instrument_matching_frame_count": 3.0,
        "final_state_observable": True,
        "rubber_dam_positioned": True,
        "rubber_dam_segmentation_conflict": False,
        "dam_color_conflict_frame_count": 0.0,
        "dam_color_conflict_frame_ratio": 0.0,
        "final_state_valid_frame_count": 3.0,
        "left_wing_complete": True,
        "right_wing_complete": True,
        "left_wing_valid_frame_count": 3.0,
        "right_wing_valid_frame_count": 3.0,
        "left_wing_complete_ratio": 1.0,
        "right_wing_complete_ratio": 1.0,
        "left_wing_hole_detected": True,
        "right_wing_hole_detected": True,
        "left_wing_hole_valid_frame_count": 3.0,
        "right_wing_hole_valid_frame_count": 3.0,
        "left_wing_hole_dam_color_ratio": 0.95,
        "right_wing_hole_dam_color_ratio": 0.95,
        "left_wing_hole_non_dam_color_ratio": 0.05,
        "right_wing_hole_non_dam_color_ratio": 0.05,
    }


def test_dense_confirmation_uses_first_shape_plausible_not_first_nonempty_mask(
    monkeypatch, tmp_path: Path,
) -> None:
    from medical_evaluation.extractors import cp08
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)

    def measure(mask):
        plausible = int(mask.sum()) > 100
        return InstrumentShapeMeasurement(
            True, True, plausible, "shape_proxy_matched" if plausible else "compact_blob",
            1, int(mask.sum()), 8.0 if plausible else 1.0,
            2.0 if plausible else 1.0, 25.0 if plausible else 0.0,
            mask if plausible else None,
        )

    monkeypatch.setattr(cp08, "measure_instrument_shape", measure)
    segmenter = ShapeSelectionSegmenter()
    result = Cp08FeatureExtractor(
        segmenter=segmenter, annotations=_annotations(), evidence_root=tmp_path / "evidence",
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    assert (segmenter.calls[1][0].start_sec, segmenter.calls[1][0].end_sec) == (6.0, 8.0)
    assert result.features["instrument_observed"] is True
    assert result.features["instrument_shape_match"] is True


def test_nonempty_but_implausible_sparse_masks_do_not_trigger_dense_confirmation(
    monkeypatch, tmp_path: Path,
) -> None:
    from medical_evaluation.extractors import cp08
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)
    monkeypatch.setattr(
        cp08,
        "measure_instrument_shape",
        lambda mask: InstrumentShapeMeasurement(
            True, True, False, "compact_blob", 1, int(mask.sum()), 1.0, 1.0, 0.0, None,
        ),
    )
    segmenter = ShapeSelectionSegmenter(include_plausible=False)
    result = Cp08FeatureExtractor(
        segmenter=segmenter, annotations=_annotations(), evidence_root=tmp_path / "evidence",
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    assert len([call for call in segmenter.calls if call[1] == "cp08_instrument"]) == 1
    assert result.features["instrument_observed"] is True
    assert result.features["instrument_shape_reliable"] is False
    assert result.features["instrument_shape_match"] is None


def test_straight_elongated_candidate_reaches_dense_wrong_shape_decision(
    monkeypatch, tmp_path: Path,
) -> None:
    from medical_evaluation.extractors import cp08
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)

    def measure(mask):
        elongated = int(mask.sum()) > 100
        return InstrumentShapeMeasurement(
            True,
            elongated,
            False if elongated else None,
            "shape_proxy_not_matched" if elongated else "compact_blob",
            1,
            int(mask.sum()),
            8.0 if elongated else 1.0,
            1.2 if elongated else None,
            0.0 if elongated else None,
            mask if elongated else None,
        )

    monkeypatch.setattr(cp08, "measure_instrument_shape", measure)
    segmenter = StraightCandidateSegmenter()
    result = Cp08FeatureExtractor(
        segmenter=segmenter, annotations=_annotations(), evidence_root=tmp_path / "evidence",
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    assert (segmenter.calls[1][0].start_sec, segmenter.calls[1][0].end_sec) == (6.0, 8.0)
    assert result.features["instrument_shape_reliable"] is True
    assert result.features["instrument_shape_match"] is False


@pytest.mark.parametrize(("later_matches", "expected"), ((True, True), (False, False)))
def test_all_distinct_plausible_candidates_are_exhausted_until_positive_match(
    monkeypatch, tmp_path: Path, later_matches: bool, expected: bool,
) -> None:
    from medical_evaluation.extractors import cp08
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)

    def measure(mask):
        matches = int(mask.sum()) > 220
        return InstrumentShapeMeasurement(
            True, True, matches, "shape_proxy_matched" if matches else "shape_proxy_not_matched",
            1, int(mask.sum()), 8.0, 2.0 if matches else 1.2,
            25.0 if matches else 0.0, mask,
        )

    monkeypatch.setattr(cp08, "measure_instrument_shape", measure)
    segmenter = MultiCandidateSegmenter(later_matches=later_matches)
    root = tmp_path / f"evidence-{later_matches}"
    result = Cp08FeatureExtractor(
        segmenter=segmenter, annotations=_annotations(), evidence_root=root,
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    instrument_calls = [call for call in segmenter.calls if call[1] == "cp08_instrument"]
    assert [(call[0].start_sec, call[0].end_sec) for call in instrument_calls] == [
        (0.0, 10.0), (1.0, 3.0), (6.0, 8.0),
    ]
    assert result.features["instrument_shape_reliable"] is True
    assert result.features["instrument_shape_match"] is expected
    metrics = json.loads(
        (root / "cp_08" / "action" / "per_frame_metrics.json").read_text("utf-8")
    )
    assert {row["session_id"] for row in metrics["frames"] if row["scan_phase"] == "dense"} == {
        "dense-01", "dense-02"
    }


def test_action_artifacts_do_not_collide_and_overlapping_frames_decode_once(
    monkeypatch, tmp_path: Path,
) -> None:
    from medical_evaluation.extractors import cp08
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)
    reads: dict[int, int] = {}

    def frame(_path, index):
        reads[index] = reads.get(index, 0) + 1
        return np.full((48, 64, 3), (30, 160, 60), np.uint8)

    monkeypatch.setattr(cp08, "read_frame", frame)
    root = tmp_path / "evidence"
    Cp08FeatureExtractor(
        segmenter=CollisionSegmenter(), annotations=_annotations(), evidence_root=root,
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    metrics = json.loads(
        (root / "cp_08" / "action" / "per_frame_metrics.json").read_text("utf-8")
    )
    collided = [row for row in metrics["frames"] if row["frame_index"] == 42]
    assert len(collided) == 2
    assert len({row["overlay_path"] for row in collided}) == 2
    assert all((root / row["overlay_path"]).is_file() for row in collided)
    assert all(row["scan_phase"] in row["overlay_path"] for row in collided)
    assert all(row["session_id"] in row["overlay_path"] for row in collided)
    assert reads[42] == 1


def test_candidate_at_half_open_dense_window_end_gets_its_own_session(
    monkeypatch, tmp_path: Path,
) -> None:
    from medical_evaluation.extractors import cp08
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)

    def measure(mask):
        matches = int(mask.sum()) > 220
        return InstrumentShapeMeasurement(
            True, True, matches, "shape_proxy_matched" if matches else "shape_proxy_not_matched",
            1, int(mask.sum()), 8.0, 2.0 if matches else 1.2,
            25.0 if matches else 0.0, mask,
        )

    monkeypatch.setattr(cp08, "measure_instrument_shape", measure)
    segmenter = BoundaryCandidateSegmenter()
    root = tmp_path / "evidence"
    result = Cp08FeatureExtractor(
        segmenter=segmenter, annotations=_annotations(), evidence_root=root,
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    instrument_ranges = [
        (call[0].start_sec, call[0].end_sec)
        for call in segmenter.calls if call[1] == "cp08_instrument"
    ]
    assert instrument_ranges == [(0.0, 10.0), (1.0, 3.0), (2.0, 4.0)]
    assert result.features["instrument_shape_match"] is True
    metrics = json.loads(
        (root / "cp_08" / "action" / "per_frame_metrics.json").read_text("utf-8")
    )
    assert {row["session_id"] for row in metrics["frames"] if row["scan_phase"] == "dense"} == {
        "dense-01", "dense-02"
    }


@pytest.mark.parametrize(
    ("present_count", "green_frame", "expected"),
    (
        (10, True, True),
        (0, False, False),
        (0, True, None),
        (10, False, None),
        (4, True, None),
    ),
)
def test_rubber_dam_position_requires_sam_and_original_green_agreement(
    monkeypatch,
    tmp_path: Path,
    present_count: int,
    green_frame: bool,
    expected: bool | None,
) -> None:
    from medical_evaluation.extractors import cp08
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)
    color = (30, 160, 60) if green_frame else (120, 120, 120)
    monkeypatch.setattr(
        cp08,
        "read_frame",
        lambda _path, _index: np.full((48, 64, 3), color, np.uint8),
    )
    root = tmp_path / f"evidence-{present_count}-{green_frame}"
    result = Cp08FeatureExtractor(
        segmenter=DamConsensusSegmenter(present_count),
        annotations=_annotations(),
        evidence_root=root,
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    assert result.features["final_state_observable"] is True
    assert result.features["rubber_dam_positioned"] is expected
    metrics = json.loads(
        (root / "cp_08" / "final" / "per_frame_metrics.json").read_text("utf-8")
    )
    assert all(
        {
            "sam_dam_present", "local_green_ratio", "local_green_state",
            "dam_green_ratio", "dam_green_state", "dam_color_conflict",
        } <= row.keys()
        for row in metrics["frames"]
    )
    if expected is None and (present_count, green_frame) in ((0, True), (10, False)):
        assert "rubber_dam_color_conflict" in metrics["failure_reasons"]
        assert any(row["dam_color_conflict"] is True for row in metrics["frames"])


@pytest.mark.parametrize(
    ("total", "conflict_count", "expected", "unresolved_conflict"),
    ((15, 1, True, False), (10, 4, None, True)),
)
def test_dam_conflicts_are_aggregated_instead_of_any_single_frame_veto(
    monkeypatch,
    tmp_path: Path,
    total: int,
    conflict_count: int,
    expected: bool | None,
    unresolved_conflict: bool,
) -> None:
    from medical_evaluation.extractors import cp08
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)

    def frame(_path, index):
        offset = index - 200
        color = (120, 120, 120) if 0 <= offset < conflict_count else (30, 160, 60)
        return np.full((48, 64, 3), color, np.uint8)

    monkeypatch.setattr(cp08, "read_frame", frame)
    root = tmp_path / "evidence"
    result = Cp08FeatureExtractor(
        segmenter=DamConsensusSegmenter(total, total_count=total),
        annotations=_annotations(), evidence_root=root,
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    assert result.features["rubber_dam_positioned"] is expected
    assert result.features["rubber_dam_segmentation_conflict"] is unresolved_conflict
    assert result.features["dam_color_conflict_frame_count"] == float(conflict_count)
    metrics = json.loads(
        (root / "cp_08" / "final" / "per_frame_metrics.json").read_text("utf-8")
    )
    assert ("rubber_dam_color_conflict" in metrics["failure_reasons"]) is unresolved_conflict


def test_insufficient_dam_agreement_without_conflict_stays_generic_review(
    monkeypatch, tmp_path: Path,
) -> None:
    from medical_evaluation.extractors import cp08
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)

    def frame(_path, _index):
        image = np.full((48, 64, 3), (120, 120, 120), np.uint8)
        image[:, :5] = (30, 160, 60)
        return image

    monkeypatch.setattr(cp08, "read_frame", frame)
    root = tmp_path / "evidence"
    result = Cp08FeatureExtractor(
        segmenter=DamConsensusSegmenter(0),
        annotations=_annotations(), evidence_root=root,
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    assert result.features["rubber_dam_positioned"] is None
    assert result.features["rubber_dam_segmentation_conflict"] is False
    metrics = json.loads(
        (root / "cp_08" / "final" / "per_frame_metrics.json").read_text("utf-8")
    )
    assert "rubber_dam_color_conflict" not in metrics["failure_reasons"]
    assert "rubber_dam_temporal_evidence_unreliable" in metrics["failure_reasons"]


def test_evidence_contract_writes_originals_masks_overlays_metrics_prompts_and_reasons(
    monkeypatch, tmp_path: Path,
) -> None:
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)
    root = tmp_path / "evidence"
    result = Cp08FeatureExtractor(
        segmenter=FakeSegmenter(), annotations=_annotations(), evidence_root=root,
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    for group, objects in (
        ("action", ("instrument_raw", "instrument_selected")),
        (
            "final",
            (
                "tooth", "clamp", "dam", "left_wing", "right_wing",
                "left_hole", "right_hole", "left_sample", "right_sample",
            ),
        ),
    ):
        directory = root / "cp_08" / group
        assert list((directory / "originals").glob("*.jpg"))
        assert list((directory / "overlays").glob("*.jpg"))
        for object_name in objects:
            assert list((directory / "masks" / object_name).glob("*.png"))
        metrics = json.loads((directory / "per_frame_metrics.json").read_text("utf-8"))
        assert metrics["selected_prompts"]
        assert all("rejection_reason" in item for item in metrics["frames"])
        assert "failure_reason" in metrics
        if group == "action":
            sparse_times = [
                item["time_sec"] for item in metrics["frames"]
                if item["scan_phase"] == "sparse"
            ]
            assert min(sparse_times) == 0.0
            assert max(sparse_times) == 10.0
            assert any(item["scan_phase"] == "dense" for item in metrics["frames"])
            measured = next(
                item for item in metrics["frames"] if item["shape_reliable"] is True
            )
            assert {
                "component_count", "selected_component_area_px", "elongation",
                "handle_to_shaft_width_ratio", "curvature_deg", "measurement_reason",
            } <= measured.keys()
            selected_path = next(
                path for path in (directory / "masks" / "instrument_selected").glob("*.png")
                if cv2.countNonZero(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
            )
            raw_path = directory / "masks" / "instrument_raw" / selected_path.name
            assert cv2.countNonZero(cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)) > (
                cv2.countNonZero(cv2.imread(str(selected_path), cv2.IMREAD_GRAYSCALE))
            )
        else:
            measured = next(
                item for item in metrics["frames"]
                if item["left_hole_reliable"] is True
            )
            assert {
                "left_wing_complete", "right_wing_complete",
                "left_wing_valid", "right_wing_valid",
                "left_wing_completeness_score", "right_wing_completeness_score",
                "left_hole_detected", "right_hole_detected",
                "left_hole_reason", "right_hole_reason",
                "left_hole_dam_color_ratio", "right_hole_dam_color_ratio",
                "left_hole_non_dam_color_ratio", "right_hole_non_dam_color_ratio",
                "left_hole_area_px", "right_hole_area_px",
                "left_sample_area_px", "right_sample_area_px",
            } <= measured.keys()
    assert result.evidence
    assert all((root / item.overlay_path).is_file() for item in result.evidence)


def test_no_sparse_candidate_never_runs_dense_confirmation_and_preserves_unreliable_final(
    monkeypatch, tmp_path: Path,
) -> None:
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)
    segmenter = FakeSegmenter(candidate=False)
    result = Cp08FeatureExtractor(
        segmenter=segmenter, annotations=_annotations(), evidence_root=tmp_path / "evidence",
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    instrument_calls = [call for call in segmenter.calls if call[1] == "cp08_instrument"]
    assert len(instrument_calls) == 1
    assert (instrument_calls[0][0].start_sec, instrument_calls[0][0].end_sec) == (0, 10)
    assert result.features["instrument_observed"] is False
    assert result.features["instrument_shape_reliable"] is False
    assert result.features["instrument_shape_match"] is None
    metrics = json.loads(
        (tmp_path / "evidence" / "cp_08" / "action" / "per_frame_metrics.json")
        .read_text("utf-8")
    )
    assert metrics["failure_reason"] == "instrument_not_observed_in_full_sparse_scan"


def test_reliable_missing_wing_is_preserved_as_incomplete_not_unknown(
    monkeypatch, tmp_path: Path,
) -> None:
    from medical_evaluation.extractors import cp08
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)
    right = np.zeros((48, 64), dtype=bool)
    right[15:35, 40:56] = True
    monkeypatch.setattr(
        cp08,
        "split_clamp_wings",
        lambda _tooth, _clamp: ClampWingSplit(
            True, "wings_split", None, right, (1.0, 0.0), (32.0, 24.0),
            False, True, 0.0, 0.9,
        ),
    )

    result = Cp08FeatureExtractor(
        segmenter=FakeSegmenter(), annotations=_annotations(),
        evidence_root=tmp_path / "evidence",
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    assert result.features["left_wing_complete"] is False
    assert result.features["left_wing_valid_frame_count"] == 3.0
    assert result.features["right_wing_complete"] is True
    metrics = json.loads(
        (tmp_path / "evidence" / "cp_08" / "final" / "per_frame_metrics.json")
        .read_text("utf-8")
    )
    assert "left_wing_not_complete" in metrics["failure_reasons"]
    assert all("left_wing_missing" in item["rejection_reasons"] for item in metrics["frames"])


def test_unreliable_holes_have_explicit_group_and_frame_rejection_reasons(
    monkeypatch, tmp_path: Path,
) -> None:
    from medical_evaluation.extractors import cp08
    from medical_evaluation.extractors.cp08 import Cp08FeatureExtractor

    _patch_measurements(monkeypatch)
    monkeypatch.setattr(
        cp08,
        "measure_wing_hole_color",
        lambda *_args: WingHoleColorMeasurement(
            False, False, "hole_not_detected", None, None, 0, 0, None, None,
        ),
    )
    root = tmp_path / "evidence"
    Cp08FeatureExtractor(
        segmenter=FakeSegmenter(), annotations=_annotations(), evidence_root=root,
    ).extract(
        tmp_path / "unused.mp4", "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=5, analysis_width=1280,
    )

    metrics = json.loads(
        (root / "cp_08" / "final" / "per_frame_metrics.json").read_text("utf-8")
    )
    assert "left_wing_hole_unreliable" in metrics["failure_reasons"]
    assert "right_wing_hole_unreliable" in metrics["failure_reasons"]
    assert all("left_hole_not_detected" in row["rejection_reasons"] for row in metrics["frames"])
    assert all("right_hole_not_detected" in row["rejection_reasons"] for row in metrics["frames"])


class _Delegate:
    def __init__(self, version: str) -> None:
        self.model_version = version
        self.calls: list[str] = []

    def extract(self, _path, checkpoint_id, _range, **_kwargs):
        self.calls.append(checkpoint_id)
        return ExtractedEvidence(features={})


def test_composite_optional_cp08_delegate_changes_no_existing_dispatch() -> None:
    from medical_evaluation.extractors.cp09_cp11 import Cp09Cp11FeatureExtractor

    cp08 = _Delegate("cp08-v1")
    cp09 = _Delegate("cp09-v1")
    cp11 = _Delegate("cp11-v1")
    extractor = Cp09Cp11FeatureExtractor(
        cp08=cp08, cp09=cp09, cp11=cp11, annotations=_annotations(),
    )

    extractor.extract(
        Path("video.mp4"), "cp_08", TimeRange(start_sec=0, end_sec=10),
        dense_fps=2, analysis_width=1280,
    )

    assert cp08.calls == ["cp_08"]
    assert cp09.calls == []
    assert cp11.calls == []
    assert extractor.model_version == (
        "cp02=disabled;cp08=cp08-v1;cp09=cp09-v1;cp10=disabled;cp11=cp11-v1"
    )

    legacy = Cp09Cp11FeatureExtractor(
        cp09=cp09, cp11=cp11, annotations=_annotations(),
    )
    assert legacy.model_version == (
        "cp02=disabled;cp09=cp09-v1;cp10=disabled;cp11=cp11-v1"
    )
