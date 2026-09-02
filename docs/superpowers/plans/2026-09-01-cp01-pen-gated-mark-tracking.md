# CP01 Pen-Gated Mark Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect whether a marking pen contacts the rubber dam, exclude pre-existing template marks, track newly added dark marks, and score the darkest one or two candidates by their minimum rubber-dam-local distance to the saved CP01 reference.

**Architecture:** SAM2 tracks `rubber_dam` and optional `marking_pen` over the full CP01 stage. Small deterministic feature modules compute stable pen contact and rubber-dam-local dark-mark tracks; the extractor separates pre-contact background from post-contact new or darkened marks, then hands explicit features to the deterministic Judge. Annotation guidance, rubric thresholds, calibration diagnostics, evidence overlays, and the combined CP01/CP11 smoke remain independently testable.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Pydantic, FastAPI, PyYAML, SAM2, pytest, Ruff.

---

## File map

- Create `src/medical_evaluation/features/contact.py`: generic mask-overlap measurements and stable-contact event detection.
- Modify `src/medical_evaluation/features/marks.py`: local darkness observations, pre-contact background matching, post-contact mark tracks, Top-2 nearest-reference selection.
- Modify `src/medical_evaluation/extractors/cp01.py`: optional pen prompt, dual-object tracking, temporal state assembly, CP01 evidence.
- Modify `src/medical_evaluation/judges/cp01_cp03.py`: contact-first CP01 state precedence.
- Modify `config/rubric.yaml`: CP01 objects and all contact/mark thresholds.
- Modify `src/medical_evaluation/web/routes.py`: `marking_pen` annotation guidance for reference and learner videos.
- Modify `scripts/calibrate_cp01_detector.py`: sweep temporal/contact thresholds and write candidate overlays.
- Modify `scripts/smoke_sam2_cp01_cp11.py`: inject CP01 thresholds from rubric and retain shared SAM2 loading.
- Create `tests/test_contact_features.py`: contact event unit tests.
- Modify `tests/test_cp01_mark_selection.py`: background exclusion, darkening, Top-2 and nearest-reference tests.
- Modify `tests/test_cp01_extractor.py`: missing pen, contact, temporal tracking and evidence contract.
- Modify `tests/test_cp01_judge.py`, `tests/test_judges_all.py`: new state precedence.
- Modify `tests/test_annotation_api.py`, `tests/test_rubric.py`: annotation and rubric contracts.
- Modify `tests/test_calibrate_cp01_detector.py`, `tests/test_smoke_cp01_cp11.py`: diagnostics and orchestration.
- Modify `docs/server-runbook.md`: reset/reannotation, calibration, GPU smoke and visual acceptance.

### Task 1: Update CP01 annotation and rubric contracts

**Files:**
- Modify: `tests/test_annotation_api.py`
- Modify: `tests/test_rubric.py`
- Modify: `src/medical_evaluation/web/routes.py`
- Modify: `config/rubric.yaml`

- [ ] **Step 1: Write failing annotation and rubric tests**

Change the reference and learner expectations to require a pen box when CP01 was performed:

```python
assert guides["cp_01"]["objects"] == ["rubber_dam", "marking_pen"]
assert reference_guides["cp_01"]["objects"] == [
    "rubber_dam",
    "marking_pen",
    "cp01_reference",
]

assert cp01.required_objects == ["rubber_dam", "marking_pen"]
assert cp01.thresholds == {
    "max_mark_distance": 0.05,
    "min_pen_dam_overlap_ratio": 0.02,
    "min_pen_contact_frames": 2.0,
    "min_mark_observed_frames": 3.0,
    "min_new_mark_darkness_delta": 15.0,
    "min_mark_area_ratio": 0.0001,
    "max_mark_area_ratio": 0.01,
    "maximum_black_value": 55.0,
    "maximum_faint_value": 160.0,
    "maximum_faint_saturation": 120.0,
    "max_mark_aspect_ratio": 2.0,
    "min_mark_circularity": 0.35,
    "max_local_cluster_distance": 0.04,
    "local_darkness_ring_radius": 5.0,
}
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/test_annotation_api.py tests/test_rubric.py -q
```

Expected: failures showing `marking_pen` and the new thresholds are absent.

- [ ] **Step 3: Implement the annotation and rubric contract**

Use these exact guide payloads:

```python
ANNOTATION_GUIDES["cp_01"] = {
    "objects": ["rubber_dam", "marking_pen"],
    "hint": "CP01已执行时：框选完整橡皮布，并在笔接触橡皮布的清晰帧紧框标记笔；未执行时不要伪造笔提示。",
}

CP01_REFERENCE_GUIDE = {
    "objects": ["rubber_dam", "marking_pen", "cp01_reference"],
    "hint": "基准视频：框选橡皮布和正在接触的标记笔，并点一次36牙固定正确位置。",
}
```

Add `marking_pen` and the tested numeric values under CP01 in `config/rubric.yaml`.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: both files pass.

- [ ] **Step 5: Commit**

```bash
git add config/rubric.yaml src/medical_evaluation/web/routes.py tests/test_annotation_api.py tests/test_rubric.py
git commit -m "feat: guide CP01 pen prompts"
```

### Task 2: Add stable pen-contact features

**Files:**
- Create: `src/medical_evaluation/features/contact.py`
- Create: `tests/test_contact_features.py`

- [ ] **Step 1: Write failing contact tests**

```python
def test_contact_requires_consecutive_overlap_frames() -> None:
    result = stable_contact_event(
        [
            MaskOverlap(frame_index=1, time_sec=0.5, ratio=0.03),
            MaskOverlap(frame_index=2, time_sec=1.0, ratio=0.00),
            MaskOverlap(frame_index=3, time_sec=1.5, ratio=0.04),
            MaskOverlap(frame_index=4, time_sec=2.0, ratio=0.05),
        ],
        minimum_ratio=0.02,
        minimum_consecutive_frames=2,
    )
    assert result.detected is True
    assert result.first_frame_index == 3
    assert result.contact_frame_count == 2


def test_pen_overlap_is_normalized_by_pen_area() -> None:
    dam = np.zeros((20, 20), dtype=bool)
    pen = np.zeros((20, 20), dtype=bool)
    pen[5:15, 5:15] = True
    dam[10:15, 5:15] = True
    assert mask_overlap_ratio(pen, dam) == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_contact_features.py -q
```

Expected: import failure because `features.contact` does not exist.

- [ ] **Step 3: Implement the minimal contact module**

```python
@dataclass(frozen=True)
class MaskOverlap:
    frame_index: int
    time_sec: float
    ratio: float


@dataclass(frozen=True)
class ContactEvent:
    detected: bool
    first_frame_index: int | None
    first_time_sec: float | None
    contact_frame_count: int
    maximum_ratio: float


def mask_overlap_ratio(subject: np.ndarray, target: np.ndarray) -> float:
    subject_mask = np.asarray(subject, dtype=bool)
    target_mask = np.asarray(target, dtype=bool)
    if subject_mask.shape != target_mask.shape:
        raise ValueError("overlap masks must have equal shapes")
    area = int(subject_mask.sum())
    return float((subject_mask & target_mask).sum() / area) if area else 0.0


def stable_contact_event(
    observations: Sequence[MaskOverlap],
    *,
    minimum_ratio: float,
    minimum_consecutive_frames: int,
) -> ContactEvent:
    ordered = sorted(observations, key=lambda item: item.frame_index)
    qualifying_count = sum(item.ratio >= minimum_ratio for item in ordered)
    maximum_ratio = max((item.ratio for item in ordered), default=0.0)
    run: list[MaskOverlap] = []
    first: MaskOverlap | None = None

    for item in ordered:
        if item.ratio < minimum_ratio:
            run.clear()
            continue
        run.append(item)
        if len(run) >= minimum_consecutive_frames:
            first = run[0]
            break

    return ContactEvent(
        detected=first is not None,
        first_frame_index=first.frame_index if first else None,
        first_time_sec=first.time_sec if first else None,
        contact_frame_count=qualifying_count,
        maximum_ratio=maximum_ratio,
    )
```

The input contains one observation for every sampled frame, including ratio zero
when a mask is absent. Therefore consecutive list entries represent consecutive
sampled frames; the scan never assumes original video frame numbers differ by one.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: all contact tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/medical_evaluation/features/contact.py tests/test_contact_features.py
git commit -m "feat: detect stable pen contact"
```

### Task 3: Track local darkness changes and select Top 2 nearest reference

**Files:**
- Modify: `src/medical_evaluation/features/marks.py`
- Modify: `tests/test_cp01_mark_selection.py`

- [ ] **Step 1: Write failing temporal mark tests**

Extend observations and candidates with darkness:

```python
MarkObservation(
    u=0.60,
    v=0.62,
    frame_index=8,
    time_sec=4.0,
    local_darkness=70.0,
)
```

Add these behavioral tests:

```python
def test_pre_contact_template_marks_are_excluded() -> None:
    tracks = build_temporal_mark_tracks(
        pre_contact=[_observation(0.30, 0.30, darkness=35, frame=1)],
        post_contact=[_observation(0.30, 0.30, darkness=36, frame=5)],
        minimum_observed_frames=1,
        maximum_local_distance=0.04,
        minimum_darkness_delta=15,
    )
    assert tracks == []


def test_existing_mark_that_darkens_after_contact_becomes_candidate() -> None:
    tracks = build_temporal_mark_tracks(
        pre_contact=[_observation(0.60, 0.62, darkness=20, frame=1)],
        post_contact=[_observation(0.60, 0.62, darkness=55, frame=5)],
        minimum_observed_frames=1,
        maximum_local_distance=0.04,
        minimum_darkness_delta=15,
    )
    assert tracks[0].darkness_delta == pytest.approx(35)


def test_darkest_two_choose_candidate_nearest_reference() -> None:
    selected = select_darkest_nearest_reference(
        [_track(0.92, 0.06, darkness=90), _track(0.62, 0.64, darkness=80), _track(0.20, 0.30, darkness=30)],
        reference_u=0.65,
        reference_v=0.65,
        maximum_candidates=2,
    )
    assert selected.punch is not None
    assert selected.punch.u == pytest.approx(0.62)
    assert len(selected.ranked_candidates) == 2


def test_track_reconnects_after_temporary_occlusion() -> None:
    tracks = build_temporal_mark_tracks(
        pre_contact=[],
        post_contact=[
            _observation(0.60, 0.62, darkness=55, frame=5),
            _observation(0.61, 0.61, darkness=58, frame=8),
        ],
        minimum_observed_frames=2,
        maximum_local_distance=0.04,
        minimum_darkness_delta=15,
    )
    assert len(tracks) == 1
    assert tracks[0].observed_frame_count == 2
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_cp01_mark_selection.py -q
```

Expected: failures for missing darkness fields/functions.

- [ ] **Step 3: Implement local darkness and temporal selection**

Use these public contracts:

```python
@dataclass(frozen=True)
class MarkObservation:
    u: float
    v: float
    frame_index: int
    time_sec: float
    local_darkness: float


@dataclass(frozen=True)
class MarkCandidate:
    u: float
    v: float
    first_frame_index: int
    last_frame_index: int
    first_sec: float
    last_sec: float
    observed_frame_count: int
    median_darkness: float
    darkness_delta: float


@dataclass(frozen=True)
class PunchSelection:
    status: Literal["selected", "missing"]
    punch: MarkCandidate | None
    ranked_candidates: tuple[MarkCandidate, ...]
    reason: str
```

For each connected component, calculate `local_darkness` from the median HSV value in a small dilated ring minus the component median. `build_temporal_mark_tracks` must match pre/post clusters by local distance, retain newly appearing tracks, and retain pre-existing tracks only when their post-contact median darkness increases by at least `minimum_darkness_delta`. `select_darkest_nearest_reference` sorts by `median_darkness` descending, truncates to two, then selects the smallest Euclidean reference distance.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: all mark-selection tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/medical_evaluation/features/marks.py tests/test_cp01_mark_selection.py
git commit -m "feat: track new CP01 marks"
```

### Task 4: Rewrite CP01 extraction around pen contact

**Files:**
- Modify: `src/medical_evaluation/extractors/cp01.py`
- Modify: `tests/test_cp01_extractor.py`

- [ ] **Step 1: Write failing extractor tests**

Create fake frames containing both object masks and verify these paths:

```python
def test_missing_pen_prompt_returns_no_contact_features_without_error() -> None:
    result = extractor_without_pen.extract(
        video_path,
        "cp_01",
        TimeRange(start_sec=0.0, end_sec=1.0),
        dense_fps=2.0,
        analysis_width=1280,
    )
    assert result.features["pen_contact_detected"] is False
    assert result.features["pen_valid_frame_count"] == 0.0


def test_contact_splits_background_and_post_contact_marks() -> None:
    result = extractor_with_contact.extract(
        video_path,
        "cp_01",
        TimeRange(start_sec=0.0, end_sec=1.0),
        dense_fps=2.0,
        analysis_width=1280,
    )
    assert {item.object_id for item in segmenter.prompts} == {"rubber_dam", "marking_pen"}
    assert result.features["pen_contact_detected"] is True
    assert result.features["preexisting_mark_candidate_count"] == 1.0
    assert result.features["new_mark_candidate_count"] == 2.0
    assert result.features["selected_mark_u"] == pytest.approx(0.62, abs=0.03)
```

Also test fewer than three dam frames produces insufficient evidence, and stable contact followed by no new mark produces `new_mark_candidate_count == 0` without fabricating a point.

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_cp01_extractor.py -q
```

Expected: failures because current extractor ignores `marking_pen` and lacks the new features.

- [ ] **Step 3: Implement the minimal extraction state machine**

The constructor receives rubric-derived thresholds:

```python
def __init__(
    self,
    *,
    segmenter: VideoSegmenter,
    annotations: VideoAnnotations,
    evidence_root: Path,
    reference_u: float,
    reference_v: float,
    min_pen_dam_overlap_ratio: float,
    min_pen_contact_frames: int,
    min_mark_observed_frames: int,
    min_new_mark_darkness_delta: float,
    min_mark_area_ratio: float,
    max_mark_area_ratio: float,
    maximum_black_value: int,
    maximum_faint_value: int,
    maximum_faint_saturation: int,
    max_mark_aspect_ratio: float,
    min_mark_circularity: float,
    max_local_cluster_distance: float,
    local_darkness_ring_radius: int,
) -> None:
    if min_pen_contact_frames <= 0 or min_mark_observed_frames <= 0:
        raise ValueError("CP01 frame thresholds must be positive")
    self.segmenter = segmenter
    self.annotations = annotations
    self.evidence_root = evidence_root
    self.reference_u = reference_u
    self.reference_v = reference_v
    self.min_pen_dam_overlap_ratio = min_pen_dam_overlap_ratio
    self.min_pen_contact_frames = min_pen_contact_frames
    self.min_mark_observed_frames = min_mark_observed_frames
    self.min_new_mark_darkness_delta = min_new_mark_darkness_delta
    self.min_mark_area_ratio = min_mark_area_ratio
    self.max_mark_area_ratio = max_mark_area_ratio
    self.maximum_black_value = maximum_black_value
    self.maximum_faint_value = maximum_faint_value
    self.maximum_faint_saturation = maximum_faint_saturation
    self.max_mark_aspect_ratio = max_mark_aspect_ratio
    self.min_mark_circularity = min_mark_circularity
    self.max_local_cluster_distance = max_local_cluster_distance
    self.local_darkness_ring_radius = local_darkness_ring_radius
```

Load exactly one `rubber_dam` box. Load zero or one `marking_pen` box without using `prompts_for_object` for the optional case. Track only the dam when no pen exists; otherwise submit both prompts in one SAM2 call. Append one `MaskOverlap` for every sampled dam-valid frame, using ratio zero when the pen mask is missing, so a missing sample breaks a consecutive contact run. Build overlap observations, determine the first stable contact, split mark observations into pre/post lists, build temporal tracks, select Top 2 nearest reference, and emit the feature names from the design. Set model version suffix to `opencv-pen-gated-marks-v1`.

Return this exact feature payload:

```python
features = {
    "pen_valid_frame_count": float(pen_valid_frame_count),
    "pen_dam_overlap_ratio": contact_event.maximum_ratio,
    "pen_contact_frame_count": float(contact_event.contact_frame_count),
    "pen_contact_detected": contact_event.detected,
    "preexisting_mark_candidate_count": float(len(preexisting_tracks)),
    "new_mark_candidate_count": float(len(new_or_darkened_tracks)),
    "selected_mark_u": selection.punch.u if selection.punch else None,
    "selected_mark_v": selection.punch.v if selection.punch else None,
    "selected_mark_darkness": (
        selection.punch.median_darkness if selection.punch else None
    ),
    "mark_reference_distance": mark_reference_distance,
    "dam_valid_frame_count": float(len(valid_frames)),
}
```

- [ ] **Step 4: Write CP01 evidence**

Produce at most three overlays under `cp_01/overlays/`:

```text
contact-<frame>.jpg
background-<frame>.jpg
selection-<frame>.jpg
```

Use yellow for the dam, cyan for the pen, magenta for overlap, gray for excluded background marks, red for Top-2 candidates, green for the reference, and a yellow line from the selected candidate to the reference.

Because `EvidenceItem` intentionally contains only time, path and rule, write a
separate `cp_01/evidence.json` with the applied thresholds, every displayed
track's first/last time and median darkness, and the final selection reason.

Keep evidence selection explicit. Add three private methods with the same argument pattern as the existing `_write_overlay`; each returns one `EvidenceItem`. Assemble them with actual frame indices:

```python
evidence = []
if contact_event.first_frame_index is not None:
    evidence.append(
        self._write_contact_evidence(
            video_path,
            valid_frames[contact_event.first_frame_index],
        )
    )
if pre_contact_observations:
    background_index = pre_contact_observations[-1].frame_index
    evidence.append(
        self._write_background_evidence(
            video_path,
            valid_frames[background_index],
            preexisting_tracks,
        )
    )
if selection.punch is not None:
    selection_index = selection.punch.last_frame_index
    evidence.append(
        self._write_selection_evidence(
            video_path,
            valid_frames[selection_index],
            selection,
        )
    )
return evidence[:3]
```

- [ ] **Step 5: Run tests and verify GREEN**

Run the Step 2 command. Expected: all extractor tests pass and evidence files exist.

- [ ] **Step 6: Commit**

```bash
git add src/medical_evaluation/extractors/cp01.py tests/test_cp01_extractor.py
git commit -m "feat: extract pen-gated CP01 evidence"
```

### Task 5: Update CP01 Judge state precedence

**Files:**
- Modify: `src/medical_evaluation/judges/cp01_cp03.py`
- Modify: `tests/test_cp01_judge.py`
- Modify: `tests/test_judges_all.py`

- [ ] **Step 1: Write failing Judge tests**

```python
def test_no_stable_pen_contact_is_incomplete() -> None:
    result = judge_cp01(
        {"dam_valid_frame_count": 10.0, "pen_contact_detected": False},
        THRESHOLDS,
    )
    assert result.status.value == "incomplete"


def test_contact_without_new_mark_is_incorrect() -> None:
    result = judge_cp01(
        {
            "dam_valid_frame_count": 10.0,
            "pen_contact_detected": True,
            "new_mark_candidate_count": 0.0,
            "mark_reference_distance": None,
        },
        THRESHOLDS,
    )
    assert result.status.value == "incorrect"


def test_insufficient_dam_frames_need_review_before_contact_state() -> None:
    result = judge_cp01(
        {"dam_valid_frame_count": 2.0, "pen_contact_detected": False},
        THRESHOLDS,
    )
    assert result.status.value == "needs_review"
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_cp01_judge.py tests/test_judges_all.py -q
```

- [ ] **Step 3: Implement precedence**

Use this order:

```python
if float(features.get("dam_valid_frame_count") or 0) < 3:
    return needs_review(
        "cp_01",
        features,
        reason_code="missing_required_evidence",
        reason="橡皮布有效画面不足，无法判断标记过程。",
    )
if features.get("pen_contact_detected") is not True:
    return incomplete(
        "cp_01",
        features,
        reason_code="marking_pen_contact_not_observed",
        reason="完整阶段内未观察到标记笔稳定接触橡皮布。",
        suggestion="请使用标记笔在橡皮布上完成目标牙位标记。",
    )
if float(features.get("new_mark_candidate_count") or 0) == 0:
    return incorrect(
        "cp_01",
        features,
        reason_code="mark_not_left_after_contact",
        reason="标记笔接触橡皮布后未形成可确认的新标记。",
        suggestion="接触橡皮布后留下清晰、稳定的牙位标记。",
    )
distance = features.get("mark_reference_distance")
if distance is None:
    return needs_review(
        "cp_01",
        features,
        reason_code="missing_required_evidence",
        reason="候选标记证据不足，无法计算与参考牙位的距离。",
    )
if float(distance) > thresholds["max_mark_distance"]:
    return incorrect(
        "cp_01",
        features,
        reason_code="mark_position_incorrect",
        reason="学员标记偏离目标牙位。",
        suggestion="重新确认36牙在橡皮布上的相对位置。",
    )
return correct(
    "cp_01",
    features,
    matched_rules=["pen_contact_then_mark_at_reference_position"],
    reason="标记笔接触后形成的标记与目标牙位一致。",
)
```

Remove the obsolete ambiguous/corner-candidate branch.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: all CP01 Judge cases pass.

- [ ] **Step 5: Commit**

```bash
git add src/medical_evaluation/judges/cp01_cp03.py tests/test_cp01_judge.py tests/test_judges_all.py
git commit -m "feat: judge pen-gated CP01 states"
```

### Task 6: Update calibration and combined smoke wiring

**Files:**
- Modify: `scripts/calibrate_cp01_detector.py`
- Modify: `scripts/smoke_sam2_cp01_cp11.py`
- Modify: `tests/test_calibrate_cp01_detector.py`
- Modify: `tests/test_smoke_cp01_cp11.py`

- [ ] **Step 1: Write failing orchestration tests**

Add one calibration test that supplies a synthetic contact split and one smoke-construction test that captures the extractor keyword arguments:

```python
def test_sweep_reports_temporal_candidates_and_overlay(tmp_path: Path) -> None:
    result = run_threshold_entry(
        frames=_synthetic_cp01_frames(),
        dam_masks=_dam_masks(),
        pen_masks=_pen_contact_masks(),
        output_dir=tmp_path,
        min_pen_dam_overlap_ratio=0.02,
        min_pen_contact_frames=2,
        min_mark_observed_frames=3,
        min_new_mark_darkness_delta=15.0,
    )
    assert result["preexisting_candidate_count"] == 1
    assert result["new_candidate_count"] == 2
    assert Path(result["overlay_path"]).is_file()


def test_smoke_passes_all_cp01_temporal_thresholds(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class CapturingCp01Extractor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(smoke_module, "Cp01FeatureExtractor", CapturingCp01Extractor)
    smoke_module.build_cp01_extractor(_rubric_with_cp01_thresholds())

    assert captured["min_pen_dam_overlap_ratio"] == 0.02
    assert captured["min_pen_contact_frames"] == 2
    assert captured["min_mark_observed_frames"] == 3
    assert captured["min_new_mark_darkness_delta"] == 15.0
    assert captured["min_mark_area_ratio"] == 0.0001
    assert captured["max_mark_area_ratio"] == 0.01
    assert captured["maximum_black_value"] == 55
    assert captured["maximum_faint_value"] == 160
    assert captured["maximum_faint_saturation"] == 120
    assert captured["max_mark_aspect_ratio"] == 2.0
    assert captured["min_mark_circularity"] == 0.35
    assert captured["max_local_cluster_distance"] == 0.04
    assert captured["local_darkness_ring_radius"] == 5
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_calibrate_cp01_detector.py tests/test_smoke_cp01_cp11.py -q
```

- [ ] **Step 3: Implement calibration and smoke wiring**

Change the calibrator to collect frames and both masks once, compute the first stable contact for each grid entry, split background/post-contact observations, and write one overlay per unique Top-2 coordinate pair under `data/calibration/cp01_detector_sweep_overlays/`. Include all thresholds and feature counts in `cp01_detector_sweep.json`.

In combined smoke, construct CP01 using:

```python
cp01_rule = next(item for item in rubric.checkpoints if item.id == "cp_01")
cp01 = Cp01FeatureExtractor(
    segmenter=backend,
    annotations=annotations,
    evidence_root=run_root,
    reference_u=float(reference["reference_u"]),
    reference_v=float(reference["reference_v"]),
    min_pen_dam_overlap_ratio=cp01_rule.thresholds["min_pen_dam_overlap_ratio"],
    min_pen_contact_frames=int(cp01_rule.thresholds["min_pen_contact_frames"]),
    min_mark_observed_frames=int(cp01_rule.thresholds["min_mark_observed_frames"]),
    min_new_mark_darkness_delta=cp01_rule.thresholds["min_new_mark_darkness_delta"],
    min_mark_area_ratio=cp01_rule.thresholds["min_mark_area_ratio"],
    max_mark_area_ratio=cp01_rule.thresholds["max_mark_area_ratio"],
    maximum_black_value=int(cp01_rule.thresholds["maximum_black_value"]),
    maximum_faint_value=int(cp01_rule.thresholds["maximum_faint_value"]),
    maximum_faint_saturation=int(
        cp01_rule.thresholds["maximum_faint_saturation"]
    ),
    max_mark_aspect_ratio=cp01_rule.thresholds["max_mark_aspect_ratio"],
    min_mark_circularity=cp01_rule.thresholds["min_mark_circularity"],
    max_local_cluster_distance=cp01_rule.thresholds[
        "max_local_cluster_distance"
    ],
    local_darkness_ring_radius=int(
        cp01_rule.thresholds["local_darkness_ring_radius"]
    ),
)
```

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: both files pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_cp01_detector.py scripts/smoke_sam2_cp01_cp11.py tests/test_calibrate_cp01_detector.py tests/test_smoke_cp01_cp11.py
git commit -m "feat: calibrate pen-gated CP01"
```

### Task 7: Document, verify, sync, reannotate and run real GPU smoke

**Files:**
- Modify: `docs/server-runbook.md`
- Verify: all files above
- User data only: `data/annotations/*.json`, `data/calibration/*.json`, `data/runs/*`

- [ ] **Step 1: Update the runbook**

Document exact backup/reset commands, `marking_pen` annotation rules, calibration command, combined smoke command, output fields, overlay colors, and `incomplete`/`incorrect`/`needs_review` precedence. State that data and videos remain untracked.

- [ ] **Step 2: Run focused and full verification**

```powershell
python -m pytest tests/test_contact_features.py tests/test_cp01_mark_selection.py tests/test_cp01_extractor.py tests/test_cp01_judge.py tests/test_calibrate_cp01_detector.py tests/test_smoke_cp01_cp11.py -q
python -m pytest -q
ruff check src scripts tests
git diff --check
```

Expected: focused tests pass, full suite passes with only the known Starlette warning, Ruff passes, and `git diff --check` reports no errors.

- [ ] **Step 3: Commit documentation and synchronize GitHub**

```bash
git add docs/server-runbook.md
git commit -m "docs: run pen-gated CP01 smoke"
git push origin HEAD:main
git ls-remote origin refs/heads/main
```

Do not claim GitHub is synchronized unless `ls-remote` equals local `HEAD`.

- [ ] **Step 4: Create and verify an incremental server bundle**

Create the bundle from the server's verified current commit, not from an assumed base:

```powershell
git bundle create cp01-pen-gated-<short-head>.bundle main ^<server-base-sha>
git bundle verify cp01-pen-gated-<short-head>.bundle
Get-FileHash cp01-pen-gated-<short-head>.bundle -Algorithm SHA256
```

- [ ] **Step 5: Back up and reannotate CP01 on the server**

For `success`, preserve the existing `rubber_dam` and `cp01_reference`, remove only old CP01 `marking_pen` prompts if present, then add exactly one tight pen box on a clear contact frame. Audit the saved prompt JSON before inference.

- [ ] **Step 6: Run real success-video calibration**

```bash
CUDA_VISIBLE_DEVICES=<free-physical-gpu> python scripts/calibrate_cp01_detector.py \
  --video-id success \
  --checkpoint-path external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --model-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda:0 \
  --sample-fps 2 \
  --videos-dir "$HOME/medical_evaluation/videos" \
  --data-dir "$HOME/medical_evaluation/data"
```

Inspect the contact overlay, background exclusions, Top-2 marks and selected nearest-reference point. Select thresholds only when all semantics match the video.

- [ ] **Step 7: Run combined CP01/CP11 smoke**

```bash
CUDA_VISIBLE_DEVICES=<free-physical-gpu> python scripts/smoke_sam2_cp01_cp11.py \
  --video-id success \
  --checkpoint-path external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --model-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda:0 \
  --sample-fps 2 \
  --videos-dir "$HOME/medical_evaluation/videos" \
  --data-dir "$HOME/medical_evaluation/data"
```

Accept success only after the JSON features and all CP01/CP11 overlays agree with the visible objects. Then repeat annotation and smoke for `failure` and `clamp_failure`; do not copy success time ranges or prompts.

### Task 8: Fast compatibility for marking-pen point prompts

**User-authorized verification exception:** The user explicitly requested that this compatibility change not run tests. Update existing test expectations where text contracts change, but do not execute pytest or Ruff for this task. Report the skipped verification explicitly.

**Files:**
- Modify: `src/medical_evaluation/extractors/cp01.py`
- Modify: `src/medical_evaluation/web/routes.py`
- Modify: `tests/test_annotation_api.py`
- Modify: `docs/server-runbook.md`

- [ ] **Step 1: Accept one box or one-to-two same-frame positive points**

After `prompts_for_object` returns the `marking_pen` prompts, validate this exact contract:

```python
if any(prompt.kind == "box" for prompt in pen_prompts):
    valid = len(pen_prompts) == 1 and pen_prompts[0].kind == "box"
else:
    frame_times = {prompt.frame_time_sec for prompt in pen_prompts}
    valid = (
        1 <= len(pen_prompts) <= 2
        and all(prompt.kind == "point" and prompt.positive for prompt in pen_prompts)
        and len(frame_times) == 1
    )
if pen_prompts and not valid:
    raise ValueError(
        "cp_01 marking_pen requires one box or 1-2 positive points on one frame"
    )
```

- [ ] **Step 2: Update annotation guidance**

Use this learner hint and the equivalent reference-video wording:

```python
"CP01已执行时：框选完整橡皮布；标记笔可紧框，或在同一帧笔身内部打1–2个正点。"
```

- [ ] **Step 3: Update the runbook and existing API expectation**

Document where pen points belong, forbid hand/fabric/black-mark points, and update the exact guide string in `tests/test_annotation_api.py` without running tests.

- [ ] **Step 4: Commit and create a server bundle**

```bash
git add src/medical_evaluation/extractors/cp01.py src/medical_evaluation/web/routes.py tests/test_annotation_api.py docs/server-runbook.md
git commit -m "fix: accept CP01 pen point prompts"
git bundle create cp01-pen-points-<short-head>.bundle main ^fbc05e0298351374735f6c102e15a0a6e154dead
```
