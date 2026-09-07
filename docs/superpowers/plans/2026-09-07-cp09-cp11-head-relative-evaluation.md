# CP09/CP11 Head-Relative Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CP09/CP11 per-video mouth/nose boxes and CP09-to-CP11 frame tracking with SAM3 head-relative evidence, automatic stable windows, and deterministic judgments for frame installation, nose clearance, and frame coverage.

**Architecture:** A prompt-feasibility gate runs first on all three real videos and records the chosen head and nose prompts. Pure geometry code converts object masks into a mannequin-head coordinate system and finds stable tail windows; CP09 persists a frame reference in those coordinates, and CP11 maps that reference into its current head pose. The existing Judge remains the only source of status and score, SAM2 remains selectable, and Qwen remains an evidence-only commentator.

**Tech Stack:** Python 3.12, NumPy, OpenCV, PyTorch/SAM3.1, Pydantic settings, pytest, Ruff, Git, SSH, Qwen3-VL/vLLM.

---

## File map

- Create `scripts/run_sam3_prompt_matrix.py`: one-model prompt experiment runner that saves raw frames, masks, overlays, per-frame mask statistics, selected prompt, model revision, threshold, elapsed time, and peak VRAM.
- Create `tests/scripts/test_run_sam3_prompt_matrix.py`: tests manifest parsing, candidate ranking, continuous-frame acceptance, and artifact output without a GPU.
- Create `src/medical_evaluation/features/head_relative.py`: mask quality, head-local transforms, 180-degree angle distance, mask mapping, robust aggregation, and stable-window selection.
- Create `tests/test_head_relative.py`: unit tests for every pure geometry and stability rule.
- Create `src/medical_evaluation/frame_reference.py`: versioned CP09 head-relative frame reference schema and safe per-job/canonical persistence.
- Create `tests/test_frame_reference.py`: round-trip, schema version, source, fallback, and corrupted-file tests.
- Modify `src/medical_evaluation/segmentation/prompt_policy.py`: add head and nose text prompts selected by the prompt gate.
- Modify `tests/test_segmentation_prompt_policy.py`: assert exact semantic prompts and no per-video point/box dependency in SAM3 mode.
- Rewrite `src/medical_evaluation/extractors/cp09.py`: track head and frame, select the final stable window, emit head-relative metrics, persist the CP11 reference, and write overlays.
- Rewrite `tests/test_cp09_extractor.py`: cover unreliable head, absent frame, missing-at-end, unstable frame, bad installation, accepted installation, and reference persistence.
- Rewrite `src/medical_evaluation/extractors/cp11.py`: retain full-stage dam presence, track head/nose/dam only in the final candidate region, map the frame reference, and measure nose overlap and frame coverage.
- Rewrite `tests/test_cp11_extractor.py`: cover every CP11 evidence branch and verify the old full-frame dam metric and long frame track are absent.
- Modify `src/medical_evaluation/judges/cp09_cp11.py`: implement the approved state priority and gray-zone handling.
- Modify `tests/test_judges_cp09_cp11.py`: table-test exact statuses and reason codes.
- Modify `src/medical_evaluation/settings.py`, `config/rubric.yaml`, and `tests/test_settings.py`: add versioned Demo geometry, stability, coverage, and review-band settings.
- Modify `src/medical_evaluation/runtime.py` and `tests/test_runtime.py`: share the CP09 reference store with CP11 and keep backend selection unchanged.
- Modify `src/medical_evaluation/segmentation/audit.py` and `tests/test_segmentation_audit.py`: record prompts, stable windows, transforms, reference source, performance, and model metadata.
- Modify `src/medical_evaluation/rubric.py` only after comparing its server dirty diff; merge new keys without overwriting server-only changes.
- Update `docs/server-runbook.md`: document the new prompt gate, feature meanings, and exact three-video validation commands.

### Task 1: Establish the prompt-feasibility gate

**Files:**
- Create: `scripts/run_sam3_prompt_matrix.py`
- Create: `tests/scripts/test_run_sam3_prompt_matrix.py`
- Reuse: `scripts/run_sam3_text_smoke.py`

- [ ] **Step 1: Write failing tests for continuous detection and candidate ranking**

```python
def test_candidate_requires_three_consecutive_valid_masks():
    rows = [
        {"valid": True, "area_ratio": 0.20},
        {"valid": False, "area_ratio": 0.00},
        {"valid": True, "area_ratio": 0.21},
    ]
    assert not candidate_passes(rows, minimum_consecutive=3)


def test_rank_candidates_prefers_cross_video_continuity_then_score():
    results = {
        "prompt-a": {"success": (4, 0.8), "failure": (4, 0.7), "clamp_failure": (4, 0.6)},
        "prompt-b": {"success": (4, 0.9), "failure": (2, 0.9), "clamp_failure": (4, 0.9)},
    }
    assert rank_candidates(results)[0] == "prompt-a"
```

- [ ] **Step 2: Run the focused test and confirm red**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/test_run_sam3_prompt_matrix.py -q`

Expected: collection fails because `candidate_passes` and `rank_candidates` do not exist.

- [ ] **Step 3: Implement the matrix runner and artifact contract**

Use the exact bounded candidates from the approved design. A candidate passes one sample only when it has at least three consecutive non-empty masks and each accepted mask has one dominant connected component occupying at least 60% of the total mask area. Rank first by number of samples passed, then by median SAM score, then by candidate order. The script must return exit code 2 unless one head prompt passes CP09 and CP11 on all samples and one nose prompt passes the CP11 final candidate region on all samples.

```python
HEAD_PROMPTS = (
    "dental training mannequin head",
    "plastic dental mannequin face",
    "dental mannequin head and face",
)
NOSE_PROMPTS = (
    "nose of the dental mannequin",
    "plastic nose on the dental mannequin face",
    "mannequin nose",
)


def candidate_passes(rows: list[dict[str, float | bool]], minimum_consecutive: int = 3) -> bool:
    run = 0
    for row in rows:
        run = run + 1 if bool(row["valid"]) else 0
        if run >= minimum_consecutive:
            return True
    return False


def rank_candidates(results: dict[str, dict[str, tuple[int, float]]]) -> list[str]:
    order = {text: index for index, text in enumerate((*HEAD_PROMPTS, *NOSE_PROMPTS))}
    return sorted(
        results,
        key=lambda text: (
            -sum(run >= 3 for run, _score in results[text].values()),
            -float(np.median([score for _run, score in results[text].values()])),
            order[text],
        ),
    )
```

Write `summary.json`, `selected_prompts.json`, and `overlays/<video>/<object>/<prompt-slug>/*.jpg`. Include `video_id`, stage, source time, frame index, mask area ratio, dominant-component ratio, SAM score when supplied, threshold, model version, git revision, elapsed seconds, and CUDA peak allocated MiB.

Add a `multiplex_probe` section that makes one controlled head-plus-frame request against the locked official predictor API. It passes only when both returned object IDs remain distinct across three consecutive source frames. An unsupported signature, merged identities, or missing identity is recorded as `supported: false` with the exception/result text and selects the already-tested independent-session path; it does not block prompt feasibility.

- [ ] **Step 4: Run local tests and lint**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/test_run_sam3_prompt_matrix.py tests/scripts/test_run_sam3_text_smoke.py -q`

Expected: all selected tests pass.

Run: `.venv\Scripts\ruff.exe check scripts/run_sam3_prompt_matrix.py tests/scripts/test_run_sam3_prompt_matrix.py`

Expected: no lint errors.

- [ ] **Step 5: Commit the experiment harness**

```powershell
git add scripts/run_sam3_prompt_matrix.py tests/scripts/test_run_sam3_prompt_matrix.py
git commit -m "test: add SAM3 head and nose prompt gate"
```

- [ ] **Step 6: Sync this commit without touching the server rubric**

Push the branch, then on the server first save `git diff -- src/medical_evaluation/rubric.py` to `data/logs/rubric-before-head-relative.patch`. Fetch and fast-forward only if the checkout can do so without overwriting that path. If Git refuses because of the dirty file, use a bundle and apply only the prompt-runner commit after verifying its changed paths.

- [ ] **Step 7: Run the real prompt gate on physical GPU 6**

```powershell
ssh -i "$env:USERPROFILE\.ssh\medical_evaluation_codex" -o IdentitiesOnly=yes tangm@10.25.64.102 "cd /home/tangm/medical_evaluation && CUDA_VISIBLE_DEVICES=6 PYTHONPATH=\"$PWD/src\" conda run --no-capture-output -n sam3_medical python scripts/run_sam3_prompt_matrix.py --annotations data/annotations --videos videos --video-ids success failure clamp_failure --output data/runs/head-nose-prompt-gate --sample-fps 2 --output-threshold 0.2 --grounding-batch-size 4 --probe-multiplex"
```

Expected: exit 0; `selected_prompts.json` names one passing head prompt and one passing nose prompt; all six head stage/sample combinations and all three nose combinations contain at least three consecutive accepted masks.

- [ ] **Step 8: Inspect every representative overlay before production changes**

Copy `data/runs/head-nose-prompt-gate` to `artifacts/.qa/head-nose-prompt-gate`. Check start/middle/end accepted frames for each video and reject a prompt if the dominant mask is an operator, glove, teeth, clamp, or background. Record the selected exact text and threshold in the run summary. If no candidate passes, stop here and report the failed object, videos, times, and overlays; do not add manual boxes as a fallback.

### Task 2: Add pure head-relative geometry and stability selection

**Files:**
- Create: `src/medical_evaluation/features/head_relative.py`
- Create: `tests/test_head_relative.py`

- [ ] **Step 1: Write failing geometry tests**

```python
def test_point_round_trip_in_rotated_head_coordinates():
    pose = MaskPose(cx=50.0, cy=40.0, width=40.0, height=60.0, angle_deg=30.0)
    local = image_to_head((62.0, 51.0), pose)
    assert head_to_image(local, pose) == pytest.approx((62.0, 51.0))


@pytest.mark.parametrize(("left", "right", "expected"), [(179.0, 1.0, 2.0), (10.0, 170.0, 20.0)])
def test_axis_angle_distance_is_180_degree_periodic(left, right, expected):
    assert axis_angle_distance(left, right) == pytest.approx(expected)


def test_latest_stable_window_uses_tail_run_of_four_frames():
    samples = synthetic_relative_samples(times=[0, .5, 1, 1.5, 2, 2.5], unstable_until=1.0)
    window = find_latest_stable_window(samples, minimum_duration_sec=1.5, tail_fraction=0.6)
    assert (window.start_sec, window.end_sec) == pytest.approx((1.0, 2.5))
```

- [ ] **Step 2: Run and confirm red**

Run: `.venv\Scripts\python.exe -m pytest tests/test_head_relative.py -q`

Expected: import fails because `head_relative.py` does not exist.

- [ ] **Step 3: Implement the complete pure API**

```python
@dataclass(frozen=True)
class MaskPose:
    cx: float
    cy: float
    width: float
    height: float
    angle_deg: float


@dataclass(frozen=True)
class RelativeSample:
    frame_index: int
    time_sec: float
    horizontal: float
    vertical: float
    angle_deg: float
    scale_ratio: float


@dataclass(frozen=True)
class StableWindow:
    start_sec: float
    end_sec: float
    sample_indices: tuple[int, ...]


def axis_angle_distance(left: float, right: float) -> float:
    return abs((left - right + 90.0) % 180.0 - 90.0)


def image_to_head(point: tuple[float, float], head: MaskPose) -> tuple[float, float]:
    theta = np.deg2rad(-head.angle_deg)
    delta = np.asarray(point, dtype=float) - np.asarray((head.cx, head.cy))
    rotated = np.asarray(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))) @ delta
    return float(rotated[0] / head.width), float(rotated[1] / head.height)


def head_to_image(point: tuple[float, float], head: MaskPose) -> tuple[float, float]:
    theta = np.deg2rad(head.angle_deg)
    scaled = np.asarray((point[0] * head.width, point[1] * head.height))
    rotated = np.asarray(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))) @ scaled
    return float(rotated[0] + head.cx), float(rotated[1] + head.cy)
```

`pose_from_mask` must keep the largest connected component, reject empty masks, reject a dominant-component ratio below 0.60, reject components touching three or four image edges, and return the PCA center, 5th-to-95th percentile extents, and a 180-degree axis. `find_latest_stable_window` must search only the final 60% of the stage, require at least two seconds and three samples, and require each sample to stay within configured horizontal, vertical, angle, and scale dispersion from the window medians.

- [ ] **Step 4: Run the full pure-feature tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_head_relative.py -q`

Expected: all tests pass, including edge clipping, fragmented masks, insufficient duration, tail restriction, and affine mask mapping.

- [ ] **Step 5: Commit**

```powershell
git add src/medical_evaluation/features/head_relative.py tests/test_head_relative.py
git commit -m "feat: add head-relative geometry and stable windows"
```

### Task 3: Persist versioned frame references

**Files:**
- Create: `src/medical_evaluation/frame_reference.py`
- Create: `tests/test_frame_reference.py`
- Create after real calibration: `config/mannequin_frame_reference.v1.json`

- [ ] **Step 1: Write failing schema and fallback tests**

```python
def test_store_prefers_current_video_reference_over_canonical(tmp_path):
    store = FrameReferenceStore(job_root=tmp_path, canonical_path=tmp_path / "canonical.json")
    store.write_current(_reference(source="cp09_current_video"))
    assert store.load_best().source == "cp09_current_video"


def test_store_uses_versioned_canonical_when_current_missing(tmp_path):
    canonical = tmp_path / "canonical.json"
    canonical.write_text(_reference(source="canonical_success_v1").model_dump_json(), encoding="utf-8")
    assert FrameReferenceStore(tmp_path / "job", canonical).load_best().source == "canonical_success_v1"
```

- [ ] **Step 2: Run and confirm red**

Run: `.venv\Scripts\python.exe -m pytest tests/test_frame_reference.py -q`

Expected: import fails because the store does not exist.

- [ ] **Step 3: Implement the schema and atomic store**

```python
class HeadRelativeFrameReference(BaseModel):
    schema_version: Literal[1] = 1
    source: Literal["cp09_current_video", "canonical_success_v1"]
    center_u: float
    center_v: float
    angle_deg: float
    width_ratio: float
    height_ratio: float
    polygon_uv: list[tuple[float, float]]
    appearance_lab: tuple[float, float, float]
    stable_start_sec: float
    stable_end_sec: float


class FrameReferenceStore:
    def __init__(self, job_root: Path, canonical_path: Path) -> None:
        self.current_path = job_root / "cp_09" / "frame_reference.json"
        self.canonical_path = canonical_path

    def write_current(self, reference: HeadRelativeFrameReference) -> None:
        self.current_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.current_path.with_suffix(".tmp")
        temporary.write_text(reference.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(self.current_path)

    def load_best(self) -> HeadRelativeFrameReference | None:
        for path in (self.current_path, self.canonical_path):
            if path.is_file():
                return HeadRelativeFrameReference.model_validate_json(path.read_text(encoding="utf-8"))
        return None
```

- [ ] **Step 4: Run tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_frame_reference.py -q`

Expected: all tests pass, including invalid schema and truncated JSON raising a clear validation error.

```powershell
git add src/medical_evaluation/frame_reference.py tests/test_frame_reference.py
git commit -m "feat: persist head-relative frame references"
```

### Task 4: Replace CP09 extraction and judgment

**Files:**
- Modify: `src/medical_evaluation/segmentation/prompt_policy.py`
- Modify: `src/medical_evaluation/extractors/cp09.py`
- Modify: `src/medical_evaluation/judges/cp09_cp11.py`
- Modify: `tests/test_segmentation_prompt_policy.py`
- Rewrite: `tests/test_cp09_extractor.py`
- Modify: `tests/test_judges_cp09_cp11.py`

- [ ] **Step 1: Add failing prompt and extractor tests**

Use the exact head prompt selected by Task 1 in `TextPromptPolicy.head_prompts()`. The fake segmenter must return head frames for `{"mannequin_head"}` and frame frames for `{"rubber_dam_frame"}` so the tests also validate the independent-session fallback supported by the locked multiplex API.

```python
@pytest.mark.parametrize(
    ("features", "status", "reason"),
    [
        ({"head_tracking_reliable": False}, "needs_review", "unreliable_head_tracking"),
        ({"head_tracking_reliable": True, "frame_presence_ratio": 0.0}, "incomplete", "frame_not_observed"),
        ({"head_tracking_reliable": True, "frame_presence_ratio": .4, "frame_present_at_end": False}, "incorrect", "frame_missing_at_stage_end"),
        ({"head_tracking_reliable": True, "frame_presence_ratio": .8, "frame_present_at_end": True, "final_stable_duration_sec": 0.0}, "incorrect", "frame_not_stabilized"),
    ],
)
def test_cp09_state_priority(features, status, reason):
    decision = judge_cp09(features, THRESHOLDS)
    assert (decision.status, decision.reason_code) == (status, reason)
```

- [ ] **Step 2: Run focused tests and confirm red**

Run: `.venv\Scripts\python.exe -m pytest tests/test_segmentation_prompt_policy.py tests/test_cp09_extractor.py tests/test_judges_cp09_cp11.py -q`

Expected: failures mention missing `head_prompts`, old oral-region features, and old reason codes.

- [ ] **Step 3: Implement CP09 extraction**

Call `segmenter.track` twice over the same CP09 `TimeRange`, once with `mannequin_head` and once with `rubber_dam_frame`, using the chosen texts and the same `dense_fps`. Join by source `frame_index`; never interpolate a missing mask. Emit:

```python
features = {
    "head_tracking_reliable": head_valid_count >= minimum_valid_frames,
    "head_valid_count": float(head_valid_count),
    "frame_presence_ratio": frame_valid_count / head_valid_count if head_valid_count else None,
    "frame_valid_count": float(frame_valid_count),
    "frame_present_at_end": tail_presence_ratio >= thresholds.min_tail_presence_ratio,
    "final_stable_duration_sec": stable_duration,
    "frame_head_horizontal_offset": median_horizontal_error,
    "frame_head_vertical_offset": median_vertical_error,
    "frame_head_angle_difference": median_angle_error,
    "frame_head_scale_ratio": median_scale_ratio,
    "frame_head_horizontal_mad": horizontal_mad,
    "frame_head_vertical_mad": vertical_mad,
    "frame_head_angle_mad": angle_mad,
    "frame_head_scale_mad": scale_mad,
}
```

Persist `HeadRelativeFrameReference` whenever both tracks are reliable and a stable window exists, even when the installation metrics later fail. Save three stable-window overlays showing head contour, head axes, calibrated target band, frame contour, center vector, and timestamp.

- [ ] **Step 4: Implement CP09 Judge priority and review bands**

For each absolute error, values at or below the pass threshold pass, values above `pass * (1 + review_band_fraction)` fail, and values between enter `needs_review/frame_installation_borderline`. Check head reliability, presence, final presence, and stability before geometry. Geometry failures return `incorrect/frame_installation_failed`; accepted values return `correct/criteria_satisfied`.

- [ ] **Step 5: Run focused tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_head_relative.py tests/test_frame_reference.py tests/test_segmentation_prompt_policy.py tests/test_cp09_extractor.py tests/test_judges_cp09_cp11.py -q`

Expected: all tests pass.

```powershell
git add src/medical_evaluation/segmentation/prompt_policy.py src/medical_evaluation/extractors/cp09.py src/medical_evaluation/judges/cp09_cp11.py tests/test_segmentation_prompt_policy.py tests/test_cp09_extractor.py tests/test_judges_cp09_cp11.py
git commit -m "feat: evaluate CP09 in mannequin-head coordinates"
```

### Task 5: Replace CP11 final-state extraction and judgment

**Files:**
- Modify: `src/medical_evaluation/extractors/cp11.py`
- Modify: `src/medical_evaluation/judges/cp09_cp11.py`
- Rewrite: `tests/test_cp11_extractor.py`
- Modify: `tests/test_judges_cp09_cp11.py`

- [ ] **Step 1: Write failing CP11 behavior tests**

```python
def test_cp11_tracks_head_nose_and_dam_without_frame_long_track(fake_segmenter, extractor):
    result = extractor.extract(VIDEO, "cp_11", CP11_RANGE, dense_fps=2, analysis_width=1280)
    assert fake_segmenter.object_calls == [
        {"mannequin_head"}, {"mannequin_nose"}, {"rubber_dam"}
    ]
    assert "dam_area_ratio" not in result.features
    assert result.features["expected_frame_dam_coverage_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("features", "status", "reason"),
    [
        ({"head_tracking_reliable": False}, "needs_review", "unreliable_head_tracking"),
        ({"head_tracking_reliable": True, "dam_stage_presence_ratio": 0.0}, "incomplete", "rubber_dam_not_observed"),
        ({"head_tracking_reliable": True, "dam_stage_presence_ratio": .3, "dam_final_presence_ratio": 0.0}, "incorrect", "rubber_dam_missing_at_end"),
        ({"head_tracking_reliable": True, "dam_stage_presence_ratio": .3, "dam_final_presence_ratio": 1.0, "final_stable_duration_sec": 0.0}, "incorrect", "final_state_not_stable"),
    ],
)
def test_cp11_state_priority(features, status, reason):
    decision = judge_cp11(features, THRESHOLDS)
    assert (decision.status, decision.reason_code) == (status, reason)
```

- [ ] **Step 2: Run and confirm red**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cp11_extractor.py tests/test_judges_cp09_cp11.py -q`

Expected: failures show the old manual nose box, old long frame track, and missing coverage feature.

- [ ] **Step 3: Implement the CP11 extractor**

Keep the 1 FPS whole-stage OpenCV green scan solely for `dam_stage_presence_ratio`. If dam appeared, track head, nose, and dam over the final 60% of CP11 using separate sessions on the same model. Join by source frame index, find the latest two-second stable window, load `FrameReferenceStore.load_best()`, and map its `polygon_uv` into each CP11 head pose.

For each valid stable frame calculate:

```python
nose_overlap_ratio = float(np.count_nonzero(dam & nose) / np.count_nonzero(nose))
expected_frame_dam_coverage_ratio = float(
    np.count_nonzero(dam & expected_frame) / np.count_nonzero(expected_frame)
)
visible = visible_reference_mask(frame, expected_frame, reference.appearance_lab, max_lab_distance=20.0)
visible_frame_ratio = float(np.count_nonzero(visible) / np.count_nonzero(expected_frame))
```

The final feature map must contain `dam_stage_presence_ratio`, `dam_final_presence_ratio`, `head_tracking_reliable`, `head_valid_count`, `nose_tracking_reliable`, `nose_valid_count`, `dam_valid_count`, `frame_reference_available`, `frame_reference_source`, `final_stable_duration_sec`, `nose_overlap_ratio`, `expected_frame_dam_coverage_ratio`, and `visible_frame_ratio`. Remove `dam_area_ratio`, `nose_overlap`, `visible_frame_area_ratio`, fixed `final_window_sec`, manual `_nose_box`, `_frame_reference_lab`, and all CP09-to-CP11 segmentation calls.

- [ ] **Step 4: Implement CP11 Judge priority**

Apply exactly: unreliable head; no dam; dam missing at end; no stable final state; unreliable nose; no reference; nose obstruction; frame coverage failure; correct. Coverage fails if the coverage value is below its minimum or visible value exceeds its maximum. Threshold-border values return `needs_review/final_position_borderline` before an incorrect classification.

- [ ] **Step 5: Run focused tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_head_relative.py tests/test_frame_reference.py tests/test_cp11_extractor.py tests/test_judges_cp09_cp11.py -q`

Expected: all tests pass.

```powershell
git add src/medical_evaluation/extractors/cp11.py src/medical_evaluation/judges/cp09_cp11.py tests/test_cp11_extractor.py tests/test_judges_cp09_cp11.py
git commit -m "feat: evaluate CP11 nose clearance and frame coverage"
```

### Task 6: Wire settings, runtime, audit, and backward compatibility

**Files:**
- Modify: `src/medical_evaluation/settings.py`
- Modify: `config/rubric.yaml`
- Modify: `src/medical_evaluation/runtime.py`
- Modify: `src/medical_evaluation/segmentation/audit.py`
- Modify carefully: `src/medical_evaluation/rubric.py`
- Modify: `tests/test_settings.py`
- Modify: `tests/test_runtime.py`
- Modify: `tests/test_segmentation_audit.py`

- [ ] **Step 1: Write failing configuration and runtime tests**

Assert that SAM3 runtime constructs one segmenter/model, injects one per-job `FrameReferenceStore` into both extractors, and supplies the exact selected prompt strings. Assert SAM2 can still construct with `AnnotationPromptPolicy`; until equivalent automatic head/nose evidence exists for SAM2, CP09/CP11 return an explicit input-unavailable review result rather than silently using removed manual boxes.

- [ ] **Step 2: Run and confirm red**

Run: `.venv\Scripts\python.exe -m pytest tests/test_settings.py tests/test_runtime.py tests/test_segmentation_audit.py -q`

Expected: failures show missing new settings and reference-store injection.

- [ ] **Step 3: Add versioned Demo settings**

Store thresholds under CP09/CP11 rubric keys, with values derived reproducibly from the success calibration run: target medians come from the stable success window; pass tolerance is `max(3 * MAD, floor)` where floors are 0.03 head widths horizontally, 0.04 head heights vertically, 8 degrees, and 0.08 scale ratio; `review_band_fraction` is 0.25. CP11 initial engineering thresholds are maximum nose overlap 0.02, minimum expected-frame dam coverage 0.90, maximum visible-frame ratio 0.05, minimum tail presence 0.50, and minimum stable duration 2.0 seconds. Mark the canonical template `schema_version=1` and `source=canonical_success_v1`.

- [ ] **Step 4: Extend audit output**

Add one `evaluation_context` object per checkpoint containing selected prompt text, output threshold, sampling range, valid counts, stable window bounds, robust dispersions, head transforms for evidence frames, frame-reference source, elapsed time, CUDA peak when available, SAM3 revision, and weight hash. Preserve existing `segmentation_metadata.json` fields so old reports remain readable.

- [ ] **Step 5: Merge the server rubric safely**

Compare local `src/medical_evaluation/rubric.py` with `data/logs/rubric-before-head-relative.patch`. Apply only the required new feature-name and threshold mappings around the user's server changes. Run `git diff --check` and inspect `git diff -- src/medical_evaluation/rubric.py` before staging.

- [ ] **Step 6: Run integration tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_settings.py tests/test_runtime.py tests/test_segmentation_audit.py tests/test_pipeline.py tests/test_scoring.py tests/test_vlm_client.py -q`

Expected: all tests pass; Qwen schema and score-override protections remain unchanged.

```powershell
git add config/rubric.yaml config/mannequin_frame_reference.v1.json src/medical_evaluation/settings.py src/medical_evaluation/runtime.py src/medical_evaluation/segmentation/audit.py src/medical_evaluation/rubric.py tests/test_settings.py tests/test_runtime.py tests/test_segmentation_audit.py
git commit -m "feat: wire head-relative evaluation settings and audit"
```

### Task 7: Calibrate from real evidence without fitting failed labels

**Files:**
- Modify from measured success output: `config/rubric.yaml`
- Modify from measured success output: `config/mannequin_frame_reference.v1.json`
- Create as run artifact, do not commit: `artifacts/.qa/head-relative-calibration/`

- [ ] **Step 1: Run feature-only success calibration on GPU 6**

Run the normal pipeline with Judge output retained but with a calibration flag that writes raw CP09 relative series before threshold comparison. Qwen may remain disabled for this calibration run. Record all success stable-window medians and MAD values, generate the canonical template, and retain every overlay.

- [ ] **Step 2: Visually verify the calibration window**

Confirm head contour follows the mannequin, frame contour follows the white U-shaped frame, head axes do not flip, and the chosen window spans at least two seconds at the end of the installed state. If any check fails, repair extraction or prompt use and rerun; do not widen Judge thresholds.

- [ ] **Step 3: Freeze the deterministic v1 values**

Write the measured medians and the `max(3 * MAD, floor)` tolerances to `config/rubric.yaml`, and write the median head-local frame polygon plus appearance Lab to `config/mannequin_frame_reference.v1.json`. Add the source video ID, source time range, git revision, SAM3 revision, weight hash, and an explicit `demo_only: true` marker.

- [ ] **Step 4: Run failure and clamp_failure feature-only passes**

Use these runs to expose branch behavior and implementation defects. Do not alter v1 thresholds merely to force the expected labels. If evidence conflicts with the design expectation, classify the mismatch as mask failure, stage-boundary error, implementation defect, or rule-assumption error and preserve the raw artifacts.

- [ ] **Step 5: Commit the measured configuration**

```powershell
git add config/rubric.yaml config/mannequin_frame_reference.v1.json
git commit -m "calibrate: freeze mannequin frame reference v1"
```

### Task 8: Complete local and server verification

**Files:**
- Modify: `docs/server-runbook.md`
- Test: full repository

- [ ] **Step 1: Run all local tests and Ruff**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass; do not reuse the historical `217 passed` result.

Run: `.venv\Scripts\ruff.exe check .`

Expected: no lint errors.

- [ ] **Step 2: Update the runbook and commit**

Document the chosen prompt strings, prompt-gate artifact location, new feature names, reason codes, canonical-template provenance, GPU command, and review-band semantics.

```powershell
git add docs/server-runbook.md
git commit -m "docs: document head-relative CP09 CP11 validation"
```

- [ ] **Step 3: Push and verify GitHub**

Push `codex/sam3-cp09-cp11-text`, then compare `git rev-parse HEAD` with `git ls-remote origin refs/heads/codex/sam3-cp09-cp11-text`. Do not claim GitHub synchronization until hashes match.

- [ ] **Step 4: Update the server without overwriting dirty data/code**

Recheck `git status --short`, the saved rubric patch, current GPU processes, `pip check`, and runtime import. Fast-forward only after preserving and reconciling the server rubric. Confirm server HEAD equals the pushed branch and run the focused test set in `sam3_medical` with `PYTHONPATH="$PWD/src"`.

- [ ] **Step 5: Run the full server regression**

Run: `conda run --no-capture-output -n sam3_medical python -m pytest -q`

Expected: all tests pass in the actual SAM3 environment.

Run: `conda run --no-capture-output -n sam3_medical ruff check .`

Expected: no lint errors.

### Task 9: Run and inspect all three complete pipelines

**Files:**
- Runtime artifacts: `data/jobs/<job_id>/report.json`
- Runtime artifacts: `data/jobs/<job_id>/segmentation_metadata.json`
- Runtime artifacts: `data/jobs/<job_id>/cp_09/frame_reference.json`
- Local review copies: `artifacts/.qa/head-relative-final/`

- [ ] **Step 1: Confirm resources and isolate GPU roles**

Verify physical GPU 6 is free enough for SAM3 and leave the existing Qwen process on its current GPU. Do not stop other users' processes. Verify ports 8001 and 57116 before launching anything.

- [ ] **Step 2: Run success with the full pipeline and Qwen enabled**

Require a complete CP09 and CP11 report, stable-window overlays, current-video CP09 reference, genuine `source=qwen` commentary, evidence citations, evaluated count `2/11`, provisional score, and `final_score=null`. Inspect masks and numerical features before accepting the Judge results.

- [ ] **Step 3: Run failure and clamp_failure with their own annotations**

Do not reuse success ranges or references. Verify failure CP11 and clamp_failure CP11 remain `incomplete/rubber_dam_not_observed`. Determine CP09 from real evidence: absent throughout is `incomplete/frame_not_observed`; appeared then lost is `incorrect/frame_missing_at_stage_end`; present but unstable is `incorrect/frame_not_stabilized`; stable bad geometry is `incorrect/frame_installation_failed`.

- [ ] **Step 4: Inspect evidence and repair until acceptance**

For every sample inspect at least three CP09 and three CP11 overlays when evidence exists, confirm stable bounds against video time, compare head/nose/dam masks, inspect reference source, and compare Judge inputs to reason codes. Fix code or prompts for reproducible extraction errors and repeat focused tests plus affected GPU runs. Preserve runs that reveal failures.

- [ ] **Step 5: Verify the report pages**

Start the report server only after jobs finish, open each report, verify every image returns HTTP 200, Qwen cites valid evidence indices, statuses and scores match `report.json`, the page says `已评估 2/11 项`, and the score is marked non-final. Stop only the temporary report server created for this validation.

- [ ] **Step 6: Record the final evidence package**

Copy the three final report directories, prompt-gate output, calibration output, test logs, Ruff output, GPU peak logs, and exact local/GitHub/server commit hashes to `artifacts/.qa/head-relative-final/`. State that the three samples demonstrate Demo execution and branch behavior, not accuracy or generalization.

## Final self-review checklist

- [ ] Every approved CP09 status and reason code has a Judge test.
- [ ] Every approved CP11 status and reason code has a Judge test.
- [ ] Head and nose prompts pass continuous multi-frame visual inspection on all three videos before production changes.
- [ ] CP09 no longer reads `oral_region`; CP11 no longer reads `nose_region` annotations.
- [ ] CP11 no longer computes whole-frame `dam_area_ratio` or tracks the support frame from CP09.
- [ ] CP09 stable references persist even when CP09 geometry is incorrect.
- [ ] CP11 prefers the current-video reference and records canonical fallback use.
- [ ] SAM2 backend selection still works and unsupported automatic evidence fails explicitly.
- [ ] Judge, scoring, Qwen schema, evidence citations, and final-score rules remain protected.
- [ ] Local, GitHub, and server hashes are independently verified.
- [ ] Full pytest, Ruff, three GPU runs, overlay inspection, and web report checks have current evidence.
