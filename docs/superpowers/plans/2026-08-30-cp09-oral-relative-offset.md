# CP09 Oral-Relative Offset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CP09 image-center scoring with paired SAM2 tracking of `rubber_dam_frame` and `oral_region`, then score the frame center relative to the oral-region center.

**Architecture:** Keep one SAM2 inference state and submit both object IDs in one prompt list. Expand only the CP09 inference window far enough to include accepted boundary prompts, filter outputs back to the original stage, compute a median relative offset over paired valid masks, and write separate masks plus one combined overlay for three representative frames.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Pydantic, pytest, Ruff, SAM2.1 video predictor.

---

### Task 1: Add boundary-tolerant prompt selection

**Files:**
- Modify: `src/medical_evaluation/segmentation/prompts.py`
- Modify: `tests/test_segmentation_prompts.py`

- [ ] **Step 1: Write failing tests for default strict filtering and explicit tolerance**

Add tests that retain the existing zero-tolerance behavior, accept `174.686926` for a `175.0` start when `boundary_tolerance_sec=0.5`, reject negative tolerance, and include the expanded allowed range in the missing-prompt error.

```python
def test_accepts_prompt_just_outside_interval_with_explicit_tolerance() -> None:
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[
            BoxPrompt(
                video_id="success",
                frame_time_sec=174.686926,
                object_id="oral_region",
                x1=0.45,
                y1=0.39,
                x2=0.59,
                y2=0.76,
            )
        ],
    )

    result = prompts_for_object(
        annotations,
        "oral_region",
        TimeRange(start_sec=175.0, end_sec=195.0),
        checkpoint_id="cp_09",
        boundary_tolerance_sec=0.5,
    )

    assert len(result) == 1
    assert result[0].frame_time_sec == pytest.approx(174.686926)


def test_default_prompt_filter_remains_strict() -> None:
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[
            BoxPrompt(
                video_id="success",
                frame_time_sec=174.9,
                object_id="oral_region",
                x1=0.1,
                y1=0.1,
                x2=0.9,
                y2=0.9,
            )
        ],
    )

    with pytest.raises(ValueError, match=r"175\.000-195\.000"):
        prompts_for_object(
            annotations,
            "oral_region",
            TimeRange(start_sec=175.0, end_sec=195.0),
        )


def test_prompt_filter_rejects_negative_boundary_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        prompts_for_object(
            VideoAnnotations(video_id="success"),
            "oral_region",
            TimeRange(start_sec=175.0, end_sec=195.0),
            boundary_tolerance_sec=-0.1,
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest tests/test_segmentation_prompts.py -q
```

Expected: FAIL because `prompts_for_object` does not accept `boundary_tolerance_sec`.

- [ ] **Step 3: Implement the minimum tolerance parameter**

Update the function signature and interval check:

```python
def prompts_for_object(
    annotations: VideoAnnotations,
    object_id: str,
    time_range: TimeRange,
    *,
    checkpoint_id: str = "cp_09",
    boundary_tolerance_sec: float = 0.0,
) -> list[SegmentationPrompt]:
    if boundary_tolerance_sec < 0:
        raise ValueError("boundary_tolerance_sec must be non-negative")
    allowed_start = time_range.start_sec - boundary_tolerance_sec
    allowed_end = time_range.end_sec + boundary_tolerance_sec

    converted: list[SegmentationPrompt] = []
    for prompt in annotations.prompts:
        if prompt.object_id != object_id:
            continue
        if not allowed_start <= prompt.frame_time_sec <= allowed_end:
            continue
        if prompt.kind == "point":
            converted.append(
                SegmentationPrompt(
                    object_id=object_id,
                    kind="point",
                    frame_time_sec=prompt.frame_time_sec,
                    coordinates=[prompt.x, prompt.y],
                    positive=prompt.positive,
                )
            )
        elif prompt.kind == "box":
            converted.append(
                SegmentationPrompt(
                    object_id=object_id,
                    kind="box",
                    frame_time_sec=prompt.frame_time_sec,
                    coordinates=[prompt.x1, prompt.y1, prompt.x2, prompt.y2],
                )
            )
    if not converted:
        raise ValueError(
            f"{annotations.video_id} {checkpoint_id} has no {object_id} prompt inside "
            f"{allowed_start:.3f}-{allowed_end:.3f}s"
        )
    return sorted(converted, key=lambda item: item.frame_time_sec)
```

Use `allowed_start` and `allowed_end` in the actionable error message.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_segmentation_prompts.py -q
```

Expected: all prompt tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medical_evaluation/segmentation/prompts.py tests/test_segmentation_prompts.py
git commit -m "feat: allow bounded CP09 prompt timing tolerance"
```

### Task 2: Add the pure oral-relative geometry feature

**Files:**
- Modify: `src/medical_evaluation/features/geometry.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Write failing geometry tests**

Import `relative_bbox_center_offset` and add:

```python
def test_relative_bbox_center_offset_uses_reference_scale() -> None:
    oral = np.zeros((100, 200), bool)
    oral[20:80, 50:150] = True  # width=100, height=60, center=(99.5, 49.5)
    centered_frame = np.zeros((100, 200), bool)
    centered_frame[40:60, 90:110] = True
    shifted_frame = np.zeros((100, 200), bool)
    shifted_frame[40:60, 140:160] = True

    assert relative_bbox_center_offset(centered_frame, oral) == pytest.approx(0.0)
    assert relative_bbox_center_offset(shifted_frame, oral) == pytest.approx(0.5)


def test_relative_bbox_center_offset_handles_missing_and_mismatched_masks() -> None:
    empty = np.zeros((10, 10), bool)
    present = np.ones((10, 10), bool)

    assert relative_bbox_center_offset(empty, present) is None
    assert relative_bbox_center_offset(present, empty) is None
    with pytest.raises(ValueError, match="dimensions"):
        relative_bbox_center_offset(present, np.ones((8, 8), bool))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest tests/test_features.py -q
```

Expected: collection FAIL because `relative_bbox_center_offset` is missing.

- [ ] **Step 3: Implement the pure feature**

Add a small private bbox helper and the public feature:

```python
def _bounding_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def relative_bbox_center_offset(
    subject: np.ndarray,
    reference: np.ndarray,
) -> float | None:
    subject_mask, reference_mask = _matching_masks(subject, reference)
    subject_box = _bounding_box(subject_mask)
    reference_box = _bounding_box(reference_mask)
    if subject_box is None or reference_box is None:
        return None
    sx = (subject_box[0] + subject_box[2]) / 2
    sy = (subject_box[1] + subject_box[3]) / 2
    rx = (reference_box[0] + reference_box[2]) / 2
    ry = (reference_box[1] + reference_box[3]) / 2
    reference_width = reference_box[2] - reference_box[0] + 1
    reference_height = reference_box[3] - reference_box[1] + 1
    return float(
        np.hypot(
            (sx - rx) / reference_width,
            (sy - ry) / reference_height,
        )
    )
```

Refactor `bounding_box_center` to use `_bounding_box` without changing its public behavior.

- [ ] **Step 4: Run geometry tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_features.py -q
```

Expected: all geometry tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medical_evaluation/features/geometry.py tests/test_features.py
git commit -m "feat: measure mask center relative to oral region"
```

### Task 3: Track two objects and extract paired CP09 evidence

**Files:**
- Modify: `src/medical_evaluation/extractors/cp09.py`
- Modify: `tests/test_cp09_extractor.py`

- [ ] **Step 1: Expand test fixtures for two targets**

Update `_annotations()` to include a frame point at `1.0` and an oral box at `0.4`, where the stage starts at `0.5`. Make `FakeSegmenter` record its received `time_range` and prompts. Update `_frame` to accept both masks.

```python
BoxPrompt(
    video_id="success",
    frame_time_sec=0.4,
    object_id="oral_region",
    x1=0.25,
    y1=0.2,
    x2=0.75,
    y2=0.8,
)
```

- [ ] **Step 2: Write a failing dual-object extraction test**

Use three paired frames whose two bbox centers match. Assert:

```python
assert {prompt.object_id for prompt in segmenter.prompts} == {
    "rubber_dam_frame",
    "oral_region",
}
assert segmenter.time_range.start_sec == pytest.approx(0.4)
assert result.features == {
    "frame_oral_center_offset": pytest.approx(0.0),
    "frame_valid_count": 3.0,
    "oral_region_valid_count": 3.0,
    "paired_valid_count": 3.0,
}
assert len(list((evidence_root / "cp_09/masks/rubber_dam_frame").glob("*.png"))) == 3
assert len(list((evidence_root / "cp_09/masks/oral_region").glob("*.png"))) == 3
```

Also assert that the segmenter may return a conditioning frame at `0.4`, but it is excluded from features and evidence because the requested stage begins at `0.5`.

- [ ] **Step 3: Run the extractor tests and verify RED**

Run:

```bash
python -m pytest tests/test_cp09_extractor.py -q
```

Expected: FAIL because only `rubber_dam_frame` is requested and the old feature key is returned.

- [ ] **Step 4: Implement dual-object extraction**

In `Cp09FeatureExtractor.extract`:

```python
frame_prompts = prompts_for_object(
    self.annotations,
    "rubber_dam_frame",
    time_range,
    checkpoint_id=checkpoint_id,
    boundary_tolerance_sec=0.5,
)
oral_prompts = prompts_for_object(
    self.annotations,
    "oral_region",
    time_range,
    checkpoint_id=checkpoint_id,
    boundary_tolerance_sec=0.5,
)
prompts = sorted(frame_prompts + oral_prompts, key=lambda item: item.frame_time_sec)
tracking_range = TimeRange(
    start_sec=min(time_range.start_sec, *(item.frame_time_sec for item in prompts)),
    end_sec=max(time_range.end_sec, *(item.frame_time_sec for item in prompts)),
)
```

Call `track` once with `tracking_range`. Before counting or pairing, filter each returned `FrameMasks` with:

```python
if not time_range.start_sec <= frame_masks.frame_time_sec <= time_range.end_sec:
    continue
```

Count non-empty frame masks, non-empty oral masks, and paired valid masks independently. Compute paired offsets with `relative_bbox_center_offset`, and take their median only when at least three paired frames exist.

For representative paired frames, save two binary masks beneath object-specific directories and call:

```python
write_overlay(
    frame,
    {
        "rubber_dam_frame": frame_mask,
        "oral_region": oral_mask,
    },
    ...,
)
```

Set `EvidenceItem.rule` to `rubber_dam_frame_relative_to_oral_region`.

- [ ] **Step 5: Add and pass missing-target and low-pair tests**

Add tests for:

- oral masks missing while frame masks remain valid;
- two paired masks produce evidence but `frame_oral_center_offset=None`;
- an annotation set with no `oral_region` raises an actionable error before `track` is called.

Run:

```bash
python -m pytest tests/test_cp09_extractor.py -q
```

Expected: all extractor tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/medical_evaluation/extractors/cp09.py tests/test_cp09_extractor.py
git commit -m "feat: extract CP09 frame-to-oral relative evidence"
```

### Task 4: Change CP09 judge semantics and rubric requirements

**Files:**
- Modify: `src/medical_evaluation/judges/cp09_cp11.py`
- Modify: `config/rubric.yaml`
- Modify: `tests/test_judges_cp09_cp11.py`
- Modify: `tests/test_judges_all.py`
- Modify: `tests/test_pipeline_vertical.py`
- Modify: `tests/test_vlm_client.py`

- [ ] **Step 1: Write failing CP09 judge tests for the new feature**

Replace old cases with:

```python
def test_cp09_marks_frame_centered_on_oral_region_correct() -> None:
    result = judge_cp09(
        {"frame_oral_center_offset": 0.03},
        {"max_oral_center_offset": 0.08},
    )

    assert result.status.value == "correct"
    assert result.matched_rules == ["frame_centered_on_oral_region"]


def test_cp09_marks_large_oral_relative_offset_incorrect() -> None:
    result = judge_cp09(
        {"frame_oral_center_offset": 0.2},
        {"max_oral_center_offset": 0.08},
    )

    assert result.status.value == "incorrect"
    assert result.reason_code == "frame_not_centered_on_oral_region"


def test_cp09_requests_review_without_paired_evidence() -> None:
    result = judge_cp09(
        {"frame_oral_center_offset": None, "paired_valid_count": 2.0},
        {"max_oral_center_offset": 0.08},
    )

    assert result.status.value == "needs_review"
    assert result.reason_code == "missing_frame_oral_evidence"
```

- [ ] **Step 2: Run judge tests and verify RED**

Run:

```bash
python -m pytest tests/test_judges_cp09_cp11.py tests/test_judges_all.py -q
```

Expected: FAIL because the judge still reads `frame_center_offset`.

- [ ] **Step 3: Implement new judge wording and threshold key**

Use `frame_oral_center_offset` and `max_oral_center_offset`. Update reason strings to refer to the oral region, not the image:

```python
reason="未能稳定识别支架与口腔区域的相对位置。"
reason="支架中心相对口腔区域偏离允许范围。"
suggestion="安装支架后，以整个口腔区域为参照调整支架位置。"
matched_rules=["frame_centered_on_oral_region"]
```

In `config/rubric.yaml`, set CP09 `required_objects` to `rubber_dam_frame` and `oral_region`, and rename the threshold to `max_oral_center_offset: 0.08`. Treat `0.08` as a provisional engineering threshold pending three-video calibration.

- [ ] **Step 4: Update integration fixtures to the new key**

Replace CP09 occurrences in `tests/test_judges_all.py`, `tests/test_pipeline_vertical.py`, and `tests/test_vlm_client.py` with `frame_oral_center_offset` and the renamed threshold where applicable.

- [ ] **Step 5: Run judge and pipeline tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_judges_cp09_cp11.py tests/test_judges_all.py tests/test_pipeline_vertical.py tests/test_vlm_client.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/medical_evaluation/judges/cp09_cp11.py config/rubric.yaml tests/test_judges_cp09_cp11.py tests/test_judges_all.py tests/test_pipeline_vertical.py tests/test_vlm_client.py
git commit -m "feat: judge CP09 relative to oral region"
```

### Task 5: Update smoke summary for two prompt types

**Files:**
- Modify: `scripts/smoke_sam2_cp09.py`
- Modify: `tests/test_smoke_sam2_cp09.py`

- [ ] **Step 1: Write a failing smoke-summary test**

Add an `oral_region` `BoxPrompt` at `174.75` to `_annotations()`. Make `FakeExtractor` return all four new features. Assert:

```python
assert summary["prompt_counts"] == {
    "rubber_dam_frame": 1,
    "oral_region": 1,
}
assert "prompt_count" not in summary
assert summary["valid_frame_count"] == 3.0
assert summary["oral_region_valid_count"] == 3.0
assert summary["paired_valid_count"] == 3.0
assert decision["status"] == "correct"
```

- [ ] **Step 2: Run the smoke unit tests and verify RED**

Run:

```bash
python -m pytest tests/test_smoke_sam2_cp09.py -q
```

Expected: FAIL because the summary still has one strict `prompt_count` and the old feature key.

- [ ] **Step 3: Implement prompt counts and valid counts**

Count accepted prompts for each object using the CP09 `±0.5` second interval and write:

```python
"prompt_counts": {
    "rubber_dam_frame": frame_prompt_count,
    "oral_region": oral_prompt_count,
},
"valid_frame_count": result.features.get("frame_valid_count", 0.0),
"oral_region_valid_count": result.features.get("oral_region_valid_count", 0.0),
"paired_valid_count": result.features.get("paired_valid_count", 0.0),
```

Keep all existing model version, timing, sampling, and degradation fields.

- [ ] **Step 4: Run smoke tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_smoke_sam2_cp09.py -q
```

Expected: all smoke unit tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_sam2_cp09.py tests/test_smoke_sam2_cp09.py
git commit -m "feat: report dual-object CP09 smoke evidence"
```

### Task 6: Full verification and server handoff documentation

**Files:**
- Modify: `docs/server-runbook.md`

- [ ] **Step 1: Update the runbook expected outputs**

Document that CP09 now requires both prompt objects, accepts the oral box up to `0.5` seconds outside the stage boundary, writes object-specific mask directories, and reports `frame_oral_center_offset` plus three valid-count fields.

- [ ] **Step 2: Run focused and full CPU verification**

Run:

```bash
python -m pytest tests/test_segmentation_prompts.py tests/test_features.py tests/test_cp09_extractor.py tests/test_judges_cp09_cp11.py tests/test_smoke_sam2_cp09.py -q
python -m pytest -q
python -m ruff check src tests scripts
```

Expected: all tests PASS and Ruff reports no errors.

- [ ] **Step 3: Verify removed CP09 semantics do not remain in active code**

Run:

```bash
rg -n "frame_center_offset|max_center_offset|frame_centered" src config scripts tests
```

Expected: no active CP09 references remain; the generic `frame_center_offset` geometry helper may remain only if another caller or its own compatibility test still uses it.

- [ ] **Step 4: Commit runbook changes**

```bash
git add docs/server-runbook.md
git commit -m "docs: run dual-object CP09 smoke test"
```

- [ ] **Step 5: Prepare server transfer and GPU acceptance commands**

After pushing or bundling the branch, run on the server:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/smoke_sam2_cp09.py \
  --checkpoint-path external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --model-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda:0 \
  --sample-fps 2 \
  --videos-dir "$HOME/medical_evaluation/videos" \
  --data-dir "$HOME/medical_evaluation/data"
```

Expected: exit code `0`, at least three paired valid frames, two object-specific mask directories, three combined overlays, and `decision.json` containing `frame_oral_center_offset` rather than `frame_center_offset`.

- [ ] **Step 6: Perform human visual acceptance**

Inspect all three representative overlays and both mask sets. Accept the vertical slice only when `rubber_dam_frame` covers the white frame and `oral_region` covers the intended whole oral region throughout the sampled evidence. Do not accept a `correct` decision when either mask is semantically wrong.

