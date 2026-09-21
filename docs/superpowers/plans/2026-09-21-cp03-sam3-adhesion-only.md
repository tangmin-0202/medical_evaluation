# CP03 SAM3 Adhesion-Only Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CP03 to the production pipeline using SAM3 rubber-dam segmentation and a deterministic adhesion-only hole result Judge.

**Architecture:** A focused `Cp03FeatureExtractor` samples the final four seconds of CP03 at 2 FPS, expands to eight seconds only when three consecutive reliable hole frames are unavailable, and uses original pixels plus SAM3 dam masks to locate the same hole and detect connected green flaps. The existing composite extractor dispatches CP03, while the deterministic Judge consumes only observation reliability, consecutive-frame count, and adhesion status.

**Tech Stack:** Python 3.11/3.12, OpenCV, NumPy, SAM3 `VideoSegmenter`, Pydantic reporting models, pytest, Ruff.

---

### Task 1: Replace CP03 quality scoring with adhesion-only Judge semantics

**Files:**
- Modify: `tests/judges/test_cp03_result_judge.py`
- Modify: `src/medical_evaluation/judges/cp03_hole.py`

- [ ] **Step 1: Write failing Judge tests**

Add cases showing that `hole_round` and `hole_complete` are ignored, adhesion produces `incorrect/hole_adhesion_detected`, and unreliable adhesion produces `needs_review/unreliable_adhesion_observation`.

```python
def test_non_round_hole_without_adhesion_is_correct():
    features = {
        "final_scan_reliable": True,
        "hole_observed": True,
        "hole_clear_consecutive_frames": 3,
        "hole_adhesion_free": True,
        "hole_round": False,
    }
    result = judge_cp03_hole(features, {"min_clear_hole_frames": 3})
    assert (result.status, result.reason_code) == ("correct", "criteria_satisfied")
```

- [ ] **Step 2: Run the Judge test and verify the expected failure**

Run: `D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/judges/test_cp03_result_judge.py -q`

Expected: the non-round case and new reason-code assertions fail against the old three-quality-field Judge.

- [ ] **Step 3: Implement the minimal adhesion-only Judge**

Judge order: unreliable scan → unobserved hole → unreliable observation → insufficient consecutive frames → unreliable adhesion → detected adhesion → correct. Remove all reads of `hole_complete` and `hole_round`.

- [ ] **Step 4: Run the focused Judge test**

Expected: all CP03 Judge tests pass.

### Task 2: Detect connected rubber-dam flaps from a SAM3 dam mask and source pixels

**Files:**
- Modify: `src/medical_evaluation/features/cp03_hole.py`
- Modify: `tests/features/test_cp03_hole_features.py`

- [ ] **Step 1: Add failing synthetic feature tests**

Create synthetic green dam images and masks for: a clear hole without a flap, a hole with a connected green flap, an isolated green speck that must not count as adhesion, an edge-clipped candidate, and a frame with no hole. Define a focused API:

```python
result = analyze_hole_adhesion(image_bgr, dam_mask)
assert result.hole_observed is True
assert result.adhesion_detected is False
```

- [ ] **Step 2: Run the feature tests and verify `analyze_hole_adhesion` is missing**

Run: `D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/features/test_cp03_hole_features.py -q`

Expected: failure caused by the missing adhesion analysis API.

- [ ] **Step 3: Implement the minimum feature API**

Add an immutable result record containing hole center, radius, observation reliability, adhesion status, and masks used for evidence. Detect interior hole candidates from closed non-dam components, reject border/noise candidates, then require a connected green structure extending from the hole boundary into the opening before setting `adhesion_detected=True`. Keep `measure_hole` for compatibility but do not use its roundness metrics for scoring.

- [ ] **Step 4: Run feature tests and refactor only after green**

Expected: synthetic CP03 feature tests pass with no regressions in existing geometry tests.

### Task 3: Add the CP03 final-window extractor and evidence output

**Files:**
- Create: `src/medical_evaluation/extractors/cp03.py`
- Create: `tests/test_cp03_extractor.py`

- [ ] **Step 1: Write failing extractor tests with a fake segmenter**

Test that the extractor:

- requests the final four seconds at exactly 2 FPS;
- expands to the final eight seconds only when fewer than three consecutive reliable frames exist;
- returns `hole_adhesion_free=False` when reliable frames contain a connected flap;
- returns an unreliable result for conflicts or persistent occlusion;
- saves raw frames, SAM3 masks, ROI/overlay images, and evidence items.

- [ ] **Step 2: Run the extractor tests and verify the module is missing**

Run: `D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_cp03_extractor.py -q`

Expected: collection/import failure for the not-yet-created extractor.

- [ ] **Step 3: Implement `Cp03FeatureExtractor`**

Use a text prompt for `rubber_dam`, call the injected `VideoSegmenter`, read matching original frames, analyze each frame, find the latest run of at least three spatially consistent hole observations, and write evidence below `cp_03/{originals,masks,rois,overlays}`. Limit report evidence to three representative frames while retaining all debug artifacts.

- [ ] **Step 4: Run extractor and feature tests**

Expected: all CP03 extractor, feature, and Judge tests pass.

### Task 4: Integrate CP03 into runtime, composite dispatch, rubric, and report audit

**Files:**
- Modify: `src/medical_evaluation/extractors/cp09_cp11.py`
- Modify: `src/medical_evaluation/runtime.py`
- Modify: `src/medical_evaluation/pipeline.py`
- Modify: `config/rubric.yaml`
- Modify: `tests/test_cp09_cp11_extractor.py`
- Modify: `tests/test_runtime.py`
- Modify: `tests/test_pipeline_vertical.py`

- [ ] **Step 1: Write failing integration tests**

Require CP03 dispatch, runtime construction sharing the audited segmenter, enabled set `cp_02/cp_03/cp_08/cp_09/cp_10/cp_11`, evaluated count 6, the new Judge mapping, and audit version `vertical-cp02-cp03-cp08-cp09-cp10-cp11`.

- [ ] **Step 2: Run integration tests and verify they fail on the five-checkpoint pipeline**

Run: `D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_cp09_cp11_extractor.py tests/test_runtime.py tests/test_pipeline_vertical.py -q`

Expected: failures for missing CP03 dispatch/construction/enabling and old Judge/audit metadata.

- [ ] **Step 3: Implement minimal integration**

Construct `Cp03FeatureExtractor` with the shared audited segmenter, add it to composite dispatch/model version, import `judge_cp03_hole`, enable CP03, and change rubric CP03 to `judge_type: hybrid`, required objects `rubber_dam` and `punched_hole`, criterion “打孔后的目标孔边缘无橡皮布薄片、翻边或残留粘连”, and `min_clear_hole_frames: 3`.

- [ ] **Step 4: Run all focused CP03 and integration tests**

Expected: focused suite passes.

### Task 5: Verify locally and run three real videos on fixed GPU5

**Files:**
- Modify only if a failing test or visual defect requires a tested correction.

- [ ] **Step 1: Run Ruff and the complete local test suite**

Run:

```powershell
& D:\Medical_evaluation\.venv\Scripts\ruff.exe check --no-cache src tests scripts/run_early_cp_sam3_gate.py
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
```

Expected: Ruff clean and all tests pass; existing documented warnings may remain.

- [ ] **Step 2: Commit the implementation and push the feature branch**

Stage only CP03/integration files and this plan. Push `codex/sam3-cp09-cp11-text`; if GitHub 443 fails, report it and use a verified incremental bundle for `.102` without altering unrelated files.

- [ ] **Step 3: Verify `.102` state before deployment**

Confirm branch/HEAD, the protected dirty `rubric.py` SHA-256 `14b3b33b010825f3afe605d9d77d12827f79b94c0833c6e3947c8d716903be77`, GPU5 exclusive/MPS state, and the `gpu5-run env CUDA_VISIBLE_DEVICES=0` CUDA smoke test. Do not reset, clean, overwrite `rubric.py`, or stop other users' GPU processes.

- [ ] **Step 4: Run focused server tests and three real CP03 evaluations**

Use the existing real-report runner or a CP03-focused runner under `/home/tangm/medical_evaluation`, always launching SAM3 through GPU5 MPS. Run success, failure, and clamp_failure with separate output directories.

- [ ] **Step 5: Inspect visual evidence and report exact outcomes**

Open representative original/mask/ROI/overlay images for each video, compare them with numeric features and reason codes, and report paths, runtime, limitations, and any `needs_review` result honestly. Do not infer truth from video filenames or old action labels.
