# CP08 Production Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CP08 as a production-evaluated checkpoint using automatic instrument-shape evidence from CP08 and two-wing/two-hole rubber-dam evidence from the CP09 tail.

**Architecture:** A dedicated `Cp08FeatureExtractor` uses independent SAM3 text sessions for the action instrument, target tooth, full clamp, and rubber dam. Pure OpenCV helpers convert masks and original frames into deterministic instrument-shape, wing-completeness, and per-hole adaptive dam-color features. `judge_cp08` owns the ordered state machine; CP09 scoring remains independent.

**Tech Stack:** Python, NumPy, OpenCV, SAM3 `VideoSegmenter`, Pydantic reporting models, pytest, Ruff, Windows CPU tests, Ubuntu RTX A5000 GPU5 MPS validation.

---

### Task 1: Replace the legacy CP08 Judge contract

**Files:**
- Modify: `tests/test_judges_all.py`
- Modify: `tests/test_rubric.py`
- Modify: `src/medical_evaluation/judges/cp07_cp08.py`
- Modify: `config/rubric.yaml`

- [ ] Add parameterized failing tests for these ordered branches: `cp08_not_performed`, `unreliable_instrument_shape`, `wrong_instrument_shape`, `final_state_unobservable`, `rubber_dam_not_positioned`, `clamp_wing_not_fully_visible`, `non_dam_color_under_wing_hole`, `unreliable_wing_hole_color`, and `criteria_satisfied`.
- [ ] Run `python -m pytest tests/test_judges_all.py tests/test_rubric.py -q` and confirm failures reference the legacy CP08 feature names.
- [ ] Implement the ordered Judge using `instrument_observed`, `instrument_shape_reliable`, `instrument_shape_match`, `final_state_observable`, `rubber_dam_positioned`, `left_wing_complete`, `right_wing_complete`, `left_wing_hole_detected`, `right_wing_hole_detected`, and per-side dam/non-dam color ratios.
- [ ] Replace `max_green_wing_hole_ratio` with `min_wing_hole_dam_color_ratio`, `max_wing_hole_non_dam_color_ratio`, and `min_wing_hole_valid_frames`; remove the tooth-neck criterion and blunt-tip claim from rubric text.
- [ ] Re-run focused tests and commit the Judge/rubric slice.

### Task 2: Add deterministic instrument and clamp-wing features

**Files:**
- Create: `src/medical_evaluation/features/cp08_positioning.py`
- Create: `tests/features/test_cp08_positioning.py`

- [ ] Write failing synthetic-mask tests showing that a valid instrument mask has a dominant long component, a thin working section, and a curved centerline; a straight rod, compact blob, or fragmented mask must not match. Add the real-gate regression where SAM3 returns the desired long curved instrument plus a disconnected compact U-shaped clamp: only the instrument component may contribute to shape evidence. If they merge and cannot be separated reliably, return unreliable rather than forcing a match.
- [ ] Write failing tests for `split_clamp_wings(tooth_mask, clamp_mask)`: both sides present returns two wing masks; a missing side returns `None` for that side; central clamp body is excluded from wing completeness.
- [ ] Write failing tests for `measure_wing_hole_color(frame, wing_mask, dam_mask)`: green/dam-matching interiors pass independently, one non-dam interior fails independently, missing hole returns unreliable, and metal rims are excluded by inward sampling.
- [ ] Implement immutable result dataclasses and minimal OpenCV operations: largest-component filtering, skeleton/polyline curvature approximation, tooth-relative clamp axis, per-side contour completeness, Hough/contour hole candidates, and Lab/HSV distance to the same-frame dam reference.
- [ ] Save no thresholds in these helpers beyond geometric validity constants; Judge thresholds remain in rubric.
- [ ] Run `python -m pytest tests/features/test_cp08_positioning.py -q` and commit the feature slice.

### Task 3: Build the cross-stage CP08 extractor

**Files:**
- Create: `src/medical_evaluation/extractors/cp08.py`
- Create: `tests/test_cp08_extractor.py`
- Modify: `src/medical_evaluation/extractors/cp09_cp11.py`

- [ ] Add failing fake-segmenter tests proving: the entire CP08 range is searched sparsely; only an automatic candidate receives dense confirmation; CP09’s last three seconds are requested independently for tooth, full clamp, and dam; annotation point/box prompts are never read; CP09 Judge results are never consulted.
- [ ] Add failing evidence tests requiring original images, masks, overlays, per-frame metrics, selected prompt, and explicit failure reason under `cp_08/action/` and `cp_08/final/`.
- [ ] Implement `Cp08FeatureExtractor` with text prompts from the validated gate catalog. Produce deterministic features only after minimum valid-frame agreement; retain `None` for unreliable evidence rather than forcing a status.
- [ ] Extend the composite extractor with an optional CP08 delegate without changing CP02/CP09/CP10/CP11 behavior.
- [ ] Run extractor and existing composite tests; commit the extractor slice.

### Task 4: Integrate CP08 into runtime, reporting, and commentary

**Files:**
- Modify: `src/medical_evaluation/runtime.py`
- Modify: `src/medical_evaluation/pipeline.py`
- Modify: `src/medical_evaluation/vlm/client.py`
- Modify: `scripts/run_real_cp09_cp11.py`
- Modify: `tests/test_runtime.py`
- Modify: `tests/test_pipeline_vertical.py`
- Modify: `tests/test_vlm_client.py`

- [ ] Add failing integration tests expecting CP08 in the enabled set, the composite model version, extraction dispatch, five-of-eleven provisional scoring, and CP08 commentary that says “符合约定外形” rather than claiming a proven blunt tip.
- [ ] Construct CP08 with the shared audited SAM3 segmenter and text prompt policy; enable `cp_08`; allow `--only cp_08` in the real runner.
- [ ] Update pipeline audit label and fallback commentary while preserving deterministic Judge ownership.
- [ ] Run all CP08/integration tests and commit the integration slice.

### Task 5: Validate on GPU5 and calibrate only measured thresholds

**Files:**
- Modify only if real evidence requires a tested correction: CP08 feature/extractor tests and their production counterparts.
- Update: `D:\Medical_evaluation\AGENTS.md` after final acceptance; do not add it to Git.

- [ ] Verify GPU5 exclusive MPS state and run the existing success gate with `gpu5-run env CUDA_VISIBLE_DEVICES=0`; inspect selected overlays at original resolution.
- [ ] Run CP08-only reports for success, failure, and clamp_failure. Preserve every prompt, mask, overlay, per-side hole measurement, feature, reason code, elapsed time, and peak memory.
- [ ] If a reproducible defect appears, add a failing regression test before changing code; never tune a threshold solely to match video filenames.
- [ ] Run Windows focused tests, full pytest, Ruff, and `git diff --check`; run server focused tests and all three real videos again on the final commit.
- [ ] Request code review, commit/push the final branch, fast-forward `.102` without touching its dirty `rubric.py`, visually verify evidence, and update `AGENTS.md` with exact results and limitations.
