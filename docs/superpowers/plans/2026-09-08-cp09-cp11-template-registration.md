# CP09/CP11 Template Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate CP09 frame installation and CP11 nose clearance/frame coverage without per-video oral or nose boxes and without CP09-to-CP11 frame tracking.

**Architecture:** SAM3 head masks constrain a versioned mannequin-template registration; a four-degree-of-freedom similarity transform maps template nose and frame regions into each frame. CP09 stores its actual frame mask in template coordinates, while CP11 maps that region back through its independently estimated head transform and combines it with the dam mask. Deterministic Judge, scoring, and Qwen boundaries stay unchanged.

**Tech Stack:** Python, NumPy, OpenCV, Pydantic, SAM3.1, pytest, Ruff, Git, SSH.

---

### Task 1: Add and validate pure similarity registration

**Files:**
- Create: `src/medical_evaluation/features/mannequin_registration.py`
- Create: `tests/test_mannequin_registration.py`

- [ ] Write tests for exact synthetic translation/rotation/scale recovery, inverse composition `M11 @ inverse(M09)`, polygon/mask warping, insufficient matches, invalid scale, high residual, and temporal jump rejection.
- [ ] Run `D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest tests/test_mannequin_registration.py -q` and confirm failures are caused by the missing API.
- [ ] Implement immutable `RegistrationResult`, similarity-matrix validation, robust feature/contour registration, mapping helpers, and deterministic quality gates. Return a failed result with metrics rather than silently using identity.
- [ ] Run the focused test and `D:\Medical_evaluation\.venv\Scripts\ruff.exe check --no-cache` on the new files, then commit.

### Task 2: Reuse prior SAM3 artifacts for the real registration gate

**Files:**
- Create: `scripts/run_head_template_registration.py`
- Create: `tests/scripts/test_run_head_template_registration.py`
- Create as untracked runtime data: `config/mannequin_template.v1.json` and its referenced image/mask under the server run directory

- [ ] Write failing tests for manifest validation, template provenance, result summaries, quality rejection, and overlay generation.
- [ ] Implement a CLI that reads the saved raw/mask directories from `head-nose-prompt-gate-897484d`, estimates transforms, maps a versioned nose polygon, and writes `summary.json`, per-frame matrices/metrics, and overlays.
- [ ] Run local focused tests and Ruff, commit, push, and fast-forward the server while preserving its dirty `rubric.py`.
- [ ] Run the CLI against all three videos' CP09/CP11 saved frames. Inspect continuous overlays; preserve failed runs and repair reproducible registration defects with a failing test first.

### Task 3: Persist current-video and canonical regions

**Files:**
- Create: `src/medical_evaluation/frame_reference.py`
- Create: `tests/test_frame_reference.py`
- Create: `config/mannequin_template.v1.json`

- [ ] Test and implement a versioned template schema containing reference frame/mask identity, nose polygon, canonical frame polygon, provenance, and `demo_only`.
- [ ] Test and implement an atomic per-job CP09 reference store containing the actual frame polygon in template coordinates, stable-window times, appearance Lab value, and registration metrics.
- [ ] Prefer the current-video CP09 reference in CP11 and use the canonical reference only when current evidence is unavailable; record the selected source.
- [ ] Run focused tests and commit.

### Task 4: Replace CP09 extraction and judgment

**Files:**
- Modify: `src/medical_evaluation/segmentation/prompt_policy.py`
- Modify: `src/medical_evaluation/extractors/cp09.py`
- Modify: `src/medical_evaluation/judges/cp09_cp11.py`
- Modify: `tests/test_segmentation_prompt_policy.py`
- Rewrite: `tests/test_cp09_extractor.py`
- Modify: `tests/test_judges_cp09_cp11.py`

- [ ] Write failing tests for two independent SAM3 sessions, registration quality, stable-tail selection, reference persistence, and every approved CP09 status priority.
- [ ] Track head and frame over CP09, join only equal source frame indices, register accepted head frames, convert frame masks to template coordinates, and aggregate a final stable window.
- [ ] Remove SAM3 dependence on `oral_region`; emit presence, registration, stability, center, angle, scale and dispersion features plus three auditable overlays.
- [ ] Implement Judge review bands and exact reason codes, run focused tests, and commit.

### Task 5: Replace CP11 extraction and judgment

**Files:**
- Modify: `src/medical_evaluation/extractors/cp11.py`
- Modify: `src/medical_evaluation/judges/cp09_cp11.py`
- Rewrite: `tests/test_cp11_extractor.py`
- Modify: `tests/test_judges_cp09_cp11.py`

- [ ] Write failing tests proving CP11 calls only head and dam sessions, does not require `nose_region`, does not long-track the frame, and maps both nose and frame polygons through the accepted transform.
- [ ] Retain the whole-stage green presence scan. In the final candidate region, register head frames, find the latest stable window, load the best frame reference, and calculate nose overlap, frame dam coverage, and white exposure within the projected region.
- [ ] Remove whole-frame `dam_area_ratio`, manual nose boxes, and old CP09-to-CP11 segmentation calls. Implement exact state priority and review bands.
- [ ] Run focused tests and commit.

### Task 6: Wire configuration, runtime, audit, and compatibility

**Files:**
- Modify: `src/medical_evaluation/settings.py`
- Modify: `config/rubric.yaml`
- Modify: `src/medical_evaluation/runtime.py`
- Modify: `src/medical_evaluation/segmentation/audit.py`
- Modify carefully after comparison: `src/medical_evaluation/rubric.py`
- Modify corresponding tests.

- [ ] Write failing tests for template loading, shared per-job reference store, audit fields, missing/mismatched template versions, and explicit SAM2 unsupported-evidence review behavior.
- [ ] Add versioned Demo registration, stability, coverage, visibility and review-band settings. Preserve existing report readers and segmentation metadata fields.
- [ ] Reconcile only required rubric mappings with the saved server patch; do not overwrite the server's existing dirty change.
- [ ] Run integration tests and commit.

### Task 7: Calibrate and validate all three real pipelines

**Files:**
- Modify from measured evidence: `config/rubric.yaml`
- Modify from measured evidence: `config/mannequin_template.v1.json`
- Update: `docs/server-runbook.md`
- Preserve runtime artifacts under `data/jobs/` and `artifacts/.qa/head-relative-final/`.

- [ ] Run success feature-only calibration on an available GPU, inspect overlays, and freeze v1 medians/tolerances with provenance and `demo_only: true`.
- [ ] Run failure and clamp_failure with their own videos and stage ranges. Fix extraction defects from evidence; do not tune thresholds simply to match expected labels.
- [ ] Run complete local pytest and Ruff, push and verify GitHub hash, fast-forward and verify server hash, then run server focused/full regressions.
- [ ] Run all three complete SAM3 CP09/CP11 pipelines with Qwen on a separate GPU. Inspect masks, mapped regions, stable windows, numerical features, Judge reasons, evidence paths, `2/11`, provisional score and `final_score=null`.
- [ ] Save final hashes, test logs, GPU peaks and visual evidence. State only Demo feasibility for these samples.
