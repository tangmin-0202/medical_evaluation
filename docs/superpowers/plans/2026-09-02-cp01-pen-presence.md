# CP01 Pen Presence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CP01 pen/dam overlap gating with stable pen presence, then rank stable full-stage dark marks by darkness and reference distance.

**Architecture:** Keep SAM2 tracking for `rubber_dam` and optional `marking_pen`. The extractor counts non-empty pen masks, clusters all full-stage mark observations, and reuses `select_darkest_nearest_reference`; the deterministic Judge consumes the renamed presence and candidate features.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Pydantic, pytest, SAM2 protocol.

---

### Task 1: Lock the new extractor and Judge contract

**Files:**
- Modify: `tests/test_cp01_extractor.py`
- Modify: `tests/test_cp01_judge.py`
- Modify: `tests/test_smoke_cp01_cp11.py`

- [ ] Replace contact fixtures with pen masks that are present but do not overlap the dam.
- [ ] Assert `pen_presence_detected`, `mark_candidate_count`, selected mark coordinates, and distance.
- [ ] Assert missing pen is `incomplete`, present pen without stable mark is `incorrect`, and a near/far stable mark is correct/incorrect.
- [ ] Assert smoke construction injects `min_pen_presence_frames` instead of overlap/contact thresholds.
- [ ] Run the three targeted files and confirm they fail because production still emits contact fields.

### Task 2: Implement pen-presence extraction

**Files:**
- Modify: `src/medical_evaluation/extractors/cp01.py`
- Modify: `src/medical_evaluation/judges/cp01_cp03.py`
- Modify: `scripts/smoke_sam2_cp01_cp11.py`
- Modify: `config/rubric.yaml`

- [ ] Count non-empty pen masks and set presence after `min_pen_presence_frames`.
- [ ] Cluster all stage observations with `cluster_stable_marks` and pass them directly to `select_darkest_nearest_reference`.
- [ ] Remove contact/preexisting/new-mark fields and emit the design feature names.
- [ ] Update Judge reasons and matched rules to describe pen appearance and mark position.
- [ ] Replace obsolete CP01 thresholds with `min_pen_presence_frames: 2`.

### Task 3: Synchronize diagnostics and documentation

**Files:**
- Modify: `scripts/calibrate_cp01_detector.py`
- Modify: `docs/server-runbook.md`
- Modify: related targeted tests only where existing assertions encode the old feature names.

- [ ] Make calibration rank stable full-stage marks without a contact split.
- [ ] Update overlay/runbook terminology from contact/background/new mark to pen presence/Top 2.
- [ ] Run only CP01 extractor, Judge, mark-selection, calibration, rubric, and combined-smoke tests.
- [ ] Run `git diff --check`, commit, merge to `main`, push GitHub, and create an incremental server bundle.
