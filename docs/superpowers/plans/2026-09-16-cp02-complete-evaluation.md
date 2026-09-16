# CP02 Complete Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing CP02 SAM3 punch-mask feasibility gate into an auditable, deterministic evaluation of the second-largest punch hole and any green residue cleanup.

**Architecture:** Reuse the automatic moving-punch seed and SAM3 masks, but measure each video frame in its own disk coordinates. Separate disk/hole observations, final punch-hole alignment, residue/cleanup observations, and Judge decisions. If a required geometric or temporal relationship is not reliable, emit `needs_review`; never infer a label from video name or a SAM3 mask alone.

**Tech Stack:** Python, OpenCV, NumPy, SAM3.1, pytest, Ruff, current pipeline and report models.

**2026-09-16 checkpoint:** The punch-mask feasibility gate is not a hole-selection gate. Real success frames 850/875/900 and failure frames 1075/1100/1125/1400/1425/1450 were replayed locally. A lower press mechanism and glove can each create false circular dark-spot candidates. Synthetic guards now cover the lower mechanism, and the focused/full unit suite passes; no second-largest-hole identity or final tip alignment is established. The visible holes in the original 1920×1080 video occupy only a few pixels each, and several are occluded by the central plunger. Do not activate CP02 scoring, claim Task 1 complete, or use success/failure filenames as classification inputs until an auditable real-frame ranking and alignment exists. Current local changes have not been pushed or GPU-rerun.

**User correction and timing check:** Judge the hole at its **last adjusted position before punching**, not from the obscured post-punch frame. Success CP02 samples end at about 22.5 s; failure CP02 annotations end at 50 s, but original frames show the wheel still being handled at 50.5–53 s in the gap before CP03 begins at 55 s. Therefore scan CP02 through the immediate pre-CP03 boundary (without changing the annotation JSON), find the last reliable adjustment/stable interval, and verify no later re-adjustment before punch contact. A sharp earlier view may establish hole-size identity, but it must be tracked to that last position; never score an earlier transient alignment.

---

## Task 1: Establish a real-video evidence map and guard against false hole candidates

**Files:** `src/medical_evaluation/features/cp02_punch.py`, `tests/features/test_cp02_punch_features.py`, `scripts/run_early_cp_sam3_gate.py`, `tests/scripts/test_early_cp_gate.py`.

- [ ] Add a failing test in `tests/features/test_cp02_punch_features.py`: a mask covering only the lower plier mechanism with three dark highlights must not produce a disk face; a mask containing the solid round disk with holes must produce a disk face. Use the public result to assert its center, candidate hole count, and reliability.
- [ ] Run `D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/features/test_cp02_punch_features.py -q` and verify that the new assertion fails for the intended reason.
- [ ] Implement a per-frame `DiskFace` result and `locate_disk_face_in_mask(frame_bgr, mask, seed_radius)` that requires a circular metal face, several spatially distinct dark hole candidates, and mask support. The result must distinguish a real disk face from dark highlights on the lower mechanism; do not use fixed screen coordinates.
- [ ] Replace the gate's Hough-only `visible_disk_center` acceptance with `DiskFace` evidence, retaining raw masks, overlays and rejection reasons. Run the focused tests again and keep success/failure GPU evidence for visual review.
- [ ] Commit only these code/test files after `ruff check --no-cache` passes.

## Task 2: Detect and rank the actual holes in the same disk plane

**Files:** `src/medical_evaluation/features/cp02_punch.py`, `tests/features/test_cp02_punch_features.py`.

- [ ] Write a failing test with a rotated/foreshortened synthetic multi-hole disk. Assert that its true second-largest hole remains second after affine rectification; ambiguous adjacent radii return no selection. Include a non-disk dark reflection negative case.
- [ ] Run the focused test and observe RED.
- [ ] Add `DiskHoleObservation` with center, screen ellipse axes, rectified radius, mask/color confidence, and frame index. Fit the disk outline; rectify only when the outline and at least three interior holes are reliable. Rank using the existing `second_largest_hole` helper on rectified radii. Do not rank by screen x coordinate or unrectified radius.
- [ ] Run focused tests, Ruff, and apply the detector to both actual CP02 windows. Save per-frame hole overlays and a JSON table with radii, ranking confidence, and rejection reason.

## Task 3: Identify the final selected hole and action window

**Files:** `src/medical_evaluation/features/cp02_punch.py`, `src/medical_evaluation/extractors/cp02.py`, `tests/features/test_cp02_punch_features.py`, `tests/extractors/test_cp02.py`.

- [ ] Write failing tests for a punch tip aligned with a hole during the final stable pre-contact window, for transient passage over the correct hole followed by selection of a wrong hole, and for an occluded tip. The transient and occluded cases must not be marked correct.
- [ ] Run the tests and observe RED.
- [ ] Build `Cp02FeatureExtractor` to sample CP02 plus a narrow CP03 boundary window, reuse an automatically located SAM3 punch mask, detect tip/selected-hole geometry only in reliable frames, and aggregate the final stable alignment. Save raw frame, mask, selected-hole overlay, normalized measurements and reason code.
- [ ] Verify on success/failure videos without using their filenames as algorithm input. If the tiny tip or hole identity is unresolved, preserve `selected_second_largest=None` and `needs_review` rather than tune a threshold to match labels.

## Task 4: Residue and cleanup evidence

**Files:** `src/medical_evaluation/features/cp02_punch.py`, `src/medical_evaluation/extractors/cp02.py`, `tests/features/test_cp02_punch_features.py`, `tests/extractors/test_cp02.py`.

- [ ] Write failing tests for (a) no green residue, (b) same-hole green residue that disappears after probe contact, (c) residue without cleanup, (d) a probe touching another hole, and (e) color/occlusion ambiguity.
- [ ] Run tests and observe RED.
- [ ] Use same-frame green dam reference and the selected hole interior for residue appearance; detect probe contact with the selected hole and verify subsequent disappearance. Distinguish `False` from unavailable `None` in every feature. Save before/contact/after evidence and numeric color ratios.
- [ ] Run focused tests, Ruff, and the two real CP02 windows; document any branch that stays `needs_review` due inadequate evidence.

## Task 5: Integrate the deterministic Judge and report

**Files:** `config/rubric.yaml`, `src/medical_evaluation/runtime.py`, `src/medical_evaluation/pipeline.py`, `src/medical_evaluation/extractors/cp09_cp11.py` or a new routing module, `tests/test_runtime.py`, `tests/test_pipeline.py`.

- [ ] Write failing integration tests asserting that CP02 uses `judges/cp02_punch.py`, is included as a fourth evaluated item only when actual extraction ran, leaves `final_score=null`, and uses `needs_review` when selection/cleanup evidence is unreliable. Ensure the old left-to-right CP02 Judge is not used for this path.
- [ ] Run the tests and observe RED.
- [ ] Update `config/rubric.yaml` CP02 criteria to “second-largest hole by diameter” and conditional residue cleanup. Route CP02 to the new extractor and Judge without modifying CP09–CP11 behavior; update the audit version string.
- [ ] Run full Windows tests and Ruff, then commit/push the feature branch. Safely fast-forward the new server after checking worktree/GPU state. Run success and failure end-to-end, inspect report JSON and evidence images, and preserve all logs/artifacts. Do not call CP02 complete if the real-data branch remains `needs_review` or if hole identity is inferred from a label.
