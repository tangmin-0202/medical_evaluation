# CP04 Local-Crop SAM3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fragmented OpenCV clamp masks with complete SAM3 masks produced from automatically isolated hand-and-object crops.

**Architecture:** The existing full-stage SAM3 session continues to segment the displayed white glove. Deterministic OpenCV logic ranks stable display frames and constructs fixed-size crops that retain original glove-and-object pixels while replacing everything else with a uniform background. A second SAM3 session processes only a 3–5 frame crop clip; its masks are mapped back to original coordinates and compared with references built by the same path.

**Tech Stack:** Python 3.11/3.12, OpenCV, NumPy, existing `VideoSegmenter`/SAM3 backend, pytest, Ruff.

---

### Task 1: Select stable display frames and build reversible hand crops

**Files:**
- Create: `src/medical_evaluation/features/cp04_display.py`
- Create: `tests/features/test_cp04_display.py`

- [ ] **Step 1: Write failing crop and stability tests**

Define synthetic tests proving that `build_hand_object_crop` retains a metal object excluded as a hole from the hand mask, replaces an outside plier with the uniform background, and maps a crop mask back to its original coordinates. Define a stability test with three adjacent observations at nearly equal hand-relative centers and one distant observation; assert that only the stable group is selected.

```python
crop = build_hand_object_crop(frame, hand_mask, seed_mask, output_size=320)
assert np.any(crop.image_bgr[object_pixels] != crop.background_bgr)
assert np.all(crop.image_bgr[outside_plier_pixels] == crop.background_bgr)
restored = crop.restore_mask(local_mask)
assert restored.shape == hand_mask.shape
assert restored[object_center]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest tests/features/test_cp04_display.py -q
```

Expected: collection fails because `medical_evaluation.features.cp04_display` does not exist.

- [ ] **Step 3: Implement focused data types and functions**

Create immutable `DisplayFrameCandidate` and `HandObjectCrop` dataclasses. Implement:

```python
def build_hand_object_crop(
    frame_bgr: np.ndarray,
    hand_mask: np.ndarray,
    seed_mask: np.ndarray,
    *,
    output_size: int = 640,
) -> HandObjectCrop: ...

def select_stable_display_frames(
    candidates: list[DisplayFrameCandidate],
    *,
    maximum_frames: int = 5,
    minimum_stable_frames: int = 2,
) -> list[DisplayFrameCandidate]: ...
```

The crop implementation must fill only enclosed holes in the selected hand component, dilate the support region by 1% of the shorter image side, crop the hand bounding box with 8% padding, letterbox it to a fixed square, and store the exact resize/padding transform. The selector must reject boundary-clipped hands, blurry candidates, and isolated positions; stability uses seed-center distance normalized by hand bounding-box diagonal.

- [ ] **Step 4: Run focused tests**

Run the command from Step 2. Expected: all CP04 display tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/medical_evaluation/features/cp04_display.py tests/features/test_cp04_display.py
git commit -m "feat: build reversible CP04 hand crops"
```

### Task 2: Build and segment the short local crop clip

**Files:**
- Modify: `src/medical_evaluation/extractors/cp04.py`
- Modify: `tests/test_cp04_extractor.py`

- [ ] **Step 1: Write failing extractor tests**

Add a fake-segmenter test with one full-stage hand call and one local-crop call. Assert that the second prompt is exactly:

```python
SegmentationPrompt(
    object_id="cp04_clamp_local",
    kind="text",
    frame_time_sec=0.0,
    text="small metal object on white glove",
)
```

Assert that the second video path is the saved crop clip, contains between 3 and 5 frames, and never contains pixels from a synthetic outside plier. Return complete local clamp masks from the fake segmenter and assert that mapped masks overlap the original-frame clamp positions.

- [ ] **Step 2: Run the new extractor test and verify failure**

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest tests/test_cp04_extractor.py -q
```

Expected: failure because the extractor still compares OpenCV seed fragments directly.

- [ ] **Step 3: Implement local clip processing**

In `_measure_hand_items`, retain OpenCV results only as `DisplayFrameCandidate` seeds. Select the stable 3–5 frame group, build 640×640 crops, and save an auditable MP4 under `cp_04/local_input/display_clip.mp4` using `mp4v` at 2 FPS. Run exactly one local SAM3 text session over that clip. Use each output `sample_position` (falling back to clip frame index) to select the corresponding `HandObjectCrop`, restore the local mask to the original frame, and reject masks that are outside hand support, too small, boundary-clipped, or elongated like a tool.

Use these constants:

```python
LOCAL_CLAMP_PROMPT = "small metal object on white glove"
LOCAL_CLAMP_OBJECT_ID = "cp04_clamp_local"
maximum_local_frames = 5
minimum_reliable_frames = 2
```

If fewer than two reliable complete masks remain, expose unreliable shape evidence instead of falling back to OpenCV fragments.

- [ ] **Step 4: Run extractor and feature tests**

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest tests/test_cp04_extractor.py tests/features/test_cp04_clamp_features.py tests/features/test_cp04_display.py -q
```

Expected: all selected tests pass and fake call count is exactly two.

- [ ] **Step 5: Commit**

```powershell
git add src/medical_evaluation/extractors/cp04.py tests/test_cp04_extractor.py
git commit -m "feat: segment CP04 clamp from local hand clip"
```

### Task 3: Rebuild references and preserve complete-mask evidence

**Files:**
- Modify: `src/medical_evaluation/extractors/cp04.py`
- Modify: `tests/scripts/test_build_cp04_reference.py`
- Modify after GPU generation: `config/cp04_reference.v1/manifest.json`
- Modify after GPU generation: `config/cp04_reference.v1/masks/*.png`

- [ ] **Step 1: Write failing reference-builder assertions**

Assert that reference generation uses the same two SAM3 sessions as production, records `candidate_source` as `sam3:local-hand-crop`, and refuses to write a reference if fewer than two complete local masks remain.

- [ ] **Step 2: Run the builder tests and verify failure**

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest tests/scripts/test_build_cp04_reference.py -q
```

Expected: failure while the builder still accepts OpenCV fragment masks.

- [ ] **Step 3: Route reference building through local SAM3 results**

Reuse the production crop-selection and local-segmentation function. Remove the OpenCV-fragment fallback from `build_reference`. Save reference rows only for complete local SAM3 masks and record both original time and crop-clip position in the manifest.

- [ ] **Step 4: Run focused tests and Ruff**

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest tests/scripts/test_build_cp04_reference.py tests/test_cp04_extractor.py -q
& D:\Medical_evaluation\.venv\Scripts\python.exe -m ruff check src/medical_evaluation/features/cp04_display.py src/medical_evaluation/extractors/cp04.py tests/features/test_cp04_display.py tests/test_cp04_extractor.py tests/scripts/test_build_cp04_reference.py --no-cache
```

Expected: all tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 5: Commit**

```powershell
git add src/medical_evaluation/extractors/cp04.py tests/scripts/test_build_cp04_reference.py
git commit -m "fix: require complete CP04 reference masks"
```

### Task 4: Verify locally and on the three real videos

**Files:**
- Modify: `D:\Medical_evaluation\AGENTS.md` after results are known; do not add it to Git.
- Generated evidence: `data/jobs/<job-id>/cp_04/`

- [ ] **Step 1: Run the complete Windows test suite**

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass; only the known Starlette/httpx and pytest-cache warnings may remain.

- [ ] **Step 2: Push or bundle-sync the tested commits**

Push `codex/sam3-cp09-cp11-text`. If server GitHub access fails, create a verified incremental bundle, upload it, preserve the server's dirty `rubric.py`, and use `git merge --ff-only`.

- [ ] **Step 3: Rebuild the success reference on fixed GPU5**

```bash
/home/tangm/.local/bin/gpu5-run env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/src" \
  /home/tangm/miniconda3/envs/sam3_medical/bin/python \
  scripts/build_cp04_reference.py --replace --job-id cp04-local-reference
```

Inspect every saved overlay before accepting the reference. Cyan contours must cover the clamp body rather than isolated reflections.

- [ ] **Step 4: Run success, failure, and clamp_failure**

Run `scripts/run_real_cp09_cp11.py --only cp_04 --no-commentary` for each video with distinct final job IDs. Expected evidence-based results: success is correct; failure is incorrect with zero reference matches; clamp_failure is correct if its displayed clamp has the same full silhouette as success.

- [ ] **Step 5: Download and inspect three evidence images per video**

Verify green hand contours, cyan full-clamp contours, normalized masks, similarity values, and reason codes. Do not accept a result whose cyan contour contains only reflection fragments.

- [ ] **Step 6: Update handoff and commit generated reference assets**

Record exact commits, job IDs, statuses, similarities, evidence paths, remaining limitations, server HEAD, and protected `rubric.py` hash in `D:\Medical_evaluation\AGENTS.md`. Commit only the reviewed reference manifest/masks and source/tests; do not add user bundles, experiments, or `AGENTS.md`.
