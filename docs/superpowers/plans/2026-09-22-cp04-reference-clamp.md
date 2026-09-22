# CP04 Reference Clamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically judge whether the clamp clearly displayed on a white-gloved palm matches the versioned clamp reference extracted from the `success` video.

**Architecture:** SAM3 runs independent text-prompt sessions for the white glove and the small metal clip. A focused OpenCV feature module gates candidates by their hand-relative position, normalizes masks for rotation/scale/reflection, compares them with saved `success` masks, and supplies deterministic multi-frame features to a CP04-only Judge. The runtime loads versioned reference masks and preserves frame-level images, masks, overlays, comparison panels, metrics, and reason codes.

**Tech Stack:** Python 3.11/3.12, NumPy, OpenCV, Pydantic, existing SAM3 `VideoSegmenter`, pytest, Ruff.

---

### Task 1: Clamp-mask normalization and reference comparison

**Files:**
- Create: `src/medical_evaluation/features/cp04_clamp.py`
- Create: `tests/features/test_cp04_clamp_features.py`

- [ ] **Step 1: Write failing invariance and rejection tests**

Create synthetic clamp masks with a central bow, two wings, two jaws, and two holes. Assert that `compare_clamp_mask()` accepts translated, scaled, rotated, and mirrored versions, while rejecting an elongated plier mask and a single solid rectangle. Also assert that `measure_display_candidate()` rejects a clip whose centroid is outside the dilated glove mask or whose mask touches the crop boundary.

```python
reference = synthetic_clamp_mask()
candidate = transform(reference, angle=73, scale=0.65, mirror=True)
match = compare_clamp_mask(candidate, [reference])
assert match.reliable is True
assert match.similarity >= 0.8

plier = synthetic_plier_mask()
assert compare_clamp_mask(plier, [reference]).similarity < 0.8
```

- [ ] **Step 2: Verify the focused tests fail**

Run: `python -m pytest tests/features/test_cp04_clamp_features.py -q`

Expected: FAIL because `medical_evaluation.features.cp04_clamp` does not exist.

- [ ] **Step 3: Implement immutable measurements and minimal OpenCV operations**

Define these public interfaces:

```python
@dataclass(frozen=True)
class ClampDisplayMeasurement:
    reliable: bool
    reason: str
    hand_proximity_ratio: float | None
    relative_area: float | None
    boundary_touch: bool
    sharpness: float | None

@dataclass(frozen=True)
class ClampMatchMeasurement:
    reliable: bool
    reason: str
    similarity: float | None
    mask_iou: float | None
    contour_similarity: float | None
    aspect_similarity: float | None

def measure_display_candidate(frame_bgr, hand_mask, clamp_mask) -> ClampDisplayMeasurement: ...
def normalize_clamp_mask(mask, *, size=160) -> np.ndarray: ...
def compare_clamp_mask(mask, reference_masks) -> ClampMatchMeasurement: ...
```

Use largest-component cleanup, a small hand-mask dilation, normalized centroid/area, PCA or `minAreaRect` orientation, and comparison over 0/180-degree and mirrored variants. Combine mask IoU, symmetric contour distance, and aspect similarity. Hole topology is recorded as an auxiliary measurement only when source resolution supports it; absence of a resolved hole alone must not force a wrong-clamp decision.

- [ ] **Step 4: Run feature tests**

Run: `python -m pytest tests/features/test_cp04_clamp_features.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the feature core**

```bash
git add src/medical_evaluation/features/cp04_clamp.py tests/features/test_cp04_clamp_features.py
git commit -m "feat: compare CP04 clamp silhouettes"
```

### Task 2: Deterministic CP04 Judge semantics

**Files:**
- Modify: `src/medical_evaluation/judges/cp04_cp06.py`
- Modify: `tests/test_judges_all.py`
- Modify: `config/rubric.yaml`

- [ ] **Step 1: Replace the old semantic-confirmation cases with the approved state machine**

Add focused cases for:

```python
correct_features = {
    "clamp_observed": True,
    "shape_evidence_reliable": True,
    "clear_frame_count": 3.0,
    "matching_frame_count": 3.0,
    "clamp_reference_similarity": 0.88,
    "evidence_consistent": True,
}
```

Cover `correct`, clear stable mismatch as `incorrect/wrong_clamp_type`, detected-but-unclear as `needs_review/unreliable_clamp_shape`, conflicting frames as `needs_review/inconsistent_clamp_evidence`, and no displayed clamp as `incomplete/clamp_not_observed`.

- [ ] **Step 2: Run the Judge tests and confirm failure**

Run: `python -m pytest tests/test_judges_all.py -q`

Expected: FAIL because the old Judge still requires `semantic_confirmed`.

- [ ] **Step 3: Implement the approved decision order**

Use rubric thresholds `min_clear_clamp_frames`, `min_matching_clamp_frames`, and `min_clamp_similarity`. Remove `semantic_confirmed`; Qwen remains commentary-only.

- [ ] **Step 4: Update CP04 rubric text and run tests**

The rubric criteria must say that the displayed clamp matches the versioned `success` reference after rotation/scale/reflection normalization. Run:

`python -m pytest tests/test_judges_all.py tests/test_rubric.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Judge and rubric changes**

```bash
git add src/medical_evaluation/judges/cp04_cp06.py tests/test_judges_all.py config/rubric.yaml
git commit -m "feat: judge CP04 reference clamp match"
```

### Task 3: CP04 presentation extractor and evidence

**Files:**
- Create: `src/medical_evaluation/extractors/cp04.py`
- Create: `tests/test_cp04_extractor.py`
- Create: `config/cp04_reference.v1/manifest.json`
- Generate after GPU validation: `config/cp04_reference.v1/masks/*.png`

- [ ] **Step 1: Write fake-segmenter extractor tests**

Test independent prompt sessions and assert the exact non-professional prompts:

```python
HAND_PROMPT = "white gloved hand showing an object on the palm"
CLAMP_PROMPTS = (
    "small shiny metal clip resting on a white gloved palm",
    "small silver U-shaped metal clip held by a white gloved hand",
    "small metal clip with two side wings displayed on a gloved palm",
)
```

Cover: two stable matching frames; clear mismatch; no clamp output; clamp on the rack outside the glove; oversized elongated plier; ambiguous text output; incomplete/blurred mask; evidence files and per-frame metrics.

- [ ] **Step 2: Run extractor tests and confirm failure**

Run: `python -m pytest tests/test_cp04_extractor.py -q`

Expected: FAIL because `Cp04FeatureExtractor` does not exist.

- [ ] **Step 3: Implement `Cp04FeatureExtractor`**

Use 2 FPS for the annotated CP04 interval. Run the hand and clamp prompts in separate SAM sessions, join by frame index, resize masks to source-frame resolution, gate each candidate with `measure_display_candidate()`, compare reliable candidates to all loaded reference masks, and select up to three highest-quality frames. Alternate clamp prompts run only if the earlier prompt yields no reliable candidates.

Return only deterministic features:

```python
{
    "clamp_observed": bool,
    "shape_evidence_reliable": bool,
    "clear_frame_count": float,
    "matching_frame_count": float,
    "clamp_reference_similarity": float | None,
    "evidence_consistent": bool | None,
}
```

Write `originals/`, `masks/hand/`, `masks/clamp/`, `rois/`, `normalized/`, `comparisons/`, `overlays/`, and `frame_analysis.json`. Green outlines mark the glove, cyan outlines the candidate clamp, and the comparison panel shows candidate/reference/overlap.

- [ ] **Step 4: Run extractor and feature tests**

Run: `python -m pytest tests/test_cp04_extractor.py tests/features/test_cp04_clamp_features.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the extractor**

```bash
git add src/medical_evaluation/extractors/cp04.py tests/test_cp04_extractor.py
git commit -m "feat: extract displayed CP04 clamp evidence"
```

### Task 4: Runtime, composite extractor, CLI, and report integration

**Files:**
- Modify: `src/medical_evaluation/settings.py`
- Modify: `src/medical_evaluation/runtime.py`
- Modify: `src/medical_evaluation/extractors/cp09_cp11.py`
- Modify: `src/medical_evaluation/pipeline.py`
- Modify: `scripts/run_real_cp09_cp11.py`
- Modify: `tests/test_runtime.py`
- Modify: `tests/test_cp09_cp11_extractor.py`
- Modify: `tests/test_pipeline_vertical.py`
- Modify: `tests/scripts/test_run_real_cp09_cp11.py`

- [ ] **Step 1: Write failing integration tests**

Assert that `Settings.cp04_reference_dir` resolves under the project root; runtime constructs CP04 with the audited shared segmenter; the composite dispatches `cp_04`; enabled checkpoints become CP02, CP03, CP04, CP08, CP09, CP10, CP11; the audit model string reflects seven enabled checkpoints; and the real runner accepts `--only cp_04` and prints it.

- [ ] **Step 2: Run integration tests and confirm failure**

Run: `python -m pytest tests/test_runtime.py tests/test_cp09_cp11_extractor.py tests/test_pipeline_vertical.py tests/scripts/test_run_real_cp09_cp11.py -q`

Expected: FAIL on the missing CP04 runtime path.

- [ ] **Step 3: Wire CP04 through production**

Add `cp04_reference_dir = Path("config/cp04_reference.v1")`, instantiate `Cp04FeatureExtractor`, add optional `cp04` delegation and model-version reporting to the composite extractor, enable CP04, update the audit string to `vertical-cp02-cp03-cp04-cp08-cp09-cp10-cp11`, and include CP04 in the real runner choices/output.

- [ ] **Step 4: Run integration tests**

Run the same command as Step 2.

Expected: PASS.

- [ ] **Step 5: Commit integration**

```bash
git add src/medical_evaluation/settings.py src/medical_evaluation/runtime.py src/medical_evaluation/extractors/cp09_cp11.py src/medical_evaluation/pipeline.py scripts/run_real_cp09_cp11.py tests/test_runtime.py tests/test_cp09_cp11_extractor.py tests/test_pipeline_vertical.py tests/scripts/test_run_real_cp09_cp11.py
git commit -m "feat: integrate CP04 evaluation pipeline"
```

### Task 5: Automatic success-reference builder and real-video validation

**Files:**
- Create: `scripts/build_cp04_reference.py`
- Create: `tests/scripts/test_build_cp04_reference.py`
- Generate: `config/cp04_reference.v1/manifest.json`
- Generate: `config/cp04_reference.v1/masks/*.png`
- Modify after final results: `D:/Medical_evaluation/AGENTS.md` (local untracked handoff file; do not commit it)

- [ ] **Step 1: Test the reference-builder contract**

The builder must accept only the configured `success` video and its annotated CP04 range, reuse `Cp04FeatureExtractor` candidate scanning/ranking, require at least two reliable consecutive displayed-clamp masks, write masks plus a manifest containing video id, frame index/time, prompt, source model version, and reference version, and refuse to overwrite an existing different reference without an explicit `--replace` flag.

- [ ] **Step 2: Run the builder tests and confirm failure**

Run: `python -m pytest tests/scripts/test_build_cp04_reference.py -q`

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement and locally test the builder**

Run: `python -m pytest tests/scripts/test_build_cp04_reference.py tests/test_cp04_extractor.py -q`

Expected: PASS.

- [ ] **Step 4: Run all local verification before GPU work**

Run:

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
& D:\Medical_evaluation\.venv\Scripts\ruff.exe check --no-cache src tests scripts
```

Expected: all tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 5: Commit, push, and update `.102` safely**

Commit only CP04 source, tests, script, config reference assets, rubric, and documentation. Push `codex/sam3-cp09-cp11-text`; before and after `.102` fast-forward, verify the protected dirty `src/medical_evaluation/rubric.py` remains modified with SHA-256 `14b3b33b010825f3afe605d9d77d12827f79b94c0833c6e3947c8d716903be77`.

- [ ] **Step 6: Build the reference on fixed GPU5 and inspect masks**

Run through `/home/tangm/.local/bin/gpu5-run env CUDA_VISIBLE_DEVICES=0` with the `sam3_medical` Python. Inspect every generated glove/clamp overlay and reference mask; the selected object must be the small clamp on the displayed white palm, not the plier or rack. If the primary prompt has no reliable candidate, test only the two approved alternate prompts before changing the design.

- [ ] **Step 7: Run all three CP04 reports**

Run `scripts/run_real_cp09_cp11.py --only cp_04 --no-commentary` for `success`, `failure`, and `clamp_failure`. Record status, reason code, clear/matching frame counts, similarity, evidence paths, runtime, and any ambiguous prompt output. Do not infer the expected result from the filename.

- [ ] **Step 8: Final verification and handoff update**

After reference assets are committed and deployed, rerun local full pytest/Ruff and server focused CP04 tests. Update `AGENTS.md` with exact commits, three-end HEADs, reports, evidence paths, limitations, and the next user-selected checkpoint. Preserve every CP01 and CP11 file.
