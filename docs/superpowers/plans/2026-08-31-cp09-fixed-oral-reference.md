# CP09 Fixed Oral Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the SAM2-tracked rubber-dam frame against one fixed, manually annotated oral reference box instead of attempting to segment the occluded oral region.

**Architecture:** Keep `oral_region` as the existing normalized `BoxPrompt`, validate exactly one reference box before inference, and send only `rubber_dam_frame` prompts to SAM2. Convert each valid frame-mask bounding-box center to normalized coordinates, compute its offset relative to the oral box center and scale, then take the stage median. Evidence overlays show the frame mask, oral box, both centers, and their connecting line.

**Tech Stack:** Python 3.11/3.12, Pydantic, NumPy, OpenCV, pytest, Ruff, SAM2 adapter.

---

### Task 1: Establish a compatible local test environment

**Files:**
- No tracked files
- Create ignored environment: `.venv/`

- [ ] **Step 1: Create the Python 3.12 environment**

Run:

```powershell
& "C:\Users\tang'min\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv
```

Expected: `.venv\Scripts\python.exe` exists and reports Python 3.12.

- [ ] **Step 2: Install the project and development dependencies**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Expected: editable project, pytest, Ruff, FastAPI, OpenCV, NumPy and PyYAML install successfully.

- [ ] **Step 3: Run the clean baseline**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

Expected: 119 tests pass with the known Starlette deprecation warning only.

### Task 2: Add fixed-box geometry and visual evidence

**Files:**
- Modify: `tests/test_features.py`
- Modify: `src/medical_evaluation/features/geometry.py`

- [ ] **Step 1: Write failing geometry tests**

Add imports and tests that express the new API:

```python
from medical_evaluation.features.geometry import (
    relative_bbox_center_offset_to_box,
    write_reference_overlay,
)

def test_relative_bbox_center_offset_to_box_uses_normalized_reference_scale() -> None:
    frame = np.zeros((100, 200), bool)
    frame[40:60, 90:110] = True
    assert relative_bbox_center_offset_to_box(
        frame, (0.25, 0.2, 0.75, 0.8)
    ) == pytest.approx(0.0)

def test_relative_bbox_center_offset_to_box_rejects_empty_mask_and_bad_box() -> None:
    empty = np.zeros((10, 10), bool)
    assert relative_bbox_center_offset_to_box(empty, (0.2, 0.2, 0.8, 0.8)) is None
    with pytest.raises(ValueError, match="reference box"):
        relative_bbox_center_offset_to_box(np.ones((10, 10), bool), (0.8, 0.2, 0.2, 0.8))

def test_reference_overlay_draws_mask_box_centers_and_line(tmp_path: Path) -> None:
    frame = np.zeros((100, 200, 3), np.uint8)
    mask = np.zeros((100, 200), bool)
    mask[40:60, 90:110] = True
    output = write_reference_overlay(
        frame, mask, (0.25, 0.2, 0.75, 0.8), tmp_path, "cp_09/overlay.jpg"
    )
    assert output.is_file()
    assert cv2.imread(str(output)).sum() > 0
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_features.py -q
```

Expected: collection fails because the two new functions do not exist.

- [ ] **Step 3: Implement the minimal geometry helpers**

Add:

```python
def relative_bbox_center_offset_to_box(
    subject: np.ndarray,
    reference_box: tuple[float, float, float, float],
) -> float | None:
    binary = _mask(subject)
    subject_box = _bounding_box(binary)
    if subject_box is None:
        return None
    x1, y1, x2, y2 = _normalized_box(reference_box)
    height, width = binary.shape
    sx = ((subject_box[0] + subject_box[2]) / 2) / max(width - 1, 1)
    sy = ((subject_box[1] + subject_box[3]) / 2) / max(height - 1, 1)
    rx, ry = (x1 + x2) / 2, (y1 + y2) / 2
    return float(np.hypot((sx - rx) / (x2 - x1), (sy - ry) / (y2 - y1)))
```

Implement `_normalized_box` with finite, ordered, `0..1` validation. Implement
`write_reference_overlay` by blending the stable frame-mask color, converting normalized box
corners to pixels, and drawing the reference rectangle, both centers, labels, and a line before
writing beneath `evidence_root`.

- [ ] **Step 4: Run the tests and verify GREEN**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_features.py -q
```

Expected: all feature tests pass.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_features.py src/medical_evaluation/features/geometry.py
git commit -m "feat: measure frame offset from fixed oral box"
```

### Task 3: Refactor CP09 to track only the frame

**Files:**
- Modify: `tests/test_cp09_extractor.py`
- Modify: `src/medical_evaluation/extractors/cp09.py`

- [ ] **Step 1: Replace paired-mask tests with fixed-reference tests**

Tests must assert:

```python
assert {prompt.object_id for prompt in segmenter.prompts} == {"rubber_dam_frame"}
assert segmenter.time_range.start_sec == pytest.approx(0.5)
assert result.features == {
    "frame_oral_center_offset": pytest.approx(0.0),
    "frame_valid_count": 3.0,
    "oral_reference_count": 1.0,
    "relative_offset_valid_count": 3.0,
}
assert not (evidence_root / "cp_09/masks/oral_region").exists()
assert all(
    item.rule == "rubber_dam_frame_relative_to_oral_reference"
    for item in result.evidence
)
```

Add separate tests proving missing, point-only and duplicate `oral_region` annotations fail before
`segmenter.track`, and two valid frame masks produce a `None` offset but still write two
evidence overlays.

- [ ] **Step 2: Run the extractor tests and verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_cp09_extractor.py -q
```

Expected: failures show oral prompts are still sent to SAM2 and old paired-mask fields remain.

- [ ] **Step 3: Implement reference-box extraction and single-object tracking**

In `Cp09FeatureExtractor.extract`:

```python
frame_prompts = prompts_for_object(...)
oral_reference = self._oral_reference_box(time_range, checkpoint_id)
tracking_range = TimeRange(
    start_sec=min(time_range.start_sec, *(p.frame_time_sec for p in frame_prompts)),
    end_sec=max(time_range.end_sec, *(p.frame_time_sec for p in frame_prompts)),
)
tracked = self.segmenter.track(video_path, tracking_range, frame_prompts, sample_fps=dense_fps)
```

`_oral_reference_box` filters annotations by object ID and the existing `0.5` second tolerance,
requires one `BoxPrompt` and no competing oral prompt, then returns
`(x1, y1, x2, y2)`. For every stage frame with a non-empty frame mask, call
`relative_bbox_center_offset_to_box`. Select first/middle/last valid entries, save only the
frame mask, and call `write_reference_overlay`.

Return exactly:

```python
{
    "frame_oral_center_offset": median_or_none,
    "frame_valid_count": float(frame_valid_count),
    "oral_reference_count": 1.0,
    "relative_offset_valid_count": float(len(valid_offsets)),
}
```

- [ ] **Step 4: Run extractor and geometry tests and verify GREEN**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_cp09_extractor.py tests/test_features.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_cp09_extractor.py src/medical_evaluation/extractors/cp09.py
git commit -m "feat: use fixed oral reference box for CP09"
```

### Task 4: Update the smoke contract, decision wording and runbook

**Files:**
- Modify: `tests/test_smoke_sam2_cp09.py`
- Modify: `tests/test_judges_cp09_cp11.py`
- Modify: `scripts/smoke_sam2_cp09.py`
- Modify: `src/medical_evaluation/judges/cp09_cp11.py`
- Modify: `docs/server-runbook.md`

- [ ] **Step 1: Write failing contract tests**

Update fake extractor features and assertions:

```python
features={
    "frame_oral_center_offset": 0.03,
    "frame_valid_count": 3.0,
    "oral_reference_count": 1.0,
    "relative_offset_valid_count": 3.0,
}
assert summary["oral_reference_count"] == 1.0
assert summary["relative_offset_valid_count"] == 3.0
assert "oral_region_valid_count" not in summary
assert "paired_valid_count" not in summary
```

Update the no-evidence Judge test to use `relative_offset_valid_count` and assert its reason text
mentions the oral reference area.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_smoke_sam2_cp09.py tests/test_judges_cp09_cp11.py -q
```

Expected: smoke summary still exposes paired-mask counts and Judge wording still says oral-region
recognition.

- [ ] **Step 3: Implement the smoke and wording changes**

Keep `prompt_counts` for both annotation object IDs, but write:

```python
"oral_reference_count": result.features.get("oral_reference_count", 0.0),
"relative_offset_valid_count": result.features.get("relative_offset_valid_count", 0.0),
```

Remove the two old summary fields. Change CP09 missing-evidence, correct and incorrect text to refer
to the fixed “口腔参考区域”. Update runbook section 8 so it documents one SAM2 object, one fixed
reference box, one mask directory, the new counters and the calibration requirement.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_smoke_sam2_cp09.py tests/test_judges_cp09_cp11.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_smoke_sam2_cp09.py tests/test_judges_cp09_cp11.py scripts/smoke_sam2_cp09.py src/medical_evaluation/judges/cp09_cp11.py docs/server-runbook.md
git commit -m "docs: update CP09 fixed-reference smoke contract"
```

### Task 5: Full verification and server handoff

**Files:**
- Verify all changed files
- Create ignored transfer bundle at repository root

- [ ] **Step 1: Run full local verification**

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m ruff check .
git diff --check
git status --short
```

Expected: all tests pass, Ruff and diff checks are clean, and only known user data plus any intended
transfer bundle remain untracked.

- [ ] **Step 2: Merge the feature branch into local main**

After reading the finishing-development-branch skill, fast-forward local `main` to the verified
feature commit without touching untracked files.

- [ ] **Step 3: Push GitHub main**

```powershell
git push origin main
```

Expected: GitHub main points at the verified commit. If network access fails, create a Git bundle
for server transfer and report GitHub as pending rather than claiming success.

- [ ] **Step 4: Create and verify the server bundle**

```powershell
git bundle create medical-evaluation-main-<shortsha>.bundle main
git bundle verify medical-evaluation-main-<shortsha>.bundle
```

Expected: bundle verifies and contains the new main commit.

- [ ] **Step 5: Server verification**

Transfer the bundle, fast-forward `~/medical_evaluation`, run the server Python 3.11 full test
suite, then run the real GPU CP09 smoke on an idle card. Accept the result only after inspecting
three overlays and confirming that no `oral_region` mask directory is created.

