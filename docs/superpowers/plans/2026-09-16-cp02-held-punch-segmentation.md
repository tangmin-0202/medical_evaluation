# CP02 Held Punch Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically generate a SAM3 seed box for the whole punch pliers held by the operator, reject masks for the green sheet or static distractor pliers, and retain auditable real-video evidence.

**Architecture:** Extend the existing moving multi-hole disk locator with a focused held-tool box estimator. Keep SAM3 as the primary segmenter, but accept a track only after deterministic anchor, box, color, shape, and temporal checks. Limit this iteration to the feasibility gate; do not connect CP02 scoring until the held-tool masks pass visual review.

**Tech Stack:** Python 3.11/3.12, OpenCV, NumPy, SAM3.1, pytest, Ruff, SSH GPU smoke tests.

---

## File map

- Modify `src/medical_evaluation/features/cp02_punch.py`: held-tool box estimation and per-mask semantic measurements.
- Modify `scripts/run_early_cp_sam3_gate.py`: use the whole-tool box, reject semantic swaps, and write audit fields.
- Modify `tests/features/test_cp02_punch_features.py`: synthetic geometry tests for held-tool box and mask checks.
- Modify `tests/scripts/test_early_cp_gate.py`: gate integration and rejection-reason tests.

### Task 1: Estimate a whole held-punch box from the automatic disk anchor

**Files:**
- Modify: `src/medical_evaluation/features/cp02_punch.py`
- Test: `tests/features/test_cp02_punch_features.py`

- [ ] **Step 1: Write failing synthetic tests**

Add a moving silver elongated punch connected to the moving disk, a large green rectangle, and a static silver distractor. Assert that the returned normalized box contains the disk and moving punch but excludes the static distractor center. Also assert that a disk without a connected moving elongated candidate returns `None` instead of inventing a box.

```python
def test_held_punch_box_follows_moving_tool_not_static_distractor():
    frames = _held_punch_frames()
    disk = punch.locate_moving_multihole_disk(frames)
    box = punch.locate_held_punch_box(frames, disk)
    assert box is not None
    assert box.contains(disk.x, disk.y)
    assert box.contains(260, 285)
    assert not box.contains(540, 90)


def test_held_punch_box_requires_connected_elongated_motion():
    frames = _disk_only_frames()
    disk = punch.locate_moving_multihole_disk(frames)
    assert punch.locate_held_punch_box(frames, disk) is None
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest tests/features/test_cp02_punch_features.py -q
```

Expected: the two new tests fail because `locate_held_punch_box` is not defined.

- [ ] **Step 3: Implement the focused estimator**

Add an immutable box result and one public estimator. The estimator must derive motion from the frame median, suppress saturated green pixels, join nearby moving low-saturation/high-value regions to the disk anchor, require an elongated union, and clip the result to the image.

```python
@dataclass(frozen=True)
class HeldPunchBox:
    frame_position: int
    x1: int
    y1: int
    x2: int
    y2: int
    elongation: float
    motion_ratio: float

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def normalized(self, width: int, height: int) -> list[float]:
        return [self.x1 / width, self.y1 / height, self.x2 / width, self.y2 / height]


def locate_held_punch_box(
    frames_bgr: Sequence[np.ndarray], disk: MovingDisk | None,
) -> HeldPunchBox | None:
    """Return an automatic whole-tool box anchored to a moving multi-hole disk."""
```

Use the disk radius to bound all morphology and padding so the algorithm remains resolution-relative. Do not add fixed screen coordinates or video-specific frame numbers.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests in that file pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/medical_evaluation/features/cp02_punch.py tests/features/test_cp02_punch_features.py
git commit -m "feat: locate whole held punch from disk motion"
```

### Task 2: Reject SAM3 semantic swaps deterministically

**Files:**
- Modify: `src/medical_evaluation/features/cp02_punch.py`
- Test: `tests/features/test_cp02_punch_features.py`

- [ ] **Step 1: Write failing mask-validation tests**

Test one acceptable elongated silver mask containing the disk anchor and three rejected cases: a large green sheet, a mask outside the automatic box, and a compact blob that does not contain or approach the disk.

```python
def test_held_punch_mask_rejects_green_sheet():
    result = punch.measure_held_punch_mask(frame, green_mask, disk, box)
    assert result.accepted is False
    assert result.reason == "green_sheet_mask"


def test_held_punch_mask_accepts_anchor_connected_elongated_metal():
    result = punch.measure_held_punch_mask(frame, tool_mask, disk, box)
    assert result.accepted is True
    assert result.reason == "criteria_satisfied"
```

- [ ] **Step 2: Run the new tests and verify RED**

Run the Task 1 pytest command. Expected: failures because the measurement API is missing.

- [ ] **Step 3: Implement auditable measurements**

Add an immutable result containing all reported metrics and a single reason code.

```python
@dataclass(frozen=True)
class HeldPunchMaskMeasurement:
    accepted: bool
    reason: str
    anchor_distance_radii: float
    inside_box_ratio: float
    green_ratio: float
    elongation: float


def measure_held_punch_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    disk: MovingDisk,
    box: HeldPunchBox,
) -> HeldPunchMaskMeasurement:
    """Measure and classify one SAM3 mask without changing the mask."""
```

Use explicit reason precedence: `empty_mask`, `anchor_missed`, `outside_tool_box`, `green_sheet_mask`, `not_elongated`, then `criteria_satisfied`. Retain the raw mask even when rejected.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/medical_evaluation/features/cp02_punch.py tests/features/test_cp02_punch_features.py
git commit -m "feat: validate held punch SAM3 masks"
```

### Task 3: Use the whole-tool box and semantic gate in the CP02 runner

**Files:**
- Modify: `scripts/run_early_cp_sam3_gate.py`
- Test: `tests/scripts/test_early_cp_gate.py`

- [ ] **Step 1: Write failing runner tests**

Update the automatic-box fixture so its synthetic punch is elongated and connected to the disk. Assert that the emitted box is the `HeldPunchBox`, every frame includes `semantic_features`, and a non-empty green mask is retained in artifacts but counted as rejected with `green_sheet_mask`.

```python
automatic = result["objects"]["rubber_dam_punch"][-1]
assert automatic["prompt"] == "automatic held punch box"
assert automatic["accepted_frame_count"] == 1
assert automatic["frames"][0]["semantic_features"]["reason"] == "criteria_satisfied"

rejected = gate._run_automatic_punch_box(
    backend=GreenMaskBackend(),
    video_path=Path("unused.mp4"),
    time_range=TimeRange(start_sec=0, end_sec=4),
    output_dir=tmp_path / "green-rejection",
    sample_fps=1,
    read_frame_fn=lambda _path, index: frames[index].image_bgr,
    sample_frames_fn=lambda *args, **kwargs: iter(frames),
)
assert rejected["accepted_frame_count"] == 0
assert rejected["frames"][0]["semantic_features"]["reason"] == "green_sheet_mask"
```

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest tests/scripts/test_early_cp_gate.py -q
```

Expected: failures because the runner still uses radius padding and area-only acceptance.

- [ ] **Step 3: Replace padding attempts with the held-tool box**

Import `locate_held_punch_box` and `measure_held_punch_mask`. Keep the disk locator artifact, draw the final whole-tool box in magenta, run one SAM3 box track, attach the measurement dictionary to every frame, and compute consecutive validity from `accepted` rather than mask area. Return `tool_box_not_found` when the disk exists but the whole-tool estimator fails.

The result must include:

```python
{
    "prompt": "automatic held punch box",
    "prompt_kind": "box",
    "locator": {
        "frame_index": seed.frame_index,
        "time_sec": seed.time_sec,
        "x": locator.x,
        "y": locator.y,
        "radius": locator.radius,
    },
    "tool_box": {
        "x1": tool_box.x1,
        "y1": tool_box.y1,
        "x2": tool_box.x2,
        "y2": tool_box.y2,
        "elongation": tool_box.elongation,
        "motion_ratio": tool_box.motion_ratio,
    },
    "accepted_frame_count": accepted_count,
    "max_consecutive_accepted_frames": consecutive_count,
    "automatic_gate_passed": consecutive_count >= 2,
    "frames": frames,
}
```

- [ ] **Step 4: Run CP02 focused regressions**

Run:

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest tests/features/test_cp02_punch_features.py tests/scripts/test_early_cp_gate.py tests/test_sam3_backend.py -q
& D:\Medical_evaluation\.venv\Scripts\ruff.exe check --no-cache src/medical_evaluation/features/cp02_punch.py scripts/run_early_cp_sam3_gate.py tests/features/test_cp02_punch_features.py tests/scripts/test_early_cp_gate.py
```

Expected: all selected tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 5: Commit Task 3**

```powershell
git add scripts/run_early_cp_sam3_gate.py tests/scripts/test_early_cp_gate.py
git commit -m "fix: segment the held CP02 punch instance"
```

### Task 4: Server and real-video validation

**Files:**
- No source changes unless the evidence exposes a reproducible defect.
- Create runtime evidence only under server `data/runs/`; do not commit it.

- [ ] **Step 1: Push and fast-forward the server safely**

Before updating, verify the server branch, HEAD, dirty `src/medical_evaluation/rubric.py`, and its protected SHA-256. Push the feature branch, then use `git merge --ff-only` on the server. Recheck the protected file after updating.

- [ ] **Step 2: Check GPUs dynamically**

Run `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader`. Select a currently free card; do not reuse a historical GPU number and do not stop another user's process.

- [ ] **Step 3: Run success and failure CP02 stages**

Run the gate with `PYTHONPATH="$PWD/src"`, the `sam3_medical` interpreter, `CUDA_VISIBLE_DEVICES` set to the selected card, 2 FPS, and separate new output directories for `success` and `failure`. Preserve stdout/stderr and `summary.json`.

- [ ] **Step 4: Inspect evidence instead of trusting metrics**

Download representative locator overlays and at least three accepted/rejected SAM3 overlays from each run. Confirm visually that accepted masks cover the held punch, include the hole disk, and exclude the static pliers and green sheet. If either video fails, record the exact reason and return to the failing-test step rather than loosening thresholds blindly.

- [ ] **Step 5: Report the bounded result**

Report test commands and counts, GPU used, per-video accepted-frame counts, reason codes, evidence paths, and remaining limits. Do not call CP02 complete until both real-video runs and visual inspection pass.
