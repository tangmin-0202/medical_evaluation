# CP08 SAM3 Feasibility Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and run an auditable SAM3 feasibility gate that determines whether the current model can localize the CP08 blunt instrument and the CP09-tail target tooth, complete rubber-dam clamp, and green rubber dam before any CP08 scoring logic is implemented.

**Architecture:** Add a standalone experiment script, separate from the production pipeline. It reads the existing stage annotations, uses a sparse whole-CP08 search followed by a short dense confirmation window for instrument prompts, and uses independent SAM3 sessions for each CP09-tail object. It saves masks, overlays, per-frame metrics, prompt rankings, and a machine-readable summary. Automatic checks cover detection continuity and mask coherence; the required visual review checks whether masks actually cover the intended object and preserve the clamp wings. Sharp-probe output is an exclusion audit, not a scoring signal at this stage.

**Tech Stack:** Python 3.11/3.12, pytest, Ruff, OpenCV, NumPy, existing `Sam3Backend`, existing annotation JSON, Git, SSH, NVIDIA GPU.

---

## Scope and non-goals

- This plan does not add CP08 to `runtime.py`, `pipeline.py`, scoring, reports, or Qwen.
- This plan does not change CP09, CP10, or CP11 behavior.
- This plan does not define blunt-tip geometry thresholds, clamp-wing completeness thresholds, or wing-hole color thresholds.
- "Blunt" applies only to the working end that contacts the dam near the tooth/clamp. The thick handle and curved shaft are not bluntness evidence. A visible rounded/smooth terminal contour that retains width is positive evidence; a stable taper to a single acute/needle-like apex is sharp evidence; an occluded terminal is unresolved.
- A gate result is not a clinical accuracy claim. It only establishes whether the present SAM3 model and prompts provide usable masks on the available Demo videos.
- All four positive objects must be run in independent SAM3 sessions: blunt instrument, target tooth, clamp, and rubber dam. Sharp-probe prompts also use independent sessions.
- The CP08 positive search may stop after a clear dense confirmation window. A negative CP08 result must scan the complete annotated stage.

## Gate contract

The script will write:

```text
data/runs/<run-id>/
├── summary.json
├── selected_prompts.json
├── manual_review.json
├── blunt_instrument/<prompt-slug>/{raw,masks,overlays}/...
├── sharp_probe/<prompt-slug>/{raw,masks,overlays}/...
├── target_tooth/<prompt-slug>/{raw,masks,overlays}/...
├── rubber_dam_clamp/<prompt-slug>/{raw,masks,overlays}/...
└── rubber_dam/<prompt-slug>/{raw,masks,overlays}/...
```

`summary.json` must contain the video ID/path, annotation path, CP08 and CP09-tail windows, model/checkpoint identity, Git HEAD, SAM revision, selected GPU, peak memory, prompt-by-prompt frame metrics, automatic gate results, and any exception. `manual_review.json` must contain one unresolved checklist entry per positive object so a numeric continuity result is never mistaken for semantic correctness.

Automatic acceptance for a positive object requires at least three consecutive valid frames and a median dominant-component ratio of at least `0.60`. These are experiment-quality gates only, not final CP08 judging thresholds. A human must additionally confirm:

- blunt instrument: mask follows the same instrument and includes a visible working tip;
- target tooth: mask follows the isolated tooth inside the clamp;
- rubber-dam clamp: this is a separate full-clamp mask, not inferred from the tooth mask; it covers the clamp body and both wing regions rather than only highlights;
- rubber dam: mask covers the green sheet surrounding the tooth/clamp;
- sharp-probe audit: inspect any masks and record whether they are a true sharp instrument or a false positive.

## Task 1: Add failing tests for the gate definition and stage windows

**Files:**

- Create: `tests/scripts/test_run_cp08_sam3_gate.py`
- Create: `scripts/run_cp08_sam3_gate.py`
- Read: `scripts/run_sam3_prompt_matrix.py`
- Read: `src/medical_evaluation/annotations.py`

- [ ] Add the following initial tests. Importing the script by file path keeps it a standalone experiment entry point.

```python
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "run_cp08_sam3_gate.py"
SPEC = importlib.util.spec_from_file_location("run_cp08_sam3_gate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def test_prompt_catalog_has_independent_cp08_and_cp09_objects() -> None:
    assert set(gate.PROMPT_CANDIDATES) == {
        "blunt_instrument",
        "sharp_probe",
        "target_tooth",
        "rubber_dam_clamp",
        "rubber_dam",
    }
    assert len(gate.PROMPT_CANDIDATES["blunt_instrument"]) >= 2
    assert len(gate.PROMPT_CANDIDATES["sharp_probe"]) >= 2
    assert "green dental rubber dam" in gate.PROMPT_CANDIDATES["rubber_dam"]


def test_load_windows_uses_all_of_cp08_and_last_three_seconds_of_cp09(
    tmp_path: Path,
) -> None:
    annotation = tmp_path / "success.json"
    annotation.write_text(
        json.dumps(
            {
                "video_id": "success",
                "steps": [
                    {
                        "checkpoint_id": "cp_08",
                        "time_range": {"start_sec": 134.0, "end_sec": 174.0},
                        "label": "correct",
                        "reason": "fixture",
                    },
                    {
                        "checkpoint_id": "cp_09",
                        "time_range": {"start_sec": 175.0, "end_sec": 195.0},
                        "label": "correct",
                        "reason": "fixture",
                    },
                ],
                "prompts": [],
                "audit_history": [],
            }
        ),
        encoding="utf-8",
    )

    windows = gate.load_gate_windows(annotation)

    assert windows.cp08 == pytest.approx((134.0, 174.0))
    assert windows.cp09_tail == pytest.approx((192.0, 195.0))
```

- [ ] Run the focused tests and verify they fail because the script does not exist yet.

```powershell
cd D:\Medical_evaluation\.worktrees\sam3-cp09-cp11-text
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest tests\scripts\test_run_cp08_sam3_gate.py -q
```

Expected: collection/import failure for `scripts/run_cp08_sam3_gate.py`.

- [ ] Create `scripts/run_cp08_sam3_gate.py` with the prompt catalog and strict annotation parsing.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medical_evaluation.annotations import VideoAnnotations


PROMPT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "blunt_instrument": (
        "blunt dental instrument",
        "dental instrument with a rounded blunt working tip",
        "metal dental instrument with a curved shaft and rounded working tip",
    ),
    "sharp_probe": (
        "sharp dental explorer probe",
        "pointed dental probe with a needle-like tip",
    ),
    "target_tooth": (
        "isolated white tooth inside the rubber dam clamp",
        "white molar enclosed by the metal rubber dam clamp",
    ),
    "rubber_dam_clamp": (
        "metal rubber dam clamp around the tooth",
        "complete stainless steel rubber dam clamp with two side wings",
    ),
    "rubber_dam": ("green dental rubber dam",),
}


@dataclass(frozen=True)
class GateWindows:
    cp08: tuple[float, float]
    cp09_tail: tuple[float, float]


def load_gate_windows(annotation_path: Path) -> GateWindows:
    annotations = VideoAnnotations.model_validate_json(annotation_path.read_text(encoding="utf-8"))
    ranges = {step.checkpoint_id: step.time_range for step in annotations.steps}
    for checkpoint_id in ("cp_08", "cp_09"):
        if checkpoint_id not in ranges:
            raise ValueError(f"Missing {checkpoint_id} annotation range")
    cp08 = ranges["cp_08"]
    cp09 = ranges["cp_09"]
    return GateWindows(
        cp08=(cp08.start_sec, cp08.end_sec),
        cp09_tail=(max(cp09.start_sec, cp09.end_sec - 3.0), cp09.end_sec),
    )
```

- [ ] Re-run the focused tests and verify both pass.

- [ ] Commit the first red-green slice.

```powershell
git add scripts/run_cp08_sam3_gate.py tests/scripts/test_run_cp08_sam3_gate.py
git commit -m "test: define CP08 SAM3 feasibility gate"
```

## Task 2: Add deterministic mask-quality metrics and prompt selection

**Files:**

- Modify: `scripts/run_cp08_sam3_gate.py`
- Modify: `tests/scripts/test_run_cp08_sam3_gate.py`

- [ ] Add tests for continuity, component coherence, and deterministic ranking.

```python
import numpy as np


def test_summarize_masks_requires_three_consecutive_valid_frames() -> None:
    masks = [
        np.zeros((8, 8), dtype=bool),
        np.pad(np.ones((3, 3), dtype=bool), 2),
        np.pad(np.ones((3, 3), dtype=bool), 2),
        np.pad(np.ones((3, 3), dtype=bool), 2),
    ]

    result = gate.summarize_masks(masks, min_area_px=4)

    assert result["valid_frame_count"] == 3
    assert result["max_consecutive_valid_frames"] == 3
    assert result["automatic_gate_passed"] is True


def test_summarize_masks_rejects_fragmented_mask() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[7:9, 7:9] = True

    result = gate.summarize_masks([mask, mask, mask], min_area_px=2)

    assert result["median_dominant_component_ratio"] == pytest.approx(0.5)
    assert result["automatic_gate_passed"] is False


def test_select_prompt_prefers_pass_then_continuity_then_coherence() -> None:
    candidates = [
        {"prompt": "a", "automatic_gate_passed": False, "max_consecutive_valid_frames": 9,
         "median_dominant_component_ratio": 0.9},
        {"prompt": "b", "automatic_gate_passed": True, "max_consecutive_valid_frames": 3,
         "median_dominant_component_ratio": 0.7},
        {"prompt": "c", "automatic_gate_passed": True, "max_consecutive_valid_frames": 4,
         "median_dominant_component_ratio": 0.6},
    ]

    assert gate.select_prompt(candidates)["prompt"] == "c"
```

- [ ] Run the focused test file and confirm the new tests fail.

- [ ] Implement `summarize_masks()` using `cv2.connectedComponentsWithStats`, count consecutive non-empty masks, and compute the dominant connected-component area divided by total foreground area for each valid frame. Set `automatic_gate_passed` only when continuity is at least 3 and median coherence is at least 0.60.

```python
import cv2
import numpy as np
from typing import Any


def _max_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _dominant_component_ratio(mask: np.ndarray) -> float:
    binary = np.asarray(mask, dtype=np.uint8)
    foreground = int(binary.sum())
    if foreground == 0:
        return 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    return largest / foreground


def summarize_masks(masks: list[np.ndarray], *, min_area_px: int) -> dict[str, Any]:
    areas = [int(np.asarray(mask, dtype=bool).sum()) for mask in masks]
    valid = [area >= min_area_px for area in areas]
    coherence = [
        _dominant_component_ratio(mask)
        for mask, is_valid in zip(masks, valid, strict=True)
        if is_valid
    ]
    median_coherence = float(np.median(coherence)) if coherence else 0.0
    longest = _max_true_run(valid)
    return {
        "frame_count": len(masks),
        "valid_frame_count": sum(valid),
        "max_consecutive_valid_frames": longest,
        "median_dominant_component_ratio": median_coherence,
        "automatic_gate_passed": longest >= 3 and median_coherence >= 0.60,
    }


def select_prompt(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one prompt result is required")
    return max(
        candidates,
        key=lambda item: (
            bool(item["automatic_gate_passed"]),
            int(item["max_consecutive_valid_frames"]),
            float(item["median_dominant_component_ratio"]),
        ),
    )
```

- [ ] Run the focused tests and verify they pass.

- [ ] Commit the second red-green slice.

```powershell
git add scripts/run_cp08_sam3_gate.py tests/scripts/test_run_cp08_sam3_gate.py
git commit -m "feat: score SAM3 feasibility masks"
```

## Task 3: Implement sparse CP08 search, dense confirmation, and independent CP09-tail sessions

**Files:**

- Modify: `scripts/run_cp08_sam3_gate.py`
- Modify: `tests/scripts/test_run_cp08_sam3_gate.py`
- Read: `src/medical_evaluation/segmentation/sam3_backend.py`
- Read: `src/medical_evaluation/video.py`

- [ ] Add a fake segmenter test that records every call. The test must prove:

  1. all CP08 prompts receive the entire CP08 range at `1.0` FPS;
  2. only a CP08 prompt with a sparse candidate receives a second, clipped three-second window at `5.0` FPS;
  3. tooth, clamp, and dam each receive the same CP09-tail range at `5.0` FPS;
  4. every call is independent, represented by a separate `track()` invocation;
  5. a prompt with no CP08 candidate is never declared absent until its full sparse scan is complete.

Use small synthetic `FrameMasks` values with known frame indices and masks. Do not mock the metric functions.

- [ ] Run the focused tests and verify failure before implementation.

- [ ] Add these immutable run settings and injectable orchestration entry point:

```python
CP08_SPARSE_FPS = 1.0
CP08_DENSE_FPS = 5.0
CP08_DENSE_HALF_WINDOW_SEC = 1.5
CP09_TAIL_FPS = 5.0
MIN_MASK_AREA_PX = 64


def run_gate(
    *,
    segmenter: Any,
    video_path: Path,
    annotation_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run all prompt sessions, save artifacts, and return the JSON-safe summary."""
```

The implementation must:

- refuse a non-empty output directory;
- call `track(video_path, TimeRange(...), [SegmentationPrompt(...)], sample_fps=...)` once per session, with one text prompt and a stable object ID;
- sparse-scan the full CP08 stage for every blunt and sharp prompt;
- also run a point-prompt diagnostic at the existing success annotation time `139.814028s`, using its three `blunt‑tipped instrument` points; record it separately as `manual_point_diagnostic` and never allow it to satisfy the automatic text-localization gate;
- derive the first candidate timestamp from returned `FrameMasks` metadata;
- run one dense window only for prompts with a valid sparse candidate;
- evaluate tooth, clamp, and dam over the last three seconds of CP09;
- save original frames, one-channel masks, and overlays with object name, prompt, timestamp, area, and contour;
- keep tooth and full-clamp masks separate; the later implementation may use the tooth center plus the clamp's own principal axis to divide the full clamp mask into two wing ROIs, but this gate must not synthesize a missing wing from the tooth position;
- save empty-mask frames too, so absence is auditable;
- store the sparse and dense metrics separately;
- rank prompts deterministically with `select_prompt()`;
- create `manual_review.json` with `status: "pending"` for the five checklist entries;
- set top-level `automatic_gate_passed` only when the four positive objects pass; sharp-probe masks must never make the positive gate pass or fail;
- catch exceptions at the CLI boundary, record `status: "failed"`, `error_type`, and `error_message` in `summary.json`, then return a non-zero exit code.

- [ ] Add artifact tests with a temporary output directory. Assert that `summary.json`, `selected_prompts.json`, `manual_review.json`, masks, and overlays exist and contain relative paths. Add a test that a non-empty output directory raises `FileExistsError`.

- [ ] Run the focused tests and verify they pass.

- [ ] Commit the orchestration slice.

```powershell
git add scripts/run_cp08_sam3_gate.py tests/scripts/test_run_cp08_sam3_gate.py
git commit -m "feat: add CP08 SAM3 feasibility runner"
```

## Task 4: Add the real SAM3 CLI and provenance audit

**Files:**

- Modify: `scripts/run_cp08_sam3_gate.py`
- Modify: `tests/scripts/test_run_cp08_sam3_gate.py`
- Read: `scripts/run_sam3_prompt_matrix.py`
- Read: `src/medical_evaluation/settings.py`

- [ ] Add parser tests for required paths and stable defaults. The CLI must accept:

```text
--video-id                  choices from PRESETS
--annotations               default data/annotations
--videos                    default videos
--output-dir
--checkpoint
--sam3-root
--bpe-path
--threshold                 default 0.2
--grounding-batch-size      default 4
--device                    default cuda
```

- [ ] Implement `build_parser()` and `main()` by following the already verified SAM3 loading pattern in `run_sam3_prompt_matrix.py`. Do not add a second model loader. Instantiate one `Sam3Backend` and let each `track()` call create/reset its own session.

- [ ] Record provenance before inference:

```python
{
    "git_head": git_output("rev-parse", "HEAD"),
    "sam3_revision": git_output("-C", str(sam3_root), "rev-parse", "HEAD"),
    "checkpoint_sha256": sha256_file(checkpoint),
    "device": args.device,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "threshold": args.threshold,
    "grounding_batch_size": args.grounding_batch_size,
}
```

Also record peak allocated/reserved CUDA memory when CUDA is used. The summary must not infer a physical GPU number from PyTorch's process-local `cuda:0`; retain `CUDA_VISIBLE_DEVICES` verbatim.

- [ ] Add tests for SHA-256, Git-command failure serialization, and exception-summary writing. Tests must not require CUDA or SAM3.

- [ ] Run focused tests:

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest tests\scripts\test_run_cp08_sam3_gate.py -q
```

- [ ] Run the complete local regression and Ruff because the user did not waive testing:

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -q
& D:\Medical_evaluation\.venv\Scripts\ruff.exe check --no-cache src tests scripts\run_cp08_sam3_gate.py
```

Expected baseline: the existing 278-test baseline plus every new gate test passes; Ruff reports `All checks passed!`.

- [ ] Commit the completed local runner.

```powershell
git add scripts/run_cp08_sam3_gate.py tests/scripts/test_run_cp08_sam3_gate.py
git commit -m "feat: finalize CP08 SAM3 feasibility gate"
```

## Task 5: Push safely and update the GPU server without touching its dirty rubric

**Files:**

- Do not modify tracked source in this task.
- Preserve server file: `/home/tangm/medical_evaluation/src/medical_evaluation/rubric.py`

- [ ] Verify the local worktree contains only the three known untracked bundle files and no accidental changes.

```powershell
git status --short
git log -5 --oneline --decorate
```

- [ ] Push the feature branch.

```powershell
git push origin codex/sam3-cp09-cp11-text
```

- [ ] Before updating the server, verify branch, HEAD, dirty file, hash, GPU state, and Qwen listener exactly as required by `AGENTS.md`.

```bash
cd /home/tangm/medical_evaluation
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short src/medical_evaluation/rubric.py
sha256sum src/medical_evaluation/rubric.py
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
ss -ltnp 'sport = :8001'
```

The rubric hash must remain `14b3b33b010825f3afe605d9d77d12827f79b94c0833c6e3947c8d716903be77`. If it differs, stop and report instead of updating.

- [ ] Fetch and fast-forward only the feature branch. Do not use reset, clean, checkout of the dirty file, or stash.

```bash
git fetch origin codex/sam3-cp09-cp11-text
git merge --ff-only origin/codex/sam3-cp09-cp11-text
```

- [ ] Recheck HEAD, dirty status, and rubric hash. Run server focused tests in `video_medical`:

```bash
/home/tangm/miniconda3/bin/conda run --no-capture-output -n video_medical \
  python -m pytest tests/scripts/test_run_cp08_sam3_gate.py -q
```

## Task 6: Run the real success-video gate and visually inspect every selected mask

**Files:**

- Create untracked evidence under: `data/runs/cp08-sam3-gate-success-<git-short-head>/`
- Do not commit `data/` artifacts.

- [ ] Recheck live GPU usage. Select a GPU only if it has enough free memory for the verified SAM3.1 batch-size-4 configuration and is not occupied by another user's heavy process. If no card is suitable, immediately report the resource block and do not start inference.

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi
```

- [ ] Confirm the success video and annotation paths from the repository; do not infer them from output directory names.

```bash
find videos -maxdepth 1 -type f -printf '%f\n' | sort
find data/annotations -maxdepth 1 -type f -printf '%f\n' | sort
```

- [ ] Run the gate with the selected physical GPU index. Replace `GPU_INDEX` only with the index just verified from `nvidia-smi` and use the actual success paths listed by the preceding command.

```bash
CUDA_VISIBLE_DEVICES=GPU_INDEX \
PYTHONPATH="$PWD/src" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/home/tangm/miniconda3/bin/conda run --no-capture-output -n sam3_medical \
python scripts/run_cp08_sam3_gate.py \
  --video-id success \
  --videos videos \
  --annotations data/annotations \
  --output-dir data/runs/cp08-sam3-gate-success-$(git rev-parse --short HEAD) \
  --checkpoint models/SAM3.1/sam3.1_multiplex.pt \
  --sam3-root external/sam3 \
  --bpe-path external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --threshold 0.2 \
  --grounding-batch-size 4 \
  --device cuda
```

If the repository uses different success filenames, use the exact listed paths and record them in `summary.json`. Do not rename or overwrite annotations.

- [ ] During the run, monitor GPU memory and output timestamps. If there is no new artifact or console progress for 60 seconds, inspect the process and GPU before waiting further. On OOM, preserve `summary.json` and logs, report peak usage, and do not silently lower resolution/FPS or change the verified batch size.

- [ ] Inspect `summary.json` and every selected overlay at original resolution, including frames immediately before and after `139.814028s`. Update `manual_review.json` from `pending` to `pass` or `fail`, with a concrete Chinese note for each item. The clamp note must explicitly say whether both wings are included. The blunt-instrument note must identify the left working end near the tooth/clamp, state whether its terminal contour is visible, and must not use the thick right handle as bluntness evidence. The sharp-probe note must classify each non-empty candidate as true sharp instrument or false positive.

- [ ] The feasibility gate passes only when all of the following are true:

  - the selected blunt-instrument dense run passes the automatic gate and visual review;
  - target tooth, complete clamp, and rubber dam each pass the automatic gate and visual review in CP09's last three seconds;
  - clamp masks visibly retain both wings;
  - artifacts and provenance are complete;
  - sharp-probe outputs have been reviewed, even though they do not control the positive gate.

- [ ] If the gate passes, stop here and write the second, implementation-level CP08 plan using measured masks and colors. If it fails, report which object/prompt failed, the exact evidence paths, and whether the cause is semantic miss, fragmentation, occlusion, or resource failure. Do not implement the Judge around unusable masks.

## Task 7: Optional negative-sample audit after the success gate

**Files:**

- Create separate untracked run directories for `failure` and `clamp_failure`.

- [ ] Only after the success gate is complete, run the same command separately for each remaining annotated video. Never reuse or overwrite the success output directory.

- [ ] Treat these runs as false-positive/absence audits, not as accuracy estimates. Confirm that a negative CP08 result scanned the full CP08 stage and that missing CP09 objects are represented by saved empty masks and explicit metrics.

- [ ] Report all three runs as Demo evidence only. Do not calculate accuracy, sensitivity, specificity, or generalization claims from three videos.
