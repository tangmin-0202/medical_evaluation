# SAM2 CP09 Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the successful video's CP09 interval through real SAM2 frame tracking, produce auditable rubber-dam-frame masks and overlays, calculate frame centering, and pass the result to the existing deterministic CP09 Judge.

**Architecture:** Keep SAM-specific interaction inside `Sam2Backend`, convert persisted annotations through a pure adapter, and implement a CP09-only extractor that depends on the backend-neutral `VideoSegmenter` protocol. A standalone smoke command constructs the real backend on the GPU server; the web application remains in fake mode until the overlays are manually accepted.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Pydantic, pytest, Meta SAM2.1, existing `VideoSegmenter`, `AnnotationStore`, feature overlays, and CP09 Judge.

---

## File map

- Modify `src/medical_evaluation/features/geometry.py`: bounding-box center and normalized frame-center offset.
- Modify `tests/test_features.py`: numerical tests for centered, shifted, and empty masks.
- Create `src/medical_evaluation/segmentation/prompts.py`: convert persisted point/box annotations into backend-neutral prompts.
- Create `tests/test_segmentation_prompts.py`: prompt selection and validation tests.
- Modify `src/medical_evaluation/segmentation/sam2_backend.py`: batch same-frame points and propagate in both directions.
- Modify `tests/test_segmentation_contract.py`: fake-predictor contract tests for batching, direction, filtering, and reset.
- Create `src/medical_evaluation/extractors/__init__.py`: extractor package marker.
- Create `src/medical_evaluation/extractors/cp09.py`: CP09-only feature and evidence extraction.
- Create `tests/test_cp09_extractor.py`: fake segmenter tests for features and evidence.
- Create `scripts/smoke_sam2_cp09.py`: real GPU command and run summary.
- Create `tests/test_smoke_sam2_cp09.py`: argument and core smoke-run tests without CUDA.
- Modify `docs/server-runbook.md`: exact installation, weight, command, and artifact-review procedure.

### Task 1: Define the CP09 centering feature

**Files:**
- Modify: `src/medical_evaluation/features/geometry.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Add failing geometry tests**

Append imports and the following test to `tests/test_features.py`:

```python
from medical_evaluation.features.geometry import bounding_box_center, frame_center_offset


def test_bounding_box_center_and_frame_offset() -> None:
    centered = np.zeros((100, 200), bool)
    centered[25:75, 50:150] = True
    shifted = np.zeros((100, 200), bool)
    shifted[25:75, 100:200] = True
    empty = np.zeros((100, 200), bool)

    assert bounding_box_center(centered) == (99.5, 49.5)
    assert frame_center_offset(centered) == pytest.approx(0.0, abs=0.004)
    assert frame_center_offset(shifted) == pytest.approx(0.25, abs=0.004)
    assert bounding_box_center(empty) is None
    assert frame_center_offset(empty) is None
```

- [ ] **Step 2: Run the focused test and verify red**

Run:

```bash
python -m pytest tests/test_features.py::test_bounding_box_center_and_frame_offset -q
```

Expected: collection fails because `bounding_box_center` and `frame_center_offset` do not exist.

- [ ] **Step 3: Implement the pure feature functions**

Add to `src/medical_evaluation/features/geometry.py`:

```python
def bounding_box_center(mask: np.ndarray) -> tuple[float, float] | None:
    binary = _mask(mask)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    return (float(xs.min() + xs.max()) / 2, float(ys.min() + ys.max()) / 2)


def frame_center_offset(mask: np.ndarray) -> float | None:
    binary = _mask(mask)
    center = bounding_box_center(binary)
    if center is None:
        return None
    height, width = binary.shape
    frame_x = (width - 1) / 2
    frame_y = (height - 1) / 2
    return float(np.hypot((center[0] - frame_x) / width, (center[1] - frame_y) / height))
```

- [ ] **Step 4: Run feature tests and verify green**

Run:

```bash
python -m pytest tests/test_features.py -q
```

Expected: all feature tests pass.

- [ ] **Step 5: Commit the feature**

```bash
git add src/medical_evaluation/features/geometry.py tests/test_features.py
git commit -m "feat: measure rubber dam frame centering"
```

### Task 2: Convert reviewed annotation prompts

**Files:**
- Create: `src/medical_evaluation/segmentation/prompts.py`
- Create: `tests/test_segmentation_prompts.py`

- [ ] **Step 1: Write failing prompt-conversion tests**

Create `tests/test_segmentation_prompts.py`:

```python
import pytest

from medical_evaluation.annotations import BoxPrompt, PointPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.prompts import prompts_for_object


def test_selects_object_prompts_inside_interval() -> None:
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[
            PointPrompt(
                video_id="success", frame_time_sec=193.0,
                object_id="rubber_dam_frame", x=0.2, y=0.3,
            ),
            BoxPrompt(
                video_id="success", frame_time_sec=194.0,
                object_id="rubber_dam_frame", x1=0.1, y1=0.2, x2=0.8, y2=0.9,
            ),
            PointPrompt(
                video_id="success", frame_time_sec=100.0,
                object_id="rubber_dam_frame", x=0.4, y=0.5,
            ),
            PointPrompt(
                video_id="success", frame_time_sec=193.0,
                object_id="clamp", x=0.4, y=0.5,
            ),
        ],
    )

    result = prompts_for_object(
        annotations, "rubber_dam_frame", TimeRange(start_sec=175, end_sec=195)
    )

    assert [(item.kind, item.coordinates) for item in result] == [
        ("point", [0.2, 0.3]),
        ("box", [0.1, 0.2, 0.8, 0.9]),
    ]


def test_missing_required_prompt_is_actionable() -> None:
    with pytest.raises(ValueError, match="success.*cp_09.*rubber_dam_frame"):
        prompts_for_object(
            VideoAnnotations(video_id="success"),
            "rubber_dam_frame",
            TimeRange(start_sec=175, end_sec=195),
            checkpoint_id="cp_09",
        )
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
python -m pytest tests/test_segmentation_prompts.py -q
```

Expected: import fails because `segmentation.prompts` does not exist.

- [ ] **Step 3: Implement prompt conversion**

Create `src/medical_evaluation/segmentation/prompts.py`:

```python
from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import SegmentationPrompt


def prompts_for_object(
    annotations: VideoAnnotations,
    object_id: str,
    time_range: TimeRange,
    *,
    checkpoint_id: str = "cp_09",
) -> list[SegmentationPrompt]:
    converted: list[SegmentationPrompt] = []
    for prompt in annotations.prompts:
        if prompt.object_id != object_id:
            continue
        if not time_range.start_sec <= prompt.frame_time_sec <= time_range.end_sec:
            continue
        if prompt.kind == "point":
            converted.append(
                SegmentationPrompt(
                    object_id=object_id,
                    kind="point",
                    frame_time_sec=prompt.frame_time_sec,
                    coordinates=[prompt.x, prompt.y],
                    positive=prompt.positive,
                )
            )
        elif prompt.kind == "box":
            converted.append(
                SegmentationPrompt(
                    object_id=object_id,
                    kind="box",
                    frame_time_sec=prompt.frame_time_sec,
                    coordinates=[prompt.x1, prompt.y1, prompt.x2, prompt.y2],
                )
            )
    if not converted:
        raise ValueError(
            f"{annotations.video_id} {checkpoint_id} has no {object_id} prompt inside "
            f"{time_range.start_sec:.3f}-{time_range.end_sec:.3f}s"
        )
    return sorted(converted, key=lambda item: item.frame_time_sec)
```

- [ ] **Step 4: Run prompt tests and verify green**

Run:

```bash
python -m pytest tests/test_segmentation_prompts.py -q
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit the converter**

```bash
git add src/medical_evaluation/segmentation/prompts.py tests/test_segmentation_prompts.py
git commit -m "feat: convert reviewed SAM prompts"
```

### Task 3: Batch points and track in both directions

**Files:**
- Modify: `src/medical_evaluation/segmentation/sam2_backend.py`
- Modify: `tests/test_segmentation_contract.py`

- [ ] **Step 1: Add a fake predictor and failing backend tests**

Add to `tests/test_segmentation_contract.py`:

```python
from medical_evaluation.video import VideoMetadata


class FakeVideoPredictor:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, object]] = []
        self.propagate_calls: list[tuple[int, bool]] = []
        self.reset = False

    def init_state(self, *, video_path: str) -> dict[str, object]:
        return {"video_path": video_path}

    def add_new_points_or_box(self, **kwargs: object) -> None:
        self.add_calls.append(kwargs)

    def propagate_in_video(
        self, _state: object, *, start_frame_idx: int, reverse: bool
    ) -> list[tuple[int, list[int], np.ndarray]]:
        self.propagate_calls.append((start_frame_idx, reverse))
        indexes = [190, 180, 170] if reverse else [190, 200]
        return [
            (index, [1], np.ones((1, 1, 4, 4), dtype=np.float32))
            for index in indexes
        ]

    def reset_state(self, _state: object) -> None:
        self.reset = True


def test_sam2_batches_same_frame_points_and_tracks_both_directions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predictor = FakeVideoPredictor()
    monkeypatch.setattr(
        "medical_evaluation.segmentation.sam2_backend.probe_video",
        lambda _path: VideoMetadata(
            path=tmp_path / "video.mp4", duration_sec=30, fps=10,
            frame_count=300, width=100, height=50,
        ),
    )
    backend = Sam2Backend("cfg", tmp_path / "weights.pt", predictor=predictor)
    prompts = [
        SegmentationPrompt(
            object_id="rubber_dam_frame", kind="point",
            frame_time_sec=19, coordinates=[0.2, 0.3],
        ),
        SegmentationPrompt(
            object_id="rubber_dam_frame", kind="point",
            frame_time_sec=19, coordinates=[0.8, 0.7],
        ),
    ]

    frames = list(
        backend.track(
            tmp_path / "video.mp4", TimeRange(start_sec=17, end_sec=20),
            prompts, sample_fps=1,
        )
    )

    assert len(predictor.add_calls) == 1
    assert np.asarray(predictor.add_calls[0]["points"]).shape == (2, 2)
    assert predictor.propagate_calls == [(190, False), (190, True)]
    assert [frame.frame_index for frame in frames] == [170, 180, 190, 200]
    assert predictor.reset is True
```

- [ ] **Step 2: Run the focused test and verify red**

Run:

```bash
python -m pytest tests/test_segmentation_contract.py::test_sam2_batches_same_frame_points_and_tracks_both_directions -q
```

Expected: failure because the backend submits two point calls and does not pass reverse propagation arguments.

- [ ] **Step 3: Refactor the backend around grouped inputs**

In `src/medical_evaluation/segmentation/sam2_backend.py`, add these internal grouping types and helper (with `dataclass` imported):

```python
@dataclass
class _PromptGroup:
    object_id: str
    frame_index: int
    points: list[np.ndarray]
    labels: list[int]
    box: np.ndarray | None = None


def _group_prompts(
    prompts: list[SegmentationPrompt], metadata: VideoMetadata
) -> list[_PromptGroup]:
    grouped: dict[tuple[str, int], _PromptGroup] = {}
    for prompt in prompts:
        if prompt.kind in {"mask", "text"}:
            continue
        frame_index = round(prompt.frame_time_sec * metadata.fps)
        key = prompt.object_id, frame_index
        group = grouped.setdefault(
            key,
            _PromptGroup(prompt.object_id, frame_index, [], []),
        )
        assert prompt.coordinates is not None
        pixels = normalized_to_pixels(prompt.coordinates, metadata.width, metadata.height)
        if prompt.kind == "point":
            group.points.append(pixels)
            group.labels.append(int(prompt.positive))
        elif group.box is not None:
            raise ValueError(
                f"multiple boxes for {prompt.object_id} on frame {frame_index}"
            )
        else:
            group.box = pixels
    return [grouped[key] for key in sorted(grouped, key=lambda item: (item[1], item[0]))]
```

Import `VideoMetadata` beside `probe_video`. Convert each group's points to one `N x 2` array and labels to one `N` array before the predictor call.

Replace the per-prompt submission loop with:

```python
groups = _group_prompts(prompts, metadata)
conditioning_frames: list[int] = []
for group in groups:
    conditioning_frames.append(group.frame_index)
    self.predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=group.frame_index,
        obj_id=object_numbers[group.object_id],
        points=group.points if len(group.points) else None,
        labels=group.labels if len(group.labels) else None,
        box=group.box,
        clear_old_points=True,
        normalize_coords=False,
    )
```

Propagate from the earliest conditioning frame forward and the latest conditioning frame backward. Store normalized results in a dictionary keyed by frame index, filter by time range and stride, then yield sorted frames. Keep `reset_state(state)` in `finally`.

Use this propagation shape:

```python
results: dict[int, FrameMasks] = {}
directions = [(min(conditioning_frames), False), (max(conditioning_frames), True)]
for start_frame_idx, reverse in directions:
    for frame_index, object_ids, logits in self.predictor.propagate_in_video(
        state, start_frame_idx=start_frame_idx, reverse=reverse
    ):
        frame_time = frame_index / metadata.fps
        if not time_range.start_sec <= frame_time <= time_range.end_sec:
            continue
        if frame_index % stride:
            continue
        raw = {
            "frame": frame_index,
            "objects": {
                reverse_ids[int(object_id)]: mask
                for object_id, mask in zip(object_ids, logits, strict=True)
            },
        }
        results[frame_index] = normalize_masks(
            raw, threshold=0, frame_time_sec=frame_time
        )
for frame_index in sorted(results):
    yield results[frame_index]
```

For mask prompts, continue using `add_new_mask` before point/box propagation; include their frame indices in `conditioning_frames`.

- [ ] **Step 4: Run segmentation tests and verify green**

Run:

```bash
python -m pytest tests/test_segmentation_contract.py tests/test_segmentation_prompts.py -q
```

Expected: all segmentation tests pass, including one add call and two propagation calls.

- [ ] **Step 5: Commit the backend correction**

```bash
git add src/medical_evaluation/segmentation/sam2_backend.py tests/test_segmentation_contract.py
git commit -m "fix: preserve SAM2 multipoint video prompts"
```

### Task 4: Implement the CP09 extractor and evidence

**Files:**
- Create: `src/medical_evaluation/extractors/__init__.py`
- Create: `src/medical_evaluation/extractors/cp09.py`
- Modify: `src/medical_evaluation/video.py`
- Create: `tests/test_cp09_extractor.py`

- [ ] **Step 1: Write failing extractor tests**

Create `tests/test_cp09_extractor.py` with a fake segmenter yielding three `FrameMasks`. Use a 20 x 40 temporary AVI created through `cv2.VideoWriter`. Assert that:

```python
result = extractor.extract(
    video_path, "cp_09", TimeRange(start_sec=1, end_sec=2),
    dense_fps=2, analysis_width=1280,
)
assert result.features["frame_center_offset"] == pytest.approx(0.0, abs=0.03)
assert len(result.evidence) == 3
assert all((evidence_root / item.overlay_path).is_file() for item in result.evidence)
```

Add a second test whose fake segmenter yields empty masks and assert:

```python
assert result.features == {"frame_center_offset": None}
assert result.evidence == []
```

Add a third test with only two valid masks and assert the feature remains `None` while the two overlays are retained for human review. Add a fourth test asserting a checkpoint other than `cp_09` raises `ValueError` containing `CP09-only`.

- [ ] **Step 2: Run extractor tests and verify red**

Run:

```bash
python -m pytest tests/test_cp09_extractor.py -q
```

Expected: import fails because `medical_evaluation.extractors.cp09` does not exist.

- [ ] **Step 3: Add exact-frame reading**

Add to `src/medical_evaluation/video.py`:

```python
def read_frame(path: Path, frame_index: int) -> np.ndarray:
    metadata = probe_video(path)
    if frame_index < 0 or frame_index >= metadata.frame_count:
        raise ValueError(f"frame index {frame_index} is outside video")
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, image = capture.read()
        if not success or image is None:
            raise ValueError(f"failed to decode frame {frame_index}")
        return image.copy()
    finally:
        capture.release()
```

- [ ] **Step 4: Implement the CP09-only extractor**

Create `src/medical_evaluation/extractors/__init__.py` as an empty package file. Create `src/medical_evaluation/extractors/cp09.py` with:

```python
from pathlib import Path

import cv2
import numpy as np

from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.features.geometry import frame_center_offset, write_overlay
from medical_evaluation.pipeline import ExtractedEvidence
from medical_evaluation.reporting import EvidenceItem
from medical_evaluation.segmentation.base import FrameMasks, VideoSegmenter
from medical_evaluation.segmentation.prompts import prompts_for_object
from medical_evaluation.storage import safe_child
from medical_evaluation.video import read_frame


class Cp09FeatureExtractor:
    minimum_valid_frames = 3

    def __init__(
        self,
        *,
        segmenter: VideoSegmenter,
        annotations: VideoAnnotations,
        evidence_root: Path,
    ) -> None:
        self.segmenter = segmenter
        self.annotations = annotations
        self.evidence_root = evidence_root

    @property
    def model_version(self) -> str:
        return self.segmenter.model_version

    def extract(
        self,
        video_path: Path,
        checkpoint_id: str,
        time_range: TimeRange,
        *,
        dense_fps: float,
        analysis_width: int,
    ) -> ExtractedEvidence:
        if checkpoint_id != "cp_09":
            raise ValueError("Cp09FeatureExtractor is CP09-only")
        if analysis_width <= 0:
            raise ValueError("analysis_width must be positive")
        prompts = prompts_for_object(
            self.annotations,
            "rubber_dam_frame",
            time_range,
            checkpoint_id=checkpoint_id,
        )
        tracked = list(
            self.segmenter.track(video_path, time_range, prompts, sample_fps=dense_fps)
        )
        valid: list[tuple[FrameMasks, np.ndarray, float]] = []
        for frame_masks in tracked:
            mask = frame_masks.masks.get("rubber_dam_frame")
            if mask is None:
                continue
            offset = frame_center_offset(mask)
            if offset is not None:
                valid.append((frame_masks, mask, offset))

        evidence: list[EvidenceItem] = []
        selected = sorted({0, len(valid) // 2, len(valid) - 1}) if valid else []
        for position in selected:
            frame_masks, mask, _offset = valid[position]
            frame = read_frame(video_path, frame_masks.frame_index)
            mask_path = safe_child(
                self.evidence_root, f"cp_09/masks/{frame_masks.frame_index:08d}.png"
            )
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
                raise OSError(f"could not write mask: {mask_path}")
            overlay = write_overlay(
                frame,
                {"rubber_dam_frame": mask},
                self.evidence_root,
                f"cp_09/overlays/{frame_masks.frame_index:08d}.jpg",
            )
            evidence.append(
                EvidenceItem(
                    time_sec=frame_masks.frame_time_sec,
                    overlay_path=overlay.relative_to(self.evidence_root).as_posix(),
                    rule="rubber_dam_frame_center",
                )
            )

        feature = (
            float(np.median([item[2] for item in valid]))
            if len(valid) >= self.minimum_valid_frames
            else None
        )
        return ExtractedEvidence(
            features={"frame_center_offset": feature}, evidence=evidence
        )
```

The `extract` method must:

1. reject checkpoint IDs other than `cp_09`;
2. call `prompts_for_object(self.annotations, "rubber_dam_frame", time_range, checkpoint_id="cp_09")`;
3. materialize `self.segmenter.track(video_path, time_range, prompts, sample_fps=dense_fps)`;
4. keep frames with a non-empty `rubber_dam_frame` mask;
5. compute every frame's `frame_center_offset`; return the median only when at least `minimum_valid_frames` offsets exist, otherwise return `None`;
6. select first, middle, and last valid frames without duplicate indices;
7. decode each selected source frame with `read_frame`;
8. write overlays beneath `evidence_root` using `write_overlay`;
9. return `ExtractedEvidence` with `EvidenceItem` entries whose relative paths are POSIX strings.

Use `safe_child` for mask PNG paths, save masks as `uint8 * 255`, and use the rule string `rubber_dam_frame_center`.

- [ ] **Step 5: Run extractor and feature tests**

Run:

```bash
python -m pytest tests/test_cp09_extractor.py tests/test_features.py -q
```

Expected: all tests pass and the temporary evidence files are non-empty.

- [ ] **Step 6: Commit the extractor**

```bash
git add src/medical_evaluation/extractors src/medical_evaluation/video.py tests/test_cp09_extractor.py
git commit -m "feat: extract CP09 SAM2 evidence"
```

### Task 5: Add the real GPU smoke command

**Files:**
- Create: `scripts/smoke_sam2_cp09.py`
- Create: `tests/test_smoke_sam2_cp09.py`
- Modify: `docs/server-runbook.md`

- [ ] **Step 1: Write failing smoke-core tests**

Create `tests/test_smoke_sam2_cp09.py`. Import `run_cp09_smoke` from the script and pass a fake extractor returning `ExtractedEvidence(features={"frame_center_offset": 0.03})`. Assert that `summary.json` and `decision.json` are written beneath the supplied run root and that the decision status is `correct`.

Add validation tests asserting `run_cp09_smoke` rejects `video_id="failure"` and `checkpoint_id="cp_10"` with messages containing `success` and `cp_09` respectively.

- [ ] **Step 2: Run smoke tests and verify red**

Run:

```bash
python -m pytest tests/test_smoke_sam2_cp09.py -q
```

Expected: import fails because `scripts/smoke_sam2_cp09.py` does not exist.

- [ ] **Step 3: Implement the smoke core and CLI**

Create `scripts/smoke_sam2_cp09.py` with:

```python
def run_cp09_smoke(
    *, extractor: Cp09FeatureExtractor, video_path: Path,
    annotations: VideoAnnotations, rubric: Rubric, run_root: Path,
    video_id: str = "success", checkpoint_id: str = "cp_09",
    sample_fps: float = 2.0, degradations: list[str] | None = None,
) -> dict[str, object]:
```

Require `video_id == "success"` and `checkpoint_id == "cp_09"`. Implement the core with this data flow:

```python
if video_id != "success":
    raise ValueError("CP09 smoke only accepts video_id=success")
if checkpoint_id != "cp_09":
    raise ValueError("CP09 smoke only accepts checkpoint_id=cp_09")
rule = next(item for item in rubric.checkpoints if item.id == checkpoint_id)
segment = next(item for item in annotations.steps if item.checkpoint_id == checkpoint_id)
started = datetime.now(UTC)
result = extractor.extract(
    video_path,
    checkpoint_id,
    segment.time_range,
    dense_fps=sample_fps,
    analysis_width=1280,
)
decision = judge_cp09(result.features, rule.thresholds)
completed = datetime.now(UTC)
prompt_count = sum(
    prompt.object_id == "rubber_dam_frame"
    and segment.time_range.start_sec <= prompt.frame_time_sec <= segment.time_range.end_sec
    for prompt in annotations.prompts
)
summary = {
    "video_id": video_id,
    "checkpoint_id": checkpoint_id,
    "time_range": segment.time_range.model_dump(mode="json"),
    "prompt_count": prompt_count,
    "evidence_count": len(result.evidence),
    "features": result.features,
    "model_version": extractor.model_version,
    "sample_fps": sample_fps,
    "started_at": started.isoformat(),
    "completed_at": completed.isoformat(),
    "runtime_sec": (completed - started).total_seconds(),
    "degradations": list(degradations or []),
}
atomic_write_json(run_root / "summary.json", summary)
atomic_write_json(run_root / "decision.json", decision.model_dump(mode="json"))
return summary
```

This writes:

- `summary.json` with video ID, checkpoint ID, time range, prompt count, valid evidence count, feature values, model version, sample FPS, UTC start/completion, runtime, and degradations;
- `decision.json` with the serialized `JudgeDecision`.

The CLI must parse `--checkpoint-path`, `--model-config`, `--device`, `--sample-fps`, `--videos-dir`, `--data-dir`, and optional `--output-dir`. Construct `Sam2Backend`, load `data/annotations/success.json`, load `config/rubric.yaml`, create a timestamped output directory under `data/runs`, and run once. Print the absolute summary and overlay directories.

Catch CUDA OOM once: call `torch.cuda.empty_cache()` when available, retry with `sample_fps=min(original, 1.0)`, and append `cuda_oom:sample_fps=1.0` to the summary for the default 2 FPS run. Re-raise a second OOM.

- [ ] **Step 4: Run smoke-core tests and all CPU tests**

Run:

```bash
python -m pytest tests/test_smoke_sam2_cp09.py -q
python -m pytest -q
ruff check src scripts tests
```

Expected: smoke tests pass; the full suite passes; Ruff reports no errors. These commands must not import or load SAM2 weights.

- [ ] **Step 5: Document the exact server procedure**

Append a `SAM2 CP09 纵向切片` section to `docs/server-runbook.md` containing:

```bash
conda activate video_medical
cd ~/medical_evaluation
git pull --ff-only
mkdir -p external models
test -d external/sam2/.git || git clone https://github.com/facebookresearch/sam2.git external/sam2
pip install -e external/sam2
cd external/sam2/checkpoints
./download_ckpts.sh
cd ~/medical_evaluation
git -C external/sam2 rev-parse HEAD
sha256sum external/sam2/checkpoints/sam2.1_hiera_large.pt
nvidia-smi
CUDA_VISIBLE_DEVICES=1 python scripts/smoke_sam2_cp09.py \
  --checkpoint-path external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --model-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda:0 \
  --sample-fps 2 \
  --videos-dir "$HOME/medical_evaluation/videos" \
  --data-dir "$HOME/medical_evaluation/data"
```

Document that `CUDA_VISIBLE_DEVICES=1` makes physical GPU 1 appear as `cuda:0` inside the process, and require checking current `nvidia-smi` before selecting it. Document the three overlay images to review and the rule that the web service remains `MED_EVAL_PIPELINE_MODE=fake`.

- [ ] **Step 6: Commit the smoke command and runbook**

```bash
git add scripts/smoke_sam2_cp09.py tests/test_smoke_sam2_cp09.py docs/server-runbook.md
git commit -m "feat: add SAM2 CP09 GPU smoke run"
```

### Task 6: Final local and server verification

**Files:**
- Verify only; no planned source changes.

- [ ] **Step 1: Run fresh local verification**

Run:

```bash
python -m pytest -q
ruff check src scripts tests
git diff --check
git status --short
```

Expected: all tests pass, Ruff and diff checks are clean, and only the user's pre-existing untracked data/video/IDE files remain.

- [ ] **Step 2: Push the implementation branch**

Run:

```bash
git push -u origin codex/sam2-cp09-vertical-slice
```

Expected: the branch is available on GitHub; do not merge before the GPU smoke result is reviewed.

- [ ] **Step 3: Run the documented command on the GPU server**

Run the exact command from Task 5 after choosing a currently free GPU. Expected: exit code 0; `summary.json`, `decision.json`, non-empty masks, and at least three overlay images exist.

- [ ] **Step 4: Review evidence before expanding scope**

Open overlays from the early, middle, and late CP09 interval. Accept the slice only if masks cover the rubber-dam frame rather than hands, face, dam sheet, or background. Record observed runtime, GPU, checkpoint digest, valid-frame count, and any degradation. If masks are wrong, keep the web pipeline fake and revise prompts/backend before implementing other checkpoints.
