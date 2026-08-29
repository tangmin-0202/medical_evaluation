# SAM2 Windowed Stage Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every SAM2 tracking call load only its requested stage window so CP09, and later CP01 through CP11, can run sequentially on a 24 GiB RTX A5000.

**Architecture:** Add a generic OpenCV frame-sequence writer that exports the requested time range at the requested FPS plus mandatory prompt frames and returns a local-to-source timeline. Update `Sam2Backend` to initialize the official predictor with that temporary JPEG directory, translate prompts to local indices, and restore original frame numbers and timestamps in results.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, Meta SAM2 predictor API, pytest, Ruff

---

### Task 1: Windowed JPEG sequence and source timeline

**Files:**
- Modify: `src/medical_evaluation/video.py`
- Modify: `tests/test_video.py`

- [ ] **Step 1: Write failing tests for regular sampling, mandatory prompt frames, deduplication, and validation**

Append tests that create a ten-FPS fixture and call the new utility:

```python
def test_write_sampled_frame_sequence_includes_required_prompt_frame(tmp_path: Path) -> None:
    from medical_evaluation.domain import TimeRange
    from medical_evaluation.video import write_sampled_frame_sequence

    video = make_test_video(tmp_path / "sample.mp4", fps=10, seconds=3, size=(160, 120))
    output = tmp_path / "frames"

    sequence = write_sampled_frame_sequence(
        video,
        output,
        time_range=TimeRange(start_sec=1.0, end_sec=2.5),
        sample_fps=2,
        required_times_sec=[1.3, 2.0],
    )

    assert [item.source_frame_index for item in sequence.entries] == [10, 13, 15, 20]
    assert [item.local_frame_index for item in sequence.entries] == [0, 1, 2, 3]
    assert [path.name for path in sorted(output.glob("*.jpg"))] == [
        "00000.jpg",
        "00001.jpg",
        "00002.jpg",
        "00003.jpg",
    ]
    assert sequence.source_to_local[13] == 1
    assert sequence.entries[1].source_time_sec == pytest.approx(1.3)


def test_lower_fps_writes_fewer_non_prompt_frames(tmp_path: Path) -> None:
    from medical_evaluation.domain import TimeRange
    from medical_evaluation.video import write_sampled_frame_sequence

    video = make_test_video(tmp_path / "sample.mp4", fps=10, seconds=4, size=(160, 120))
    interval = TimeRange(start_sec=1, end_sec=3)
    two_fps = write_sampled_frame_sequence(
        video, tmp_path / "two", time_range=interval, sample_fps=2, required_times_sec=[1.3]
    )
    one_fps = write_sampled_frame_sequence(
        video, tmp_path / "one", time_range=interval, sample_fps=1, required_times_sec=[1.3]
    )

    assert len(one_fps.entries) < len(two_fps.entries)
    assert 13 in one_fps.source_to_local
    assert 13 in two_fps.source_to_local


def test_windowed_sequence_rejects_prompt_outside_interval(tmp_path: Path) -> None:
    from medical_evaluation.domain import TimeRange
    from medical_evaluation.video import write_sampled_frame_sequence

    video = make_test_video(tmp_path / "sample.mp4", fps=10, seconds=3, size=(160, 120))
    with pytest.raises(ValueError, match="prompt time"):
        write_sampled_frame_sequence(
            video,
            tmp_path / "frames",
            time_range=TimeRange(start_sec=1, end_sec=2),
            sample_fps=2,
            required_times_sec=[2.5],
        )
```

- [ ] **Step 2: Run the new tests and verify they fail for the missing API**

Run:

```bash
python -m pytest -p no:cacheprovider tests/test_video.py -q
```

Expected: FAIL because `write_sampled_frame_sequence` does not exist.

- [ ] **Step 3: Implement the focused frame-sequence utility**

Add immutable timeline types and a writer to `video.py`:

```python
from dataclasses import dataclass

from medical_evaluation.domain import TimeRange


@dataclass(frozen=True)
class FrameTimelineEntry:
    local_frame_index: int
    source_frame_index: int
    source_time_sec: float


@dataclass(frozen=True)
class SampledFrameSequence:
    directory: Path
    metadata: VideoMetadata
    entries: tuple[FrameTimelineEntry, ...]
    source_to_local: dict[int, int]


def write_sampled_frame_sequence(
    path: Path,
    output_dir: Path,
    *,
    time_range: TimeRange,
    sample_fps: float,
    required_times_sec: list[float],
) -> SampledFrameSequence:
    metadata = probe_video(path)
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")
    if time_range.end_sec > metadata.duration_sec + 1e-6:
        raise ValueError("sample range is outside video duration")
    if any(
        time_sec < time_range.start_sec or time_sec > time_range.end_sec
        for time_sec in required_times_sec
    ):
        raise ValueError("prompt time is outside the requested stage interval")

    source_indices: set[int] = set()
    time_sec = time_range.start_sec
    while time_sec < time_range.end_sec - 1e-9:
        source_indices.add(min(round(time_sec * metadata.fps), metadata.frame_count - 1))
        time_sec += 1.0 / sample_fps
    source_indices.update(
        min(round(time_sec * metadata.fps), metadata.frame_count - 1)
        for time_sec in required_times_sec
    )
    ordered_indices = sorted(source_indices)

    output_dir.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(path))
    entries: list[FrameTimelineEntry] = []
    try:
        for local_index, source_index in enumerate(ordered_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            success, image = capture.read()
            if not success or image is None:
                raise ValueError(f"failed to decode frame {source_index}")
            destination = output_dir / f"{local_index:05d}.jpg"
            if not cv2.imwrite(str(destination), image):
                raise ValueError(f"failed to write sampled frame {destination}")
            entries.append(
                FrameTimelineEntry(local_index, source_index, source_index / metadata.fps)
            )
    finally:
        capture.release()

    return SampledFrameSequence(
        directory=output_dir,
        metadata=metadata,
        entries=tuple(entries),
        source_to_local={item.source_frame_index: item.local_frame_index for item in entries},
    )
```

During implementation, keep output-directory creation before decoding and let the caller own cleanup. Use exact source frame indices so mandatory prompt frames are never approximated twice.

- [ ] **Step 4: Run focused tests and Ruff**

Run:

```bash
python -m pytest -p no:cacheprovider tests/test_video.py -q
python -m ruff check --no-cache src/medical_evaluation/video.py tests/test_video.py
```

Expected: all video tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 5: Commit the frame-sequence unit**

```bash
git add src/medical_evaluation/video.py tests/test_video.py
git commit -m "feat: export sampled stage frame sequences"
```

### Task 2: Remap SAM2 prompts and outputs through the window timeline

**Files:**
- Modify: `src/medical_evaluation/segmentation/sam2_backend.py`
- Modify: `tests/test_segmentation_contract.py`

- [ ] **Step 1: Replace the fake predictor with local-sequence propagation and add regression assertions**

Make `FakeVideoPredictor.init_state` record the passed path and count numbered JPEG files. Its propagation returns local indices:

```python
class FakeVideoPredictor:
    def __init__(self, *, fail_on_add: bool = False) -> None:
        self.add_calls: list[dict[str, object]] = []
        self.propagate_calls: list[tuple[int, bool, int]] = []
        self.init_video_path: Path | None = None
        self.frame_count = 0
        self.reset = False
        self.fail_on_add = fail_on_add

    def init_state(self, *, video_path: str) -> dict[str, object]:
        self.init_video_path = Path(video_path)
        self.frame_count = len(list(self.init_video_path.glob("*.jpg")))
        return {"video_path": video_path}

    def propagate_in_video(
        self,
        _state: object,
        *,
        start_frame_idx: int,
        reverse: bool,
        max_frame_num_to_track: int,
    ) -> list[tuple[int, list[int], np.ndarray]]:
        self.propagate_calls.append((start_frame_idx, reverse, max_frame_num_to_track))
        stop = max(-1, start_frame_idx - max_frame_num_to_track) if reverse else min(
            self.frame_count, start_frame_idx + max_frame_num_to_track
        )
        indexes = (
            range(start_frame_idx, stop, -1)
            if reverse
            else range(start_frame_idx, stop)
        )
        return [
            (index, [1], np.ones((1, 1, 4, 4), dtype=np.float32))
            for index in indexes
        ]
```

Use a ten-FPS, three-second `make_test_video` fixture. Request the generic
interval 1.0 to 2.5 seconds at two FPS and place two point prompts at 1.3
seconds. The sampled source indices are then 10, 13, 15, and 20, and the prompt
is local index 1. Assert the exact mapping and cleanup:

```python
assert predictor.init_video_path is not None
sample_directory = predictor.init_video_path
assert sample_directory.is_dir() is False  # cleaned after iterator completion
assert sample_directory.suffix != ".mp4"
assert predictor.add_calls[0]["frame_idx"] == 1
assert predictor.propagate_calls == [(1, False, 3), (1, True, 2)]
assert [frame.frame_index for frame in frames] == [10, 13, 15, 20]
assert [frame.frame_time_sec for frame in frames] == pytest.approx([1.0, 1.3, 1.5, 2.0])
assert predictor.reset is True
```

Extend the prompt-submission failure test to assert the recorded temporary directory no longer exists after the exception.

- [ ] **Step 2: Run the segmentation contract tests and verify the full-MP4 behavior fails the new assertions**

Run:

```bash
python -m pytest -p no:cacheprovider tests/test_segmentation_contract.py -q
```

Expected: FAIL because `init_state` still receives the original MP4 and prompt indices are still source-video indices.

- [ ] **Step 3: Initialize SAM2 from a temporary sampled sequence**

Refactor `track` to prepare and own a temporary directory:

```python
from tempfile import TemporaryDirectory

from medical_evaluation.video import SampledFrameSequence, write_sampled_frame_sequence


def track(
    self,
    video_path: Path,
    time_range: TimeRange,
    prompts: list[SegmentationPrompt],
    sample_fps: float,
) -> Iterator[FrameMasks]:
    if any(prompt.kind == "text" for prompt in prompts):
        raise ValueError("SAM2 does not support text-only prompts; use point/box/mask or SAM3")
    if not prompts:
        raise ValueError("SAM2 requires at least one point, box, or mask prompt")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    required_times = [prompt.frame_time_sec for prompt in prompts if prompt.kind != "text"]
    with TemporaryDirectory(prefix="medical-evaluation-sam2-") as temporary_root:
        sequence = write_sampled_frame_sequence(
            video_path,
            Path(temporary_root) / "frames",
            time_range=time_range,
            sample_fps=sample_fps,
            required_times_sec=required_times,
        )
        state = self.predictor.init_state(video_path=str(sequence.directory))
        try:
            yield from self._track_initialized(state, sequence, prompts)
        finally:
            if hasattr(self.predictor, "reset_state"):
                self.predictor.reset_state(state)
```

Update `_track_initialized` to use local sequence bounds instead of source-video FPS strides:

```python
def _track_initialized(
    self,
    state: Any,
    sequence: SampledFrameSequence,
    prompts: list[SegmentationPrompt],
) -> Iterator[FrameMasks]:
    metadata = sequence.metadata
    object_numbers = {
        object_id: number
        for number, object_id in enumerate(
            dict.fromkeys(prompt.object_id for prompt in prompts), start=1
        )
    }
    reverse_ids = {number: object_id for object_id, number in object_numbers.items()}
    conditioning_frames: list[int] = []
    for prompt in (item for item in prompts if item.kind == "mask"):
        source_index = round(prompt.frame_time_sec * metadata.fps)
        local_index = sequence.source_to_local[source_index]
        conditioning_frames.append(local_index)
        self.predictor.add_new_mask(
            inference_state=state,
            frame_idx=local_index,
            obj_id=object_numbers[prompt.object_id],
            mask=prompt.mask,
        )
    for group in _group_prompts(prompts, sequence):
        conditioning_frames.append(group.frame_index)
        points = np.asarray(group.points, dtype=np.float32).reshape(-1, 2)
        labels = np.asarray(group.labels, dtype=np.int32)
        self.predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=group.frame_index,
            obj_id=object_numbers[group.object_id],
            points=points if len(points) else None,
            labels=labels if len(labels) else None,
            box=group.box,
            clear_old_points=True,
            normalize_coords=False,
        )

    results: dict[int, FrameMasks] = {}
    first_local = 0
    last_local = len(sequence.entries) - 1
    forward_start = min(conditioning_frames)
    reverse_start = max(conditioning_frames)
    directions = [
        (forward_start, False, last_local - forward_start + 1),
        (reverse_start, True, reverse_start - first_local + 1),
    ]
    for start_frame_idx, reverse, maximum_frames in directions:
        propagation = self.predictor.propagate_in_video(
            state,
            start_frame_idx=start_frame_idx,
            reverse=reverse,
            max_frame_num_to_track=maximum_frames,
        )
        for local_index, object_ids, logits in propagation:
            timeline = sequence.entries[local_index]
            raw = {
                "frame": timeline.source_frame_index,
                "objects": {
                    reverse_ids[int(object_id)]: mask
                    for object_id, mask in zip(object_ids, logits, strict=True)
                },
            }
            results[timeline.source_frame_index] = normalize_masks(
                raw,
                threshold=0,
                frame_time_sec=timeline.source_time_sec,
            )
    for source_index in sorted(results):
        yield results[source_index]
```

Change `_group_prompts` to use the same timeline and original image dimensions:

```python
def _group_prompts(
    prompts: list[SegmentationPrompt],
    sequence: SampledFrameSequence,
) -> list[_PromptGroup]:
    metadata = sequence.metadata
    grouped: dict[tuple[str, int], _PromptGroup] = {}
    for prompt in prompts:
        if prompt.kind in {"mask", "text"}:
            continue
        source_index = round(prompt.frame_time_sec * metadata.fps)
        local_index = sequence.source_to_local[source_index]
        key = prompt.object_id, local_index
        group = grouped.setdefault(
            key,
            _PromptGroup(prompt.object_id, local_index, [], []),
        )
        assert prompt.coordinates is not None
        pixels = normalized_to_pixels(prompt.coordinates, metadata.width, metadata.height)
        if prompt.kind == "point":
            group.points.append(pixels)
            group.labels.append(int(prompt.positive))
        elif group.box is not None:
            raise ValueError(f"multiple boxes for {prompt.object_id} on frame {local_index}")
        else:
            group.box = pixels
    return [grouped[key] for key in sorted(grouped, key=lambda item: (item[1], item[0]))]
```

Remove the old post-propagation FPS stride because the input sequence is already
sampled.

- [ ] **Step 4: Run segmentation and CP09 focused tests**

Run:

```bash
python -m pytest -p no:cacheprovider tests/test_segmentation_contract.py tests/test_cp09_extractor.py tests/test_smoke_sam2_cp09.py -q
python -m ruff check --no-cache src/medical_evaluation/segmentation/sam2_backend.py tests/test_segmentation_contract.py
```

Expected: focused tests pass; temporary frame paths are gone after both success and prompt failure.

- [ ] **Step 5: Commit the generic SAM2 window adapter**

```bash
git add src/medical_evaluation/segmentation/sam2_backend.py tests/test_segmentation_contract.py
git commit -m "fix: window SAM2 input by assessment stage"
```

### Task 3: Server dependency documentation and complete verification

**Files:**
- Modify: `docs/server-runbook.md`

- [ ] **Step 1: Document the MP4 decoder dependency and the corrected OOM behavior**

In the SAM2 CP09 section, add the exact minimal install and verification commands:

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple eva-decord
python -c "import decord; print(decord.__version__)"
```

State that the backend exports only the current checkpoint interval at the requested FPS, includes reviewed prompt frames, and retries once at one FPS after CUDA OOM. Record that physical GPU 7 appears as `cuda:0` under `CUDA_VISIBLE_DEVICES=7`.

- [ ] **Step 2: Run the complete local verification suite**

Run:

```bash
python -m pytest -p no:cacheprovider -q
python -m ruff check --no-cache src scripts tests
git diff --check
```

Expected: all tests pass, Ruff passes, and `git diff --check` produces no errors.

- [ ] **Step 3: Commit documentation and verification record**

```bash
git add docs/server-runbook.md
git commit -m "docs: explain windowed SAM2 smoke input"
```

- [ ] **Step 4: Push and perform the A5000 smoke rerun**

Push `codex/sam2-cp09-vertical-slice`, then on the server run:

```bash
git pull --ff-only
CUDA_VISIBLE_DEVICES=7 python scripts/smoke_sam2_cp09.py \
  --checkpoint-path external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --model-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda:0 \
  --sample-fps 2 \
  --videos-dir "$HOME/medical_evaluation/videos" \
  --data-dir "$HOME/medical_evaluation/data"
```

Expected: no 175 GiB allocation attempt; the script prints paths for `summary.json` and the overlay directory. Inspect first, middle, and last overlays before accepting CP09 as visually correct.
