# Rubber Dam Video Evaluation Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local single-user web Demo that aligns rubber-dam procedure videos to 11 checkpoints, extracts SAM-backed visual evidence, computes deterministic scores, and uses a local Qwen3-VL service for constrained Chinese feedback.

**Architecture:** A FastAPI application persists jobs and annotations locally, runs a single queued analysis pipeline, and serves lightweight HTML/JavaScript pages. The pipeline separates step localization, SAM2.1/SAM3.1 segmentation, feature extraction, checkpoint-specific Judges, scoring, and VLM commentary behind typed interfaces so every layer can be tested with fakes before GPU integration.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, Jinja2, vanilla JavaScript/CSS, OpenCV/FFmpeg, NumPy/SciPy, PyYAML, httpx, pytest; optional GPU integrations for SAM3.1/SAM2.1, OpenCLIP, and a local vLLM-served Qwen3-VL-4B-Instruct.

---

## Execution checkpoints

- Checkpoint A after Task 6: the website can import the rubric, upload/select videos, edit annotations, run a fake job, and render a score report without a GPU.
- Checkpoint B after Task 11: real step localization, SAM evidence, and Checkpoints 9/11 run end-to-end on the successful video.
- Checkpoint C after Task 15: all 11 Judges, local Qwen feedback, three-video regression, and server documentation are complete.

## Planned file structure

```text
pyproject.toml                         Python package and test configuration
.gitignore                             Generated data, weights, uploads, caches
src/medical_evaluation/
  app.py                               FastAPI application factory and lifespan
  settings.py                          Environment-driven paths and model settings
  domain.py                            Shared typed domain models and enums
  rubric.py                            Excel/YAML rubric import and validation
  annotations.py                       Segment/prompt annotation persistence
  storage.py                           Safe local paths, atomic JSON, job artifacts
  jobs.py                              Persistent single-worker job queue
  scoring.py                           Equal-weight scoring and review ranges
  reporting.py                         Report assembly and audit metadata
  video.py                             Probe, decode, sparse/dense frame extraction
  pipeline.py                          Analysis orchestration only
  step_localizer.py                    Ordered alignment and boundary confidence
  segmentation/base.py                Backend-neutral segmentation protocol
  segmentation/sam2_backend.py         SAM2.1 adapter
  segmentation/sam3_backend.py         SAM3.1 adapter
  features/geometry.py                 Mask geometry and contact/coverage features
  features/motion.py                   Trajectory and state-transition features
  features/appearance.py               Color and reference-embedding features
  judges/base.py                       Judge protocol and shared evidence helpers
  judges/cp01_cp03.py                  Checkpoints 1–3
  judges/cp04_cp06.py                  Checkpoints 4–6
  judges/cp07_cp08.py                  Checkpoints 7–8
  judges/cp09_cp11.py                  Checkpoints 9–11
  vlm/client.py                        Local OpenAI-compatible Qwen client
  vlm/schemas.py                       Strict request/response schemas
  web/routes.py                        Page and JSON API routes
  web/templates/*.html                 Home, annotation, progress, report pages
  web/static/app.js                    Upload, polling, timeline, report interactions
  web/static/styles.css                Local offline styles
config/rubric.yaml                     Reviewed 11-checkpoint rubric
config/models.example.yaml             Model/device example without machine secrets
data/annotations/*.json                Human labels and prompts; ignored by default
scripts/import_rubric.py                One-time Excel-to-YAML conversion
scripts/run_server.py                   Local/server launch entry point
scripts/validate_demo.py                Three-video regression summary
tests/                                 Unit, API, integration, and golden tests
```

## Task 1: Python project skeleton and typed domain model

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/medical_evaluation/__init__.py`
- Create: `src/medical_evaluation/settings.py`
- Create: `src/medical_evaluation/domain.py`
- Test: `tests/test_domain.py`

- [ ] **Step 1: Write failing domain-model tests**

```python
from pydantic import ValidationError
from medical_evaluation.domain import CheckpointStatus, TimeRange


def test_time_range_rejects_reverse_bounds() -> None:
    try:
        TimeRange(start_sec=4.0, end_sec=3.0)
    except ValidationError:
        return
    raise AssertionError("reverse time range must fail")


def test_checkpoint_status_has_four_explicit_states() -> None:
    assert {item.value for item in CheckpointStatus} == {
        "correct", "incorrect", "incomplete", "needs_review"
    }
```

- [ ] **Step 2: Run the test and verify the package is missing**

Run: `python -m pytest tests/test_domain.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'medical_evaluation'`.

- [ ] **Step 3: Add package metadata, settings, and domain types**

```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "medical-evaluation"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  "fastapi>=0.116,<1",
  "httpx>=0.28,<1",
  "jinja2>=3.1,<4",
  "numpy>=2.1,<3",
  "opencv-python-headless>=4.11,<5",
  "openpyxl>=3.1,<4",
  "pydantic>=2.11,<3",
  "pydantic-settings>=2.9,<3",
  "python-multipart>=0.0.20,<1",
  "PyYAML>=6.0,<7",
  "scipy>=1.15,<2",
  "uvicorn[standard]>=0.34,<1",
]

[project.optional-dependencies]
dev = ["pytest>=8.3,<9", "pytest-asyncio>=0.26,<1", "ruff>=0.11,<1"]
vision = ["open-clip-torch>=2.32,<3"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```python
# src/medical_evaluation/domain.py
from enum import StrEnum
from pydantic import BaseModel, Field, model_validator


class CheckpointStatus(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    INCOMPLETE = "incomplete"
    NEEDS_REVIEW = "needs_review"


class TimeRange(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "TimeRange":
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        return self
```

`settings.py` must define a `Settings(BaseSettings)` with `project_root`, `data_dir`, `model_config_path`, `sam_backend`, `sam_device`, `vlm_base_url`, and `vlm_model`. Resolve all relative paths beneath `project_root`.

Add `.gitignore` entries for `.venv/`, `.pytest_cache/`, `.ruff_cache/`, `.superpowers/`, `data/uploads/`, `data/runs/`, `data/cache/`, `data/models/`, and model files `*.pt`, `*.pth`, `*.safetensors`.

- [ ] **Step 4: Install development dependencies and run tests**

Run: `python -m pip install -e ".[dev]" && python -m pytest tests/test_domain.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit the skeleton**

```bash
git add pyproject.toml .gitignore src/medical_evaluation tests/test_domain.py
git commit -m "chore: initialize evaluation application"
```

## Task 2: Rubric import and reviewed configuration

**Files:**
- Create: `src/medical_evaluation/rubric.py`
- Create: `scripts/import_rubric.py`
- Create: `config/rubric.yaml`
- Test: `tests/test_rubric.py`

- [ ] **Step 1: Write failing rubric tests**

```python
from pathlib import Path
from medical_evaluation.rubric import Rubric, load_rubric


def test_reviewed_rubric_has_11_equal_weight_checkpoints() -> None:
    rubric = load_rubric(Path("config/rubric.yaml"))
    assert isinstance(rubric, Rubric)
    assert [cp.id for cp in rubric.checkpoints] == [f"cp_{n:02d}" for n in range(1, 12)]
    assert abs(sum(cp.weight for cp in rubric.checkpoints) - 1.0) < 1e-9
    assert len({round(cp.weight, 12) for cp in rubric.checkpoints}) == 1


def test_each_checkpoint_defines_a_judge_and_criteria() -> None:
    rubric = load_rubric(Path("config/rubric.yaml"))
    assert all(cp.judge_type and cp.criteria for cp in rubric.checkpoints)
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tests/test_rubric.py -q`

Expected: FAIL because `medical_evaluation.rubric` does not exist.

- [ ] **Step 3: Implement strict rubric models and loader**

```python
# src/medical_evaluation/rubric.py
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, model_validator
from medical_evaluation.domain import TimeRange


class CheckpointRule(BaseModel):
    id: str = Field(pattern=r"^cp_\d{2}$")
    name: str
    criteria: list[str]
    weight: float = Field(gt=0, le=1)
    judge_type: str
    required_objects: list[str]
    reference_time: TimeRange
    thresholds: dict[str, float] = {}


class Rubric(BaseModel):
    version: str
    checkpoints: list[CheckpointRule]

    @model_validator(mode="after")
    def validate_total(self) -> "Rubric":
        if len(self.checkpoints) != 11:
            raise ValueError("rubric must contain 11 checkpoints")
        if abs(sum(item.weight for item in self.checkpoints) - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1")
        return self


def load_rubric(path: Path) -> Rubric:
    return Rubric.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
```

Implement `scripts/import_rubric.py` with `openpyxl.load_workbook(..., read_only=True, data_only=True)` solely for one-time source extraction. It must write YAML only to the explicit `--output` path, never modify the source workbook, convert `H:MM:SS-H:MM:SS` to seconds, assign IDs `cp_01`–`cp_11`, split numbered criteria into a list, and set each weight to `1 / 11` using sufficient decimal precision. Manually review `config/rubric.yaml` against all 11 rows after generation.

- [ ] **Step 4: Run import verification and tests**

Run: `python scripts/import_rubric.py "橡皮障隔离术操作解析.xlsx" --output config/rubric.generated.yaml && python -m pytest tests/test_rubric.py -q`

Expected: importer reports `11 checkpoints`; tests PASS. Compare `config/rubric.generated.yaml` with reviewed `config/rubric.yaml`, then remove the generated comparison file before commit.

- [ ] **Step 5: Commit the rubric layer**

```bash
git add src/medical_evaluation/rubric.py scripts/import_rubric.py config/rubric.yaml tests/test_rubric.py
git commit -m "feat: add reviewed checkpoint rubric"
```

## Task 3: Annotation schemas and atomic local persistence

**Files:**
- Create: `src/medical_evaluation/annotations.py`
- Create: `src/medical_evaluation/storage.py`
- Test: `tests/test_annotations.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write failing annotation and path-safety tests**

```python
from pathlib import Path
import pytest
from medical_evaluation.annotations import AnnotationStore, SegmentAnnotation
from medical_evaluation.domain import CheckpointStatus, TimeRange
from medical_evaluation.storage import safe_child


def test_annotation_store_round_trips(tmp_path: Path) -> None:
    store = AnnotationStore(tmp_path)
    item = SegmentAnnotation(
        checkpoint_id="cp_01",
        time_range=TimeRange(start_sec=0, end_sec=14),
        label=CheckpointStatus.CORRECT,
        reason="点位正确",
    )
    store.save_segments("success", [item], actor="tangmin")
    assert store.load_segments("success").steps == [item]


def test_safe_child_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside storage root"):
        safe_child(tmp_path, "../escape.json")
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_annotations.py tests/test_storage.py -q`

Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement annotation models, audit history, and atomic JSON**

```python
# src/medical_evaluation/storage.py
from pathlib import Path
import json, os, tempfile


def safe_child(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError("path resolves outside storage root")
    return candidate


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
```

`annotations.py` must define `SegmentAnnotation`, `PointPrompt`, `BoxPrompt`, `MaskPrompt`, `VideoAnnotations`, and `AuditEntry`. `AnnotationStore.save_segments()` must append an audit entry containing UTC timestamp, actor, prior SHA-256 digest, and new digest before atomically replacing the current JSON. Reject duplicate checkpoint IDs and overlapping step ranges that violate fixed order.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_annotations.py tests/test_storage.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit annotation persistence**

```bash
git add src/medical_evaluation/annotations.py src/medical_evaluation/storage.py tests/test_annotations.py tests/test_storage.py
git commit -m "feat: persist audited video annotations"
```

## Task 4: Deterministic scoring and report domain

**Files:**
- Create: `src/medical_evaluation/scoring.py`
- Create: `src/medical_evaluation/reporting.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write failing score tests**

```python
from medical_evaluation.domain import CheckpointStatus
from medical_evaluation.scoring import aggregate_equal_weight_score


def test_known_results_produce_final_score() -> None:
    statuses = [CheckpointStatus.CORRECT] * 8 + [CheckpointStatus.INCORRECT] * 3
    result = aggregate_equal_weight_score(statuses)
    assert result.final_score == 72.72727272727273
    assert result.minimum_score == result.maximum_score == result.final_score


def test_review_item_produces_range_and_no_final_score() -> None:
    statuses = [CheckpointStatus.CORRECT] * 8 + [CheckpointStatus.INCORRECT] * 2 + [CheckpointStatus.NEEDS_REVIEW]
    result = aggregate_equal_weight_score(statuses)
    assert result.final_score is None
    assert result.minimum_score == 72.72727272727273
    assert result.maximum_score == 81.81818181818181
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_scoring.py -q`

Expected: FAIL because `aggregate_equal_weight_score` is missing.

- [ ] **Step 3: Implement high-precision aggregation and report models**

```python
# src/medical_evaluation/scoring.py
from pydantic import BaseModel
from medical_evaluation.domain import CheckpointStatus


class ScoreSummary(BaseModel):
    final_score: float | None
    minimum_score: float
    maximum_score: float


def aggregate_equal_weight_score(statuses: list[CheckpointStatus]) -> ScoreSummary:
    if len(statuses) != 11:
        raise ValueError("exactly 11 statuses are required")
    unit = 100.0 / 11.0
    correct = sum(value is CheckpointStatus.CORRECT for value in statuses)
    reviews = sum(value is CheckpointStatus.NEEDS_REVIEW for value in statuses)
    minimum = correct * unit
    maximum = (correct + reviews) * unit
    return ScoreSummary(
        final_score=None if reviews else minimum,
        minimum_score=minimum,
        maximum_score=maximum,
    )
```

`reporting.py` must define `EvidenceItem`, `CheckpointResult`, `RunAudit`, and `EvaluationReport`. Enforce exactly 11 unique checkpoint results and derive summary scores from statuses rather than accepting caller-supplied totals.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_scoring.py -q`

Expected: PASS.

- [ ] **Step 5: Commit scoring**

```bash
git add src/medical_evaluation/scoring.py src/medical_evaluation/reporting.py tests/test_scoring.py
git commit -m "feat: add auditable equal-weight scoring"
```

## Task 5: Video probing, validation, and frame extraction

**Files:**
- Create: `src/medical_evaluation/video.py`
- Create: `tests/fixtures/make_test_video.py`
- Test: `tests/test_video.py`

- [ ] **Step 1: Write failing video tests using a generated fixture**

```python
from pathlib import Path
from medical_evaluation.video import probe_video, sample_frames
from tests.fixtures.make_test_video import make_test_video


def test_probe_and_sparse_sampling(tmp_path: Path) -> None:
    path = make_test_video(tmp_path / "sample.mp4", fps=10, seconds=2, size=(160, 120))
    meta = probe_video(path)
    assert meta.duration_sec == 2.0
    assert (meta.width, meta.height, round(meta.fps)) == (160, 120, 10)
    frames = list(sample_frames(path, start_sec=0, end_sec=2, sample_fps=2))
    assert len(frames) == 4
    assert [round(item.time_sec, 1) for item in frames] == [0.0, 0.5, 1.0, 1.5]
```

- [ ] **Step 2: Run the test and verify failure**

Run: `python -m pytest tests/test_video.py -q`

Expected: FAIL because video utilities are missing.

- [ ] **Step 3: Implement deterministic metadata and timestamp-based sampling**

```python
# src/medical_evaluation/video.py
from pathlib import Path
import cv2
from pydantic import BaseModel


class VideoMetadata(BaseModel):
    path: Path
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int


class SampledFrame(BaseModel):
    time_sec: float
    frame_index: int
    image_bgr: object


def probe_video(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count <= 0:
        raise ValueError("video has invalid fps or frame count")
    return VideoMetadata(path=path, duration_sec=count / fps, fps=fps, frame_count=count,
                         width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
```

Implement `sample_frames()` by seeking to exact timestamp-derived frame indices, returning copies, and releasing the capture in `finally`. Reject ranges outside video duration, sample rates ≤0, files larger than the configured limit, and unsupported extensions before copying uploads.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_video.py -q`

Expected: PASS.

- [ ] **Step 5: Commit video utilities**

```bash
git add src/medical_evaluation/video.py tests/fixtures/make_test_video.py tests/test_video.py
git commit -m "feat: validate videos and sample frames"
```

## Task 6: FastAPI shell, preset/upload flow, and persistent job queue

**Files:**
- Create: `src/medical_evaluation/jobs.py`
- Create: `src/medical_evaluation/app.py`
- Create: `src/medical_evaluation/web/routes.py`
- Create: `src/medical_evaluation/web/templates/index.html`
- Create: `src/medical_evaluation/web/templates/job.html`
- Create: `src/medical_evaluation/web/static/styles.css`
- Create: `src/medical_evaluation/web/static/app.js`
- Create: `scripts/run_server.py`
- Test: `tests/test_web_jobs.py`

- [ ] **Step 1: Write failing upload and queue API tests**

```python
from fastapi.testclient import TestClient
from medical_evaluation.app import create_app


def test_home_lists_three_presets(test_settings) -> None:
    client = TestClient(create_app(test_settings))
    response = client.get("/")
    assert response.status_code == 200
    assert "橡皮障完整.mp4" in response.text


def test_create_preset_job_returns_pollable_id(test_settings) -> None:
    client = TestClient(create_app(test_settings))
    response = client.post("/api/jobs", json={"preset_id": "success"})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] in {"queued", "running", "complete"}
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tests/test_web_jobs.py -q`

Expected: FAIL because the application factory is missing.

- [ ] **Step 3: Implement a single-worker persistent queue and web shell**

```python
# src/medical_evaluation/jobs.py
class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class JobRecord(BaseModel):
    id: str
    video_path: str
    status: JobStatus
    progress: float = Field(ge=0, le=1)
    current_checkpoint: str | None = None
    error: str | None = None
```

Implement `JobManager` with one `asyncio.Queue`, a lifespan-started worker, atomic JSON snapshots, and dependency-injected `pipeline(job, update_progress)`. On process restart, convert stale `running` jobs back to `queued`. `POST /api/jobs` accepts exactly one of `preset_id` or `video` upload, stores uploaded files under a generated UUID name, and never uses the client filename as a path. `GET /api/jobs/{id}` returns JSON; `GET /jobs/{id}` serves polling UI.

The initial injected pipeline must be a fake that writes 11 `needs_review` results so Checkpoint A is usable before GPU work.

- [ ] **Step 4: Run API tests and launch smoke test**

Run: `python -m pytest tests/test_web_jobs.py -q && python scripts/run_server.py --help`

Expected: tests PASS; help lists `--host`, `--port`, and `--config`.

- [ ] **Step 5: Commit the web shell**

```bash
git add src/medical_evaluation/jobs.py src/medical_evaluation/app.py src/medical_evaluation/web scripts/run_server.py tests/test_web_jobs.py
git commit -m "feat: add local web job workflow"
```

## Task 7: Annotation API and timeline editor

**Files:**
- Modify: `src/medical_evaluation/web/routes.py`
- Create: `src/medical_evaluation/web/templates/annotate.html`
- Create: `src/medical_evaluation/web/static/annotate.js`
- Test: `tests/test_annotation_api.py`

- [ ] **Step 1: Write failing annotation API tests**

```python
def test_annotation_update_requires_all_11_ordered_steps(client, valid_steps_payload) -> None:
    short = {"actor": "tangmin", "steps": valid_steps_payload["steps"][:10]}
    assert client.put("/api/videos/success/segments", json=short).status_code == 422


def test_annotation_update_round_trips(client, valid_steps_payload) -> None:
    response = client.put("/api/videos/success/segments", json=valid_steps_payload)
    assert response.status_code == 200
    loaded = client.get("/api/videos/success/segments").json()
    assert loaded["steps"] == valid_steps_payload["steps"]
```

- [ ] **Step 2: Run tests and verify the endpoints are absent**

Run: `python -m pytest tests/test_annotation_api.py -q`

Expected: FAIL with 404 responses.

- [ ] **Step 3: Add validated endpoints and a dependency-free timeline editor**

Add `GET/PUT /api/videos/{video_id}/segments`, `GET/PUT /api/videos/{video_id}/prompts`, and `GET /annotate/{video_id}`. The editor must render 11 colored blocks proportional to duration, use pointer events to drag adjacent boundaries, prevent crossing, expose label/reason fields, and submit all steps in one atomic request. Frame clicks must store normalized point coordinates; box drag must store normalized box coordinates. The server must validate video ID against known presets/uploads and actor as a non-empty string.

```javascript
// core boundary rule in annotate.js
function clampBoundary(value, previousEnd, nextStart) {
  const epsilon = 0.05;
  return Math.max(previousEnd + epsilon, Math.min(value, nextStart - epsilon));
}
```

- [ ] **Step 4: Run API tests and manually exercise the editor**

Run: `python -m pytest tests/test_annotation_api.py -q`

Expected: PASS. Manual smoke: dragging a boundary updates both adjacent segments; reload preserves it; invalid overlap receives HTTP 422 and leaves prior JSON intact.

- [ ] **Step 5: Commit annotation UI**

```bash
git add src/medical_evaluation/web tests/test_annotation_api.py
git commit -m "feat: add audited step annotation editor"
```

## Task 8: Fixed-order dynamic step localization

**Files:**
- Create: `src/medical_evaluation/step_localizer.py`
- Test: `tests/test_step_localizer.py`

- [ ] **Step 1: Write failing synthetic alignment tests**

```python
import numpy as np
from medical_evaluation.step_localizer import align_ordered_steps


def test_alignment_stretches_steps_without_reordering() -> None:
    prototypes = np.eye(3, dtype=np.float32)
    sequence = np.stack([prototypes[0], prototypes[0], prototypes[1], prototypes[2], prototypes[2]])
    result = align_ordered_steps(sequence, prototypes, frame_times=np.arange(5.0), skip_penalty=1.5)
    assert [item.step_index for item in result.segments] == [0, 1, 2]
    assert result.segments[0].end_sec <= result.segments[1].start_sec


def test_alignment_can_mark_a_missing_step() -> None:
    prototypes = np.eye(3, dtype=np.float32)
    sequence = np.stack([prototypes[0], prototypes[0], prototypes[2], prototypes[2]])
    result = align_ordered_steps(sequence, prototypes, frame_times=np.arange(4.0), skip_penalty=0.2)
    assert result.segments[1].missing is True


def test_event_anchor_refines_boundary_without_changing_raw_boundary() -> None:
    segment = localized_segment(start_sec=10.0, end_sec=20.0)
    refined = refine_with_anchors(segment, [event_anchor("floss_appears", 12.5, confidence=0.9)])
    assert refined.raw_start_sec == 10.0
    assert refined.start_sec == 12.5
    assert refined.anchor_evidence[0].name == "floss_appears"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_step_localizer.py -q`

Expected: FAIL because the localizer is missing.

- [ ] **Step 3: Implement monotonic dynamic programming and confidence**

```python
def cosine_cost(sequence: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    seq = sequence / np.clip(np.linalg.norm(sequence, axis=1, keepdims=True), 1e-8, None)
    proto = prototypes / np.clip(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8, None)
    return 1.0 - seq @ proto.T
```

Implement Viterbi-style dynamic programming with transitions `stay`, `advance`, and `skip`; disallow backward transitions. Convert the optimal state path into 11 `LocalizedSegment` records, including `missing`, raw/refined bounds, `mean_cost`, and confidence `clip(1 - mean_cost / confidence_scale, 0, 1)`. Define an `EmbeddingEncoder` protocol and an `OpenClipEncoder` optional adapter; tests use deterministic NumPy embeddings. Implement `refine_with_anchors()` for typed events such as punch contact, clamp appearance, mouth entry, floss appearance, and frame expansion. Preserve raw bounds and append every accepted/rejected anchor with confidence and reason.

- [ ] **Step 4: Run localizer tests**

Run: `python -m pytest tests/test_step_localizer.py -q`

Expected: PASS for stretch, skip, and no-backtracking cases.

- [ ] **Step 5: Commit dynamic alignment**

```bash
git add src/medical_evaluation/step_localizer.py tests/test_step_localizer.py
git commit -m "feat: align variable-duration procedure steps"
```

## Task 9: Segmentation protocol and SAM2.1/SAM3.1 adapters

**Files:**
- Create: `src/medical_evaluation/segmentation/base.py`
- Create: `src/medical_evaluation/segmentation/sam2_backend.py`
- Create: `src/medical_evaluation/segmentation/sam3_backend.py`
- Create: `config/models.example.yaml`
- Test: `tests/test_segmentation_contract.py`

- [ ] **Step 1: Write backend-contract tests with a fake predictor**

```python
import numpy as np
from medical_evaluation.segmentation.base import SegmentationPrompt, normalize_masks


def test_masks_are_boolean_and_keyed_by_object() -> None:
    raw = {"frame": 3, "objects": {"rubber_dam": np.array([[0.1, 0.9]])}}
    result = normalize_masks(raw, threshold=0.5, frame_time_sec=1.5)
    assert result.frame_time_sec == 1.5
    assert result.masks["rubber_dam"].dtype == np.bool_


def test_prompt_coordinates_are_normalized() -> None:
    prompt = SegmentationPrompt(object_id="frame", kind="point", coordinates=[0.5, 0.25])
    assert prompt.coordinates == [0.5, 0.25]
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_segmentation_contract.py -q`

Expected: FAIL because the segmentation package does not exist.

- [ ] **Step 3: Implement a backend-neutral protocol and lazy adapters**

```python
class VideoSegmenter(Protocol):
    @property
    def model_version(self) -> str: ...

    def track(
        self,
        video_path: Path,
        time_range: TimeRange,
        prompts: list[SegmentationPrompt],
        sample_fps: float,
    ) -> Iterator[FrameMasks]: ...
```

Both adapters must import their third-party library inside the constructor so CPU unit tests do not require SAM. Convert normalized prompts to pixels at runtime, preserve object IDs, return boolean masks, and expose model/checkpoint digest. `sam3_backend.py` uses text prompts when configured and point/box refinements when present. `sam2_backend.py` rejects text-only prompts with an actionable validation error. Neither adapter owns scoring logic.

`config/models.example.yaml` must show `backend`, `checkpoint`, `device`, `analysis_width`, `sparse_fps`, `dense_fps`, local `vlm_base_url`, and `vlm_model` with safe example values.

- [ ] **Step 4: Run contract tests and a server-only import smoke test**

Run: `python -m pytest tests/test_segmentation_contract.py -q && python -c "from medical_evaluation.segmentation.sam3_backend import Sam3Backend"`

Expected: tests PASS and import succeeds without loading weights.

- [ ] **Step 5: Commit segmentation adapters**

```bash
git add src/medical_evaluation/segmentation config/models.example.yaml tests/test_segmentation_contract.py
git commit -m "feat: add swappable SAM video backends"
```

## Task 10: Mask feature extraction and evidence overlays

**Files:**
- Create: `src/medical_evaluation/features/geometry.py`
- Create: `src/medical_evaluation/features/motion.py`
- Create: `src/medical_evaluation/features/appearance.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing numerical feature tests**

```python
import numpy as np
from medical_evaluation.features.geometry import centroid_xy, intersection_ratio
from medical_evaluation.features.motion import displacement_sequence


def test_centroid_and_intersection_ratio() -> None:
    a = np.zeros((10, 10), bool); a[2:6, 2:6] = True
    b = np.zeros((10, 10), bool); b[4:8, 4:8] = True
    assert centroid_xy(a) == (3.5, 3.5)
    assert intersection_ratio(a, b) == 4 / 16


def test_displacement_preserves_time_order() -> None:
    assert displacement_sequence([(0.0, 0.0), (1.0, 2.0), (1.0, 5.0)]) == [(1.0, 2.0), (0.0, 3.0)]
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_features.py -q`

Expected: FAIL because feature modules are missing.

- [ ] **Step 3: Implement pure feature functions and overlay writer**

Implement functions for area ratio, normalized centroid, boundary distance/contact approximation, IoU, overlap relative to reference region, visible ratio, HSV green/dark ratio, blur score, centroid trajectory, velocity, direction changes, and reference cosine similarity. Empty masks return explicit missing values rather than zero geometry. All functions accept NumPy arrays and contain no model calls.

```python
def centroid_xy(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())
```

Add `write_overlay(frame_bgr, masks, output_path)` that blends stable per-object colors, draws object labels, and writes only beneath the job evidence directory resolved with `safe_child`.

- [ ] **Step 4: Run numerical tests**

Run: `python -m pytest tests/test_features.py -q`

Expected: PASS, including empty-mask and zero-area cases.

- [ ] **Step 5: Commit feature extraction**

```bash
git add src/medical_evaluation/features tests/test_features.py
git commit -m "feat: extract auditable mask features"
```

## Task 11: Vertical slice with Checkpoints 9 and 11

**Files:**
- Create: `src/medical_evaluation/judges/base.py`
- Create: `src/medical_evaluation/judges/cp09_cp11.py`
- Create: `src/medical_evaluation/pipeline.py`
- Modify: `src/medical_evaluation/app.py`
- Test: `tests/test_judges_cp09_cp11.py`
- Test: `tests/test_pipeline_vertical.py`

- [ ] **Step 1: Write failing Judge tests**

```python
from medical_evaluation.judges.cp09_cp11 import judge_cp09, judge_cp11


def test_cp09_marks_centered_frame_correct() -> None:
    result = judge_cp09({"frame_center_offset": 0.03}, {"max_center_offset": 0.08})
    assert result.status.value == "correct"
    assert result.matched_rules == ["frame_centered"]


def test_cp11_requests_review_when_face_region_is_missing() -> None:
    result = judge_cp11({"mouth_nose_visible": False}, {"max_face_overlap": 0.02, "min_frame_coverage": 0.85})
    assert result.status.value == "needs_review"


def test_pipeline_retries_oom_once_with_degraded_sampling(fake_pipeline) -> None:
    fake_pipeline.segmenter.fail_once_with_oom = True
    report = fake_pipeline.run(make_job())
    assert report.audit.degradations == ["cuda_oom:dense_fps=1.0,analysis_width=960"]


def test_low_alignment_confidence_pauses_checkpoint(fake_pipeline) -> None:
    fake_pipeline.localizer.confidence = 0.2
    report = fake_pipeline.run(make_job())
    assert report.checkpoints[0].status.value == "needs_review"
    assert report.checkpoints[0].reason_code == "low_step_alignment_confidence"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_judges_cp09_cp11.py tests/test_pipeline_vertical.py -q`

Expected: FAIL because Judges and pipeline are missing.

- [ ] **Step 3: Implement Judge protocol, two Judges, and orchestration**

```python
class JudgeDecision(BaseModel):
    checkpoint_id: str
    status: CheckpointStatus
    confidence: float = Field(ge=0, le=1)
    matched_rules: list[str]
    features: dict[str, float | bool | None]
    reason_code: str
```

`judge_cp09` compares the frame-center offset with configured maximum. `judge_cp11` requires visible face landmarks, rubber-dam/face overlap below threshold, and dam/frame coverage above threshold. Missing required evidence returns `needs_review`; threshold violation returns `incorrect`; all conditions passing returns `correct`.

`AnalysisPipeline.run(job)` must load rubric and annotations, localize or use confirmed step ranges, invoke only registered Judges, generate evidence overlays, create temporary `needs_review` results for Judges not yet registered, aggregate scoring, persist `report.json`, and update progress after every checkpoint. Low alignment confidence must pause that checkpoint with `low_step_alignment_confidence`. Catch CUDA out-of-memory once, release cached tensors, reduce to configured fallback sampling/resolution, retry the affected checkpoint, and record the exact degradation in `RunAudit`; a second OOM fails the job. Replace the fake pipeline injection in `app.py` with a configurable real/fake factory.

- [ ] **Step 4: Run vertical tests and one successful-video smoke run**

Run: `python -m pytest tests/test_judges_cp09_cp11.py tests/test_pipeline_vertical.py -q`

Expected: PASS. On the GPU server, run only Checkpoints 9 and 11 against the successful video and verify both reports include an overlay, feature values, and matched rules; record actual runtime in the job audit.

- [ ] **Step 5: Commit the first vertical slice**

```bash
git add src/medical_evaluation/judges src/medical_evaluation/pipeline.py src/medical_evaluation/app.py tests/test_judges_cp09_cp11.py tests/test_pipeline_vertical.py
git commit -m "feat: evaluate frame placement checkpoints"
```

## Task 12: Implement Checkpoints 1–8 and 10 as isolated Judges

**Files:**
- Create: `src/medical_evaluation/judges/cp01_cp03.py`
- Create: `src/medical_evaluation/judges/cp04_cp06.py`
- Create: `src/medical_evaluation/judges/cp07_cp08.py`
- Modify: `src/medical_evaluation/judges/cp09_cp11.py`
- Modify: `src/medical_evaluation/pipeline.py`
- Test: `tests/test_judges_all.py`

- [ ] **Step 1: Write one failing positive, negative, and missing-evidence case per Judge**

```python
@pytest.mark.parametrize(
    ("judge", "features", "expected"),
    [
        (judge_cp01, {"mark_reference_distance": 0.02}, "correct"),
        (judge_cp03, {"punch_contact": True, "released_from_tip": False}, "incorrect"),
        (judge_cp06, {"opening_increase": None, "spring_backward": None}, "needs_review"),
        (judge_cp10, {"mesial_crossing": True, "distal_crossing": False}, "incorrect"),
    ],
)
def test_checkpoint_decisions(judge, features, expected, thresholds_by_checkpoint) -> None:
    assert judge(features, thresholds_by_checkpoint[judge.checkpoint_id]).status.value == expected
```

Add explicit tests for all 11 IDs, including CP02 residue absent/present branches, CP04 clamp similarity, CP05 layer order, CP07 partial contact visibility, and CP08 blunt-tool/wing visibility/green-hole criteria.

- [ ] **Step 2: Run tests and verify unimplemented Judges fail**

Run: `python -m pytest tests/test_judges_all.py -q`

Expected: FAIL on imports or missing Judge registrations.

- [ ] **Step 3: Implement each Judge as a pure decision function**

Use the same decision priority in every file:

```python
def decide(required: dict[str, object], failures: list[bool], checkpoint_id: str) -> JudgeDecision:
    if any(value is None for value in required.values()):
        return review(checkpoint_id, reason_code="missing_required_evidence")
    if any(failures):
        return incorrect(checkpoint_id, reason_code="criterion_failed")
    return correct(checkpoint_id)
```

Implement CP01 reference-point distance; CP02 selected-hole index and conditional residue-cleaning sequence; CP03 punch-contact/release/reverse-motion state; CP04 reference similarity plus optional semantic confirmation; CP05 wing/bow layer ordering; CP06 clamp opening and spring-back lock; CP07 target-tooth, visible contact stability, and distal bow; CP08 blunt tool, exposed wings, neck enclosure, and green wing-hole ratio; CP10 mesial and distal crossings. Put thresholds only in `config/rubric.yaml`, never as numeric literals inside Judges.

Register all Judges by checkpoint ID in `pipeline.py`. Each registered Judge must declare its required features so the pipeline extracts only necessary evidence.

- [ ] **Step 4: Run Judge and regression tests**

Run: `python -m pytest tests/test_judges_all.py tests/test_pipeline_vertical.py -q`

Expected: all Judge cases PASS; the prior CP09/CP11 vertical slice remains green.

- [ ] **Step 5: Commit all deterministic Judges**

```bash
git add src/medical_evaluation/judges src/medical_evaluation/pipeline.py config/rubric.yaml tests/test_judges_all.py
git commit -m "feat: add eleven checkpoint judges"
```

## Task 13: Local Qwen3-VL adapter with strict structured output

**Files:**
- Create: `src/medical_evaluation/vlm/schemas.py`
- Create: `src/medical_evaluation/vlm/client.py`
- Modify: `src/medical_evaluation/reporting.py`
- Test: `tests/test_vlm_client.py`

- [ ] **Step 1: Write failing retry and fallback tests with MockTransport**

```python
import httpx
from medical_evaluation.vlm.client import QwenVlmClient


def test_invalid_json_retries_once_then_returns_template() -> None:
    calls = 0
    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})
    client = QwenVlmClient("http://local/v1", "Qwen3-VL-4B-Instruct", transport=httpx.MockTransport(handler))
    result = client.review(make_review_request())
    assert calls == 2
    assert result.source == "template_fallback"
    assert result.score_override is None
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_vlm_client.py -q`

Expected: FAIL because the VLM client is missing.

- [ ] **Step 3: Implement schema-constrained local calls**

```python
class VlmReview(BaseModel):
    evidence_supported: bool
    semantic_status: Literal["supports", "contradicts", "uncertain"]
    reason_zh: str
    suggestion_zh: str
    cited_evidence_indices: list[int]
    source: Literal["qwen", "template_fallback"] = "qwen"
    score_override: None = None
```

Send an OpenAI-compatible request only to the configured local base URL. Include checkpoint text, deterministic decision, selected evidence images, and feature JSON. The system prompt must state: use only supplied evidence, cite evidence indices, return JSON only, never assign or change a score. Validate response with `VlmReview`; retry once with a JSON-repair instruction; then return a deterministic Chinese template based on `reason_code`. Enforce request timeout and maximum image count. Log model name and response digest, not raw video.

- [ ] **Step 4: Run VLM tests including service-down behavior**

Run: `python -m pytest tests/test_vlm_client.py -q`

Expected: PASS for valid response, invalid JSON retry, timeout fallback, and forbidden score field.

- [ ] **Step 5: Commit local VLM integration**

```bash
git add src/medical_evaluation/vlm src/medical_evaluation/reporting.py tests/test_vlm_client.py
git commit -m "feat: add constrained local Qwen feedback"
```

## Task 14: Report UI, review resolution, and audit display

**Files:**
- Modify: `src/medical_evaluation/web/routes.py`
- Create: `src/medical_evaluation/web/templates/report.html`
- Create: `src/medical_evaluation/web/static/report.js`
- Test: `tests/test_report_api.py`

- [ ] **Step 1: Write failing report and review tests**

```python
def test_report_page_contains_11_checkpoint_rows(client, completed_report) -> None:
    response = client.get(f"/reports/{completed_report.job_id}")
    assert response.status_code == 200
    assert response.text.count('data-checkpoint-id="cp_') == 11


def test_resolving_review_recomputes_final_score(client, review_report) -> None:
    response = client.put(
        f"/api/reports/{review_report.job_id}/cp_07/review",
        json={"actor": "tangmin", "status": "incorrect", "reason": "人工确认接触不足"},
    )
    assert response.status_code == 200
    assert response.json()["summary"]["final_score"] is not None


def test_single_checkpoint_rerun_preserves_prior_audit(client, review_report) -> None:
    response = client.post(f"/api/jobs/{review_report.job_id}/checkpoints/cp_07/rerun")
    assert response.status_code == 202
    assert response.json()["checkpoint_id"] == "cp_07"


def test_delete_job_removes_only_derived_artifacts(client, completed_report) -> None:
    response = client.delete(f"/api/jobs/{completed_report.job_id}/artifacts")
    assert response.status_code == 204
    assert completed_report.source_video.exists()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_report_api.py -q`

Expected: FAIL because report routes/templates are missing.

- [ ] **Step 3: Implement report rendering and audited review resolution**

The report page must show total score or min–max interval, overall feedback, model/config/runtime audit, and exactly 11 checkpoint cards. Each card shows status, displayed score, actual interval, confidence, key evidence thumbnails, mask overlays, feature values, matched rules, reason, and suggestion. `needs_review` cards expose an administrator form for final status and reason. The API must append a review audit entry and recompute scores; it must never delete the automatic result. Add `POST /api/jobs/{job_id}/checkpoints/{checkpoint_id}/rerun` to queue only the selected interval while retaining the prior result in audit history. Add `DELETE /api/jobs/{job_id}/artifacts` to remove the run directory after verifying it is beneath `data/runs`; never delete the source preset/upload or annotation JSON.

```javascript
function scoreLabel(summary) {
  return summary.final_score === null
    ? `${summary.minimum_score.toFixed(1)}–${summary.maximum_score.toFixed(1)}（待复核）`
    : `${summary.final_score.toFixed(1)} / 100`;
}
```

- [ ] **Step 4: Run report tests and browser smoke test**

Run: `python -m pytest tests/test_report_api.py -q`

Expected: PASS. Manual smoke: evidence opens at full size; review changes only the selected checkpoint; the score interval becomes a final score when the last review is resolved.

- [ ] **Step 5: Commit report UI**

```bash
git add src/medical_evaluation/web tests/test_report_api.py
git commit -m "feat: present evidence-backed evaluation reports"
```

## Task 15: Three-video golden regression and server runbook

**Files:**
- Create: `data/annotations/success.json`
- Create: `data/annotations/failure_clamp_flew.json`
- Create: `data/annotations/failure_general.json`
- Create: `scripts/validate_demo.py`
- Create: `tests/test_demo_golden.py`
- Create: `docs/server-runbook.md`
- Create: `README.md`

- [ ] **Step 1: Complete human ground truth before asserting model accuracy**

Use the administrator annotation page to label all 33 video/checkpoint combinations with actual start/end, status, and reason. Add SAM point/box prompts to required success-video keyframes. Run a validation command that rejects any missing label, empty reason for a non-correct item, overlapping segment, absent prompt required by the rubric, or step outside video duration.

Run: `python scripts/validate_demo.py --annotations-only`

Expected: `3 videos, 33 checkpoint labels, annotation validation passed`.

- [ ] **Step 2: Write failing golden-report structural tests**

```python
@pytest.mark.parametrize("video_id", ["success", "failure_clamp_flew", "failure_general"])
def test_demo_report_is_complete(video_id, run_demo_with_fakes) -> None:
    report = run_demo_with_fakes(video_id)
    assert len(report.checkpoints) == 11
    assert all(item.evidence or item.status.value == "incomplete" for item in report.checkpoints)
    assert report.audit.rubric_version
    assert report.audit.model_versions
```

- [ ] **Step 3: Implement validation metrics without overstating generalization**

`scripts/validate_demo.py` must run each preset, compare automatic step intervals with human intervals using temporal IoU and boundary absolute error, compare checkpoint statuses with ground truth, emit a 4×4 confusion matrix, list every disagreement, and report per-video runtime/peak GPU memory. It must label the output `demo-set agreement`, never `test accuracy` or `generalization accuracy`.

```python
def temporal_iou(a: TimeRange, b: TimeRange) -> float:
    intersection = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    union = max(a.end_sec, b.end_sec) - min(a.start_sec, b.start_sec)
    return intersection / union if union else 0.0
```

- [ ] **Step 4: Write and verify the Ubuntu server runbook**

`docs/server-runbook.md` must include exact commands to create the Python environment, install the project, install the selected SAM backend from its official repository, record the installed commit and model SHA-256, download or point to local weights, start Qwen3-VL through vLLM, choose GPU devices through configuration, start FastAPI, run tests, execute the three-video validator, and diagnose CUDA OOM. It must note SAM3 uses the SAM License, SAM2.1 and Qwen3-VL use Apache-2.0, and videos remain local.

Run on the server:

```bash
python -m pytest -q
python scripts/validate_demo.py --annotations-only
python scripts/validate_demo.py --run-all --output data/runs/demo-validation.json
```

Expected: tests PASS; three reports are produced; validator prints all disagreements and resource measurements. Any mismatch remains documented rather than being hidden by changing ground truth.

- [ ] **Step 5: Run final verification and commit the Demo**

Run:

```bash
python -m pytest -q
ruff check src tests scripts
git diff --check
git status --short
```

Expected: test suite PASS, Ruff reports no errors, diff check is clean, and only intentionally ignored runtime/model/video artifacts remain untracked.

Commit only code, reviewed configuration, documentation, and any annotation files the user explicitly approves for version control. Do not commit videos, weights, caches, uploads, or evidence images.

```bash
git add README.md docs/server-runbook.md scripts/validate_demo.py tests/test_demo_golden.py
git commit -m "test: validate three-video evaluation demo"
```

## Final manual acceptance

- [ ] Start the local Qwen3-VL service and the FastAPI application on the Ubuntu server.
- [ ] From the browser, run all three preset videos without using command-line shortcuts.
- [ ] Confirm every report has 11 rows, understandable Chinese feedback, evidence links, and audit metadata.
- [ ] Confirm a low-confidence time boundary can be adjusted and rerun.
- [ ] Confirm disabling Qwen changes wording/fallback source but not deterministic scores.
- [ ] Confirm a deliberately obscured required object produces `needs_review`, not a guessed result.
- [ ] Record the final demo-set agreement, runtime, peak memory, known failures, SAM backend/version, and Qwen model/version in the handoff notes.
