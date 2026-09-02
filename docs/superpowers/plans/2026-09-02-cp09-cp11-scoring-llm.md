# CP09/CP11 Scoring and LLM Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run only the available CP09/CP11 vision checks, calculate an explicitly provisional score over the checks actually evaluated, and attach local Qwen evidence commentary without allowing the model to alter deterministic decisions.

**Architecture:** Keep the eleven-row report contract and add an `included_in_provisional_score` flag to each row. A job-scoped CP09/CP11 extractor dispatches to the two existing extractors and raises a typed missing-input exception before invoking CP09 when annotations are incomplete. `AnalysisPipeline` catches only that typed exception, excludes the row from provisional scoring, and invokes the existing structured Qwen reviewer only for checks that actually ran.

**Tech Stack:** Python 3.11, Pydantic 2, FastAPI/Jinja2, httpx, PyYAML, SAM2, pytest, Ruff.

---

### Task 1: Represent provisional scoring without changing the eleven-item rubric

**Files:**
- Modify: `src/medical_evaluation/scoring.py`
- Modify: `src/medical_evaluation/reporting.py`
- Modify: `tests/test_scoring.py`

- [ ] **Step 1: Write failing tests for partial scoring**

Add tests which build eleven statuses with inclusion flags and assert the public summary contract:

```python
def test_two_evaluated_items_produce_provisional_score_only() -> None:
    statuses = [CheckpointStatus.NEEDS_REVIEW] * 11
    statuses[8] = CheckpointStatus.CORRECT
    statuses[10] = CheckpointStatus.INCORRECT
    included = [False] * 11
    included[8] = included[10] = True

    result = aggregate_evaluation_score(statuses, included)

    assert result.final_score is None
    assert result.evaluated_count == 2
    assert result.total_count == 11
    assert result.provisional_score == pytest.approx(50.0)
    assert result.provisional_minimum_score == pytest.approx(50.0)
    assert result.provisional_maximum_score == pytest.approx(50.0)


def test_one_evaluated_correct_item_is_provisional_100() -> None:
    statuses = [CheckpointStatus.NEEDS_REVIEW] * 11
    statuses[10] = CheckpointStatus.CORRECT
    included = [False] * 11
    included[10] = True

    result = aggregate_evaluation_score(statuses, included)

    assert result.evaluated_count == 1
    assert result.provisional_score == pytest.approx(100.0)


def test_evaluated_review_item_produces_provisional_range() -> None:
    statuses = [CheckpointStatus.NEEDS_REVIEW] * 11
    statuses[8] = CheckpointStatus.CORRECT
    statuses[10] = CheckpointStatus.NEEDS_REVIEW
    included = [False] * 11
    included[8] = included[10] = True

    result = aggregate_evaluation_score(statuses, included)

    assert result.provisional_score is None
    assert result.provisional_minimum_score == pytest.approx(50.0)
    assert result.provisional_maximum_score == pytest.approx(100.0)
```

Also update the report-model test so omitted flags default to `True`, preserving full-report behavior.

- [ ] **Step 2: Run the new scoring tests and confirm RED**

Run:

```powershell
python -m pytest tests/test_scoring.py -q
```

Expected: failures because `aggregate_evaluation_score`, inclusion flags, and provisional fields do not exist.

- [ ] **Step 3: Implement the minimal scoring model**

Extend `ScoreSummary` with:

```python
provisional_score: float | None
provisional_minimum_score: float
provisional_maximum_score: float
evaluated_count: int
total_count: int
```

Add `included_in_provisional_score: bool = True` to `CheckpointResult`. Implement `aggregate_evaluation_score(statuses, included)` so excluded rows do not affect provisional values; `final_score` is populated only when all eleven rows are included and none needs review. Make `EvaluationReport.summary` call it using each row's flag. Keep `aggregate_equal_weight_score` as a compatibility wrapper with eleven `True` flags.

- [ ] **Step 4: Run scoring tests and confirm GREEN**

Run:

```powershell
python -m pytest tests/test_scoring.py -q
```

Expected: all scoring tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/medical_evaluation/scoring.py src/medical_evaluation/reporting.py tests/test_scoring.py
git commit -m "feat: add provisional partial scoring"
```

### Task 2: Restrict the real pipeline to CP09 and CP11 with dynamic CP09 availability

**Files:**
- Create: `src/medical_evaluation/extractors/cp09_cp11.py`
- Modify: `src/medical_evaluation/pipeline.py`
- Modify: `tests/test_pipeline_vertical.py`
- Create: `tests/test_cp09_cp11_extractor.py`

- [ ] **Step 1: Write failing pipeline-boundary tests**

Replace the fake extractor assertions with an extractor that records calls. Assert that a complete job invokes exactly `cp_09` and `cp_11`, while the other nine rows have `automatic_evaluation_not_implemented` and `included_in_provisional_score is False`.

Add a typed-missing-input case:

```python
class MissingCp09Extractor(FakeExtractor):
    def extract(self, *args, **kwargs):
        checkpoint_id = args[1]
        if checkpoint_id == "cp_09":
            raise EvaluationInputMissing("cp_09 requires frame prompts and one oral box")
        return super().extract(*args, **kwargs)


def test_missing_cp09_input_excludes_cp09_but_still_runs_cp11(tmp_path: Path) -> None:
    report = make_pipeline(tmp_path, extractor=MissingCp09Extractor()).run(make_job(tmp_path))

    assert report.checkpoints[8].reason_code == "automatic_evaluation_input_missing"
    assert report.checkpoints[8].included_in_provisional_score is False
    assert report.checkpoints[10].included_in_provisional_score is True
    assert report.summary.evaluated_count == 1
```

- [ ] **Step 2: Run pipeline tests and confirm RED**

Run:

```powershell
python -m pytest tests/test_pipeline_vertical.py tests/test_cp09_cp11_extractor.py -q
```

Expected: failures because the enabled-checkpoint boundary, typed exception, and dispatcher do not exist.

- [ ] **Step 3: Implement the boundary and dispatcher**

Add this typed exception in `pipeline.py`:

```python
class EvaluationInputMissing(ValueError):
    pass
```

Give `AnalysisPipeline` an immutable default enabled set and support a job-scoped extractor factory while retaining the existing static extractor constructor for tests:

```python
enabled_checkpoint_ids: frozenset[str] = frozenset({"cp_09", "cp_11"})

class FeatureExtractorFactory(Protocol):
    def __call__(self, job: JobRecord) -> FeatureExtractor: ...
```

At the start of `run`, resolve one extractor with `self.extractor_factory(job)` when a factory was provided; otherwise use the injected static extractor. For non-enabled rows, immediately create an excluded review result. Catch only `EvaluationInputMissing` around extraction and create an excluded result with `automatic_evaluation_input_missing`. Successful extraction, including a Judge `needs_review`, remains included because the model actually ran.

Create `Cp09Cp11FeatureExtractor`, initialized with the two existing delegates plus the current `VideoAnnotations`. Before dispatching CP09, validate at least one in-range `rubber_dam_frame` point/box and exactly one in-range `oral_region` box; otherwise raise `EvaluationInputMissing`. Dispatch CP11 directly so its green-presence scan can classify videos with no final dam.

- [ ] **Step 4: Run boundary tests and confirm GREEN**

Run:

```powershell
python -m pytest tests/test_pipeline_vertical.py tests/test_cp09_cp11_extractor.py -q
```

Expected: all tests pass and the fake extractor call list contains no CP01–CP08 or CP10 calls.

- [ ] **Step 5: Commit**

```powershell
git add src/medical_evaluation/pipeline.py src/medical_evaluation/extractors/cp09_cp11.py tests/test_pipeline_vertical.py tests/test_cp09_cp11_extractor.py
git commit -m "feat: limit real evaluation to CP09 and CP11"
```

### Task 3: Attach structured Qwen commentary without mutating Judge output

**Files:**
- Modify: `src/medical_evaluation/vlm/client.py`
- Modify: `src/medical_evaluation/reporting.py`
- Modify: `src/medical_evaluation/pipeline.py`
- Modify: `tests/test_vlm_client.py`
- Modify: `tests/test_pipeline_vertical.py`

- [ ] **Step 1: Write failing commentary-integration tests**

Add a fake reviewer returning a `VlmReview`. Assert it receives only included CP09/CP11 rows, its result is stored as `ai_commentary`, and the original Judge `status`, `reason_code`, `matched_rules`, features, and score remain unchanged. Add a case where the reviewer raises an exception and assert the report still saves with a template fallback commentary.

```python
assert reviewer.requests[0].deterministic_status == "correct"
assert report.checkpoints[8].status is CheckpointStatus.CORRECT
assert report.checkpoints[8].ai_commentary.reason_zh == "证据支持支架居中。"
assert report.checkpoints[0].ai_commentary is None
```

Extend the client test so a missing evidence file is handled as a review failure rather than aborting the evaluation job.

- [ ] **Step 2: Run commentary tests and confirm RED**

Run:

```powershell
python -m pytest tests/test_vlm_client.py tests/test_pipeline_vertical.py -q
```

Expected: failures because report rows do not contain commentary and the pipeline does not invoke the reviewer.

- [ ] **Step 3: Implement commentary integration**

Add `ai_commentary: VlmReview | None = None` to `CheckpointResult`. Define a small `CommentaryProvider` protocol in `pipeline.py` with `review(VlmReviewRequest) -> VlmReview`, and accept an optional provider in `AnalysisPipeline`.

After deterministic conversion, build the request from the rubric rule, decision, features, and evidence overlay paths. Never copy any VLM field into `status`, `reason_code`, `matched_rules`, `features`, or score fields. Catch provider-level exceptions and call a public deterministic fallback function. Expand fallback templates for all current CP09 and CP11 reason codes. Include `OSError` in the Qwen client's retry/fallback handling.

Build `overall_feedback` from the fixed prefix “本报告仅自动评估 N/11 项，以下为阶段性结果，不是最终成绩。” plus the two stored commentary summaries. This aggregation performs no model scoring.

- [ ] **Step 4: Run commentary tests and confirm GREEN**

Run:

```powershell
python -m pytest tests/test_vlm_client.py tests/test_pipeline_vertical.py -q
```

Expected: all tests pass, including service-down and invalid-JSON fallback cases.

- [ ] **Step 5: Commit**

```powershell
git add src/medical_evaluation/vlm/client.py src/medical_evaluation/reporting.py src/medical_evaluation/pipeline.py tests/test_vlm_client.py tests/test_pipeline_vertical.py
git commit -m "feat: attach guarded Qwen commentary"
```

### Task 4: Build the real CP09/CP11 runtime from server configuration

**Files:**
- Create: `src/medical_evaluation/runtime.py`
- Modify: `src/medical_evaluation/settings.py`
- Modify: `src/medical_evaluation/app.py`
- Create: `tests/test_runtime.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write failing runtime-factory tests**

Use dependency injection for the SAM backend constructor so tests do not load CUDA. Assert fake mode never constructs SAM, real mode constructs an `AnalysisPipeline`, loads per-video annotations, creates job-scoped evidence paths, and installs `QwenVlmClient` from settings. Assert missing checkpoint files fail at startup with a clear `ValueError`.

- [ ] **Step 2: Run runtime tests and confirm RED**

Run:

```powershell
python -m pytest tests/test_runtime.py tests/test_app.py -q
```

Expected: failures because no production runtime factory exists.

- [ ] **Step 3: Implement settings and runtime factory**

Add settings fields with `MED_EVAL_` environment names:

```python
sam2_checkpoint_path: Path = Path("external/sam2/checkpoints/sam2.1_hiera_large.pt")
sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
sample_fps: float = 2.0
analysis_width: int = 1280
```

In `runtime.py`, construct one `Sam2Backend`, a `QwenVlmClient`, and the `FeatureExtractorFactory` accepted by `AnalysisPipeline`. For every job it loads `AnnotationStore.load_segments(job.video_id)`, creates `Cp09FeatureExtractor` and `Cp11FeatureExtractor` with `output_root / job.id`, and wraps them in `Cp09Cp11FeatureExtractor`. Validate the checkpoint file before initializing SAM.

Update `create_app`: fake mode remains unchanged; real mode uses an explicitly injected pipeline when supplied, otherwise calls the runtime factory. Do not silently fall back to fake mode.

- [ ] **Step 4: Run runtime tests and confirm GREEN**

Run:

```powershell
python -m pytest tests/test_runtime.py tests/test_app.py -q
```

Expected: all tests pass without importing or loading real CUDA weights.

- [ ] **Step 5: Commit**

```powershell
git add src/medical_evaluation/runtime.py src/medical_evaluation/settings.py src/medical_evaluation/app.py tests/test_runtime.py tests/test_app.py
git commit -m "feat: wire real CP09 CP11 runtime"
```

### Task 5: Display provisional results and AI commentary clearly

**Files:**
- Modify: `src/medical_evaluation/web/templates/report.html`
- Modify: `src/medical_evaluation/web/static/report.js`
- Modify: `tests/test_report_api.py`

- [ ] **Step 1: Write failing report-page tests**

Build a partial report and assert the HTML contains “阶段性得分”, “已评估 2/11 项”, “非最终成绩”, the AI commentary text, and “尚未接入自动评估” for excluded rows. Assert it does not render the provisional value as the official `/ 100` score.

- [ ] **Step 2: Run report tests and confirm RED**

Run:

```powershell
python -m pytest tests/test_report_api.py -q
```

Expected: failures because the page only knows the legacy final-score block.

- [ ] **Step 3: Update template and browser-side refresh**

Render provisional exact scores or ranges from the new fields. Add a permanent “非最终成绩” label and evaluated count. Render deterministic reason/suggestion separately from an “AI 点评” section. Update `scoreLabel` to use provisional fields after human review while keeping excluded rows excluded.

- [ ] **Step 4: Run report tests and confirm GREEN**

Run:

```powershell
python -m pytest tests/test_report_api.py -q
```

Expected: all report tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/medical_evaluation/web/templates/report.html src/medical_evaluation/web/static/report.js tests/test_report_api.py
git commit -m "feat: show provisional evaluation report"
```

### Task 6: Document, verify, and package server synchronization

**Files:**
- Modify: `README.md`
- Modify: `docs/server-runbook.md`

- [ ] **Step 1: Update operational documentation**

Document that CP01 is paused, success currently evaluates 2/11 while missing CP09 inputs reduce other videos to 1/11, and the score is provisional. Add exact environment variables and commands for starting Qwen, starting the real web pipeline, submitting each preset, and locating `report.json`. State that SAM and Qwen should use different GPUs.

- [ ] **Step 2: Run targeted and full verification**

Run:

```powershell
python -m pytest tests/test_scoring.py tests/test_pipeline_vertical.py tests/test_cp09_cp11_extractor.py tests/test_vlm_client.py tests/test_runtime.py tests/test_app.py tests/test_report_api.py -q
python -m pytest -q
ruff check src scripts tests
git diff --check
```

Expected: all tests and Ruff pass. The existing Starlette TestClient deprecation warning may remain documented, but no new warnings or failures are accepted.

- [ ] **Step 3: Commit documentation**

```powershell
git add README.md docs/server-runbook.md
git commit -m "docs: run CP09 CP11 scoring with local Qwen"
```

- [ ] **Step 4: Merge and create an incremental bundle**

After branch review, fast-forward `main`, confirm `origin/main` if network access is available, and create a bundle based on the server's current `124dcf1` ancestor:

```powershell
$shortSha = git rev-parse --short HEAD
$bundle = "cp09-cp11-scoring-llm-$shortSha.bundle"
git bundle create $bundle 124dcf1..main
git bundle verify $bundle
Get-FileHash $bundle -Algorithm SHA256
```

Do not add the bundle, annotations, videos, model files, or other user data to Git.

- [ ] **Step 5: Server acceptance**

On the GPU server, verify and fast-forward the bundle, run the full test suite, start Qwen on one free GPU and the SAM-backed web service on another, then submit `success`, `failure`, and `clamp_failure`. Confirm the reports show 2/11 for success and the actual dynamic count for the other videos; inspect CP09/CP11 evidence images and confirm LLM failures cannot alter deterministic JSON status or scores.
