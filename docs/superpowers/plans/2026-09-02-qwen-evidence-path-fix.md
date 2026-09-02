# Qwen Evidence Path Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make real CP09/CP11 evidence images reach Qwen while preserving relative report paths, and apply the approved rubber-dam-specific system prompt.

**Architecture:** Keep extractor and report paths task-relative. Resolve those paths against `output_root / job.id` only at the pipeline-to-commentary boundary. Keep the Qwen schema and deterministic scoring guardrails unchanged.

**Tech Stack:** Python 3.10+, pathlib, Pydantic, httpx, pytest, Ruff

---

### Task 1: Resolve task-relative evidence paths

**Files:**
- Modify: `tests/test_pipeline_vertical.py`
- Modify: `src/medical_evaluation/pipeline.py`

- [ ] **Step 1: Write the failing regression test**

Add an extractor fixture that returns:

```python
EvidenceItem(
    time_sec=1.0,
    overlay_path="cp_09/overlays/frame.jpg",
    rule="test_evidence",
)
```

Create `tmp_path / "jobs" / "job-1" / "cp_09" / "overlays" / "frame.jpg"`,
run the pipeline, and assert the CP09 reviewer request receives that complete existing path.

- [ ] **Step 2: Verify the regression test fails**

Run:

```powershell
python -m pytest tests/test_pipeline_vertical.py::test_commentary_resolves_evidence_inside_job_directory -q
```

Expected: FAIL because the reviewer currently receives
`cp_09/overlays/frame.jpg`.

- [ ] **Step 3: Implement the minimal path resolution**

Pass `self.output_root / job.id` into `AnalysisPipeline._review` and construct the request with:

```python
evidence_images=[
    evidence_root / item.overlay_path
    for item in result.evidence
]
```

Do not modify the `EvidenceItem` objects stored in the report.

- [ ] **Step 4: Verify the regression test passes**

Run the same targeted pytest command. Expected: PASS.

### Task 2: Apply the approved Qwen system prompt

**Files:**
- Modify: `tests/test_vlm_client.py`
- Modify: `src/medical_evaluation/vlm/client.py`

- [ ] **Step 1: Write the failing prompt test**

Capture the outgoing payload with `httpx.MockTransport` and assert the system message is exactly:

```text
你是牙科操作考核的证据点评助手，现在需要对橡皮障隔离技术相关操作进行点评。
只能使用请求中给出的考核标准、确定性规则结论、特征和证据图。
必须引用使用过的证据图索引；证据不足时明确说明不确定。
只能返回符合给定字段的 JSON，不要 Markdown，不要额外字段。
绝对不能给出、修改或建议任何分数；确定性规则结论不可被覆盖。
```

- [ ] **Step 2: Verify the prompt test fails**

Run the new test alone. Expected: FAIL because the first sentence lacks the approved scope.

- [ ] **Step 3: Update only `SYSTEM_PROMPT`**

Replace the first line with the approved sentence and leave every other constraint unchanged.

- [ ] **Step 4: Verify the prompt test passes**

Run the same targeted pytest command. Expected: PASS.

### Task 3: Regression verification and delivery

**Files:**
- Verify all modified source, tests, spec, and plan files

- [ ] **Step 1: Run focused tests**

```powershell
python -m pytest tests/test_pipeline_vertical.py tests/test_vlm_client.py -q
```

- [ ] **Step 2: Run full verification**

```powershell
python -m pytest -q
ruff check src scripts tests
git diff --check
```

- [ ] **Step 3: Commit and synchronize**

Commit only the planned files, push `main`, verify `origin/main`, and create a full Git bundle
for the GPU server. Preserve all user data and existing untracked files.

- [ ] **Step 4: Verify on the GPU server**

Fast-forward the bundle, run focused tests, restart only the real web pipeline, submit a new
`success` job, and confirm CP09/CP11 both have `ai_commentary.source == "qwen"`.

