# CP01 Pen Presence Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automatic, text-prompted CP01 marking-pen presence gate that uses no per-video object annotations, short-circuits absent cases, and preserves auditable positive or negative evidence.

**Architecture:** A focused OpenCV feature module validates the visible appearance of each SAM3 pen mask. A standalone extractor runs one sparse full-stage text session, aggregates consecutive accepted frames, writes per-frame evidence, and returns pen-gate features without invoking template or ink analysis. The existing template Judge is updated to consume `pen_presence_detected` rather than pen-tip contact.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, Pydantic domain models, SAM3 `VideoSegmenter`, pytest.

---

### Task 1: Pen candidate appearance measurement

**Files:**
- Create: `src/medical_evaluation/features/cp01_pen.py`
- Create: `tests/features/test_cp01_pen.py`

- [ ] **Step 1: Write failing tests for empty, bright, fragmented, and dark dominant masks**

Define tests against a wished-for `measure_pen_candidate(frame_bgr, mask)` API. Assert that empty masks return `pen_not_observed`, bright metal-like regions return `pen_appearance_mismatch`, fragmented masks return `fragmented_pen_mask`, and a dark dominant elongated component is accepted with explicit metrics.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/features/test_cp01_pen.py -q
```

Expected: collection or import failure because `cp01_pen` does not exist.

- [ ] **Step 3: Implement the minimal measurement module**

Add an immutable `PenCandidateMeasurement` containing `observed`, `accepted`, `reason`, `area_px`, `dark_pixel_ratio`, `dominant_component_ratio`, and `elongation`. Validate matching image/mask shapes, retain the dominant connected component, measure HSV value and minimum-area-rectangle elongation, and return explicit rejection reasons.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2 and expect all tests to pass.

### Task 2: Automatic full-stage pen extractor and negative evidence

**Files:**
- Create: `src/medical_evaluation/extractors/cp01_pen.py`
- Create: `tests/test_cp01_pen_extractor.py`

- [ ] **Step 1: Write failing extractor tests**

Use a fake `VideoSegmenter` and a tiny generated video to verify:

- the only model prompt is text `black pen held in a gloved hand`;
- no annotation prompt is read;
- one accepted frame does not pass;
- three consecutive accepted sampled frames set `pen_presence_detected=true`;
- no accepted masks set `pen_presence_detected=false` and still write raw/mask/overlay negative evidence;
- output includes valid count, maximum consecutive count, first/last seen seconds, stage reliability, per-frame rejection reasons, and an evidence JSON;
- only one full-stage segmentation session occurs and no template/dam analysis is invoked.

- [ ] **Step 2: Run the extractor tests and verify RED**

Run:

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_cp01_pen_extractor.py -q
```

Expected: import failure because `Cp01PenGateExtractor` does not exist.

- [ ] **Step 3: Implement the minimal extractor**

Implement `Cp01PenGateExtractor` with a 1 FPS full-stage scan, a minimum of three consecutive accepted frames, raw-frame fallback when SAM3 returns no frames, and auditable bundles under `cp_01/pen_gate/`. Return only the pen-gate features and representative evidence; do not call template or ink code.

- [ ] **Step 4: Run extractor and feature tests and verify GREEN**

Run:

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/features/test_cp01_pen.py tests/test_cp01_pen_extractor.py -q
```

Expected: all tests pass.

### Task 3: Align deterministic CP01 state with the approved gate

**Files:**
- Modify: `src/medical_evaluation/judges/cp01_template.py`
- Modify: `tests/judges/test_cp01_template_judge.py`

- [ ] **Step 1: Write failing Judge tests**

Replace `pen_dam_contact_observed` expectations with `pen_presence_detected`. Verify reliable absence returns `incomplete/cp01_not_performed`, while reliable presence proceeds to template/ink states without any pen-tip or overlap field.

- [ ] **Step 2: Run the Judge tests and verify RED**

Run:

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/judges/test_cp01_template_judge.py -q
```

Expected: failure because the Judge still requires `pen_dam_contact_observed`.

- [ ] **Step 3: Implement the approved state transition**

Make reliable pen absence return `incomplete/cp01_not_performed`. Remove the pen-tip-contact dependency and keep every downstream template/ink reliability branch unchanged.

- [ ] **Step 4: Run the focused Judge tests and verify GREEN**

Run the command from Step 2 and expect all tests to pass.

### Task 4: Focused and full local verification

**Files:**
- Modify only files changed by Tasks 1-3 if verification exposes a direct regression.

- [ ] **Step 1: Run the CP01 focused suite**

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/features/test_cp01_pen.py tests/test_cp01_pen_extractor.py tests/features/test_cp01_template.py tests/judges/test_cp01_template_judge.py -q
```

- [ ] **Step 2: Run the complete suite**

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
```

- [ ] **Step 3: Run Ruff and diff checks**

```powershell
& D:\Medical_evaluation\.venv\Scripts\ruff.exe check --no-cache src tests scripts/run_early_cp_sam3_gate.py
git diff --check
```

Expected: all commands exit zero. The existing Starlette/httpx deprecation warning is allowed.

### Task 5: Real three-video pen-gate practice

**Files:**
- Modify: `scripts/run_early_cp_sam3_gate.py`
- Modify: `tests/scripts/test_early_cp_gate.py`

- [ ] **Step 1: Write a failing test for a CP01 pen-only entry point**

Add `--cp01-pen-only` behavior that invokes the production `Cp01PenGateExtractor`, rejects non-CP01 use, and writes one summary without running board or dam prompts.

- [ ] **Step 2: Verify RED, implement the minimal entry point, and verify GREEN**

Run:

```powershell
& D:\Medical_evaluation\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/scripts/test_early_cp_gate.py -q
```

- [ ] **Step 3: Commit and synchronize the focused implementation**

Commit only the plan, CP01 pen feature/extractor/Judge, tests, and runner changes. Push the feature branch when network permits. Before and after server fast-forward, verify the server `rubric.py` hash remains `14b3b33b010825f3afe605d9d77d12827f79b94c0833c6e3947c8d716903be77`.

- [ ] **Step 4: Check GPU5 MPS and run success, failure, and clamp_failure**

Use only:

```bash
/home/tangm/.local/bin/gpu5-run env CUDA_VISIBLE_DEVICES=0 \
  /home/tangm/miniconda3/envs/sam3_medical/bin/python \
  scripts/run_early_cp_sam3_gate.py --video-id VIDEO_ID \
  --checkpoint-id cp_01 --cp01-pen-only --output-dir OUTPUT_DIR \
  --checkpoint models/SAM3.1/sam3.1_multiplex.pt \
  --bpe-path external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz --device cuda:0
```

Run one video at a time. Inspect original frames, raw masks, selected masks, overlays, consecutive counts, and rejection reasons. Do not infer truth from video names or legacy JSON object prompts.

- [ ] **Step 5: Record the gate conclusion**

Report the observed state for each video. Pen absence must short-circuit. Pen presence only authorizes the next CP01 template task and must not increment the formal evaluated checkpoint count.
