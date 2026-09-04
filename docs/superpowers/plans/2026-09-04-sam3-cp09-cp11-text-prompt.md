# SAM3 CP09/CP11 Text Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留 SAM2、Judge、评分和 Qwen 行为的前提下，用固定 SAM3 文本提示完成 CP09 支架与 CP11 橡皮布/支架跟踪，并在 A5000 上完成三个样例的可审计验证。

**Architecture:** 运行时显式选择 SAM2 或 SAM3；一个独立提示策略把 SAM2 标注提示与 SAM3 固定文本隔离开，CP09/CP11 只消费统一提示接口。Sam3Backend 复用现有采样帧序列，将官方 SAM3.1 会话输出映射回 FrameMasks；审计包装器在不改变分割契约的情况下记录每次跟踪。

**Tech Stack:** Python 3.11/3.12、Pydantic 2、NumPy、OpenCV、PyTorch 2.10、Meta SAM3.1 multiplex predictor、pytest、Ruff。

---

## 文件结构

- Create: `src/medical_evaluation/segmentation/prompt_policy.py` — CP09/CP11 的标注提示与固定文本提示策略。
- Create: `src/medical_evaluation/segmentation/audit.py` — VideoSegmenter 透明审计包装器和 segmentation_metadata.json。
- Create: `tests/test_segmentation_prompt_policy.py` — 两类提示策略的契约测试。
- Create: `tests/test_sam3_backend.py` — 官方请求格式、采样映射、候选选择和清理测试。
- Create: `tests/test_segmentation_audit.py` — 审计文件原子更新与失败记录测试。
- Create: `scripts/run_sam3_text_smoke.py` — 服务器短视频文本推理与掩膜导出入口。
- Create: `tests/scripts/test_run_sam3_text_smoke.py` — smoke CLI 的无 GPU 单元测试。
- Modify: `src/medical_evaluation/segmentation/sam3_backend.py` — 改为真实 SAM3.1 multiplex API。
- Modify: `src/medical_evaluation/settings.py` — 严格后端类型和 SAM3 路径配置。
- Modify: `src/medical_evaluation/runtime.py` — 显式构建后端、提示策略和任务级审计。
- Modify: `src/medical_evaluation/extractors/cp09.py` — 从提示策略取得支架提示。
- Modify: `src/medical_evaluation/extractors/cp11.py` — 从提示策略取得橡皮布和跨阶段支架提示。
- Modify: `src/medical_evaluation/extractors/cp09_cp11.py` — 用提示策略验证 CP09 输入，不再硬要求人工支架提示。
- Modify: `tests/test_cp09_extractor.py`、`tests/test_cp11_extractor.py`、`tests/test_cp09_cp11_extractor.py` — 文本模式和 SAM2 回归。
- Modify: `tests/test_settings.py`、`tests/test_runtime.py` — 后端配置和构建分支。
- Modify: `docs/server-runbook.md` — SAM3 环境、命令、版本和失败处理。

### Task 1: 建立后端无关的提示策略

**Files:**
- Create: `src/medical_evaluation/segmentation/prompt_policy.py`
- Create: `tests/test_segmentation_prompt_policy.py`

- [ ] **Step 1: 写失败测试**

```python
from medical_evaluation.annotations import PointPrompt, VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.prompt_policy import (
    AnnotationPromptPolicy,
    TextPromptPolicy,
)


def test_text_policy_ignores_annotation_object_prompts() -> None:
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[
            PointPrompt(
                video_id="success",
                frame_time_sec=9,
                object_id="rubber_dam_frame",
                x=0.1,
                y=0.2,
            )
        ],
    )
    policy = TextPromptPolicy()

    frame = policy.frame_prompts(
        annotations,
        TimeRange(start_sec=10, end_sec=20),
        checkpoint_id="cp_09",
        boundary_tolerance_sec=0.5,
    )
    dam = policy.dam_prompts(
        annotations,
        TimeRange(start_sec=30, end_sec=33),
        checkpoint_id="cp_11",
    )

    assert [(p.kind, p.text, p.frame_time_sec) for p in frame] == [
        ("text", "white U-shaped dental frame", 10)
    ]
    assert [(p.kind, p.text, p.frame_time_sec) for p in dam] == [
        ("text", "green dental rubber dam", 30)
    ]


def test_annotation_policy_preserves_existing_prompt_rules() -> None:
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[
            PointPrompt(
                video_id="success",
                frame_time_sec=10,
                object_id="rubber_dam_frame",
                x=0.5,
                y=0.5,
            )
        ],
    )
    result = AnnotationPromptPolicy().frame_prompts(
        annotations,
        TimeRange(start_sec=10, end_sec=20),
        checkpoint_id="cp_09",
        boundary_tolerance_sec=0.5,
    )
    assert result[0].kind == "point"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_segmentation_prompt_policy.py -q`
Expected: FAIL，提示 `medical_evaluation.segmentation.prompt_policy` 不存在。

- [ ] **Step 3: 写最小实现**

```python
# src/medical_evaluation/segmentation/prompt_policy.py
from typing import Protocol

from medical_evaluation.annotations import VideoAnnotations
from medical_evaluation.domain import TimeRange
from medical_evaluation.segmentation.base import SegmentationPrompt
from medical_evaluation.segmentation.prompts import prompts_for_object

FRAME_TEXT_PROMPT = "white U-shaped dental frame"
DAM_TEXT_PROMPT = "green dental rubber dam"


class Cp09Cp11PromptPolicy(Protocol):
    def frame_prompts(
        self,
        annotations: VideoAnnotations,
        time_range: TimeRange,
        *,
        checkpoint_id: str,
        boundary_tolerance_sec: float = 0.0,
    ) -> list[SegmentationPrompt]: ...

    def dam_prompts(
        self,
        annotations: VideoAnnotations,
        time_range: TimeRange,
        *,
        checkpoint_id: str,
    ) -> list[SegmentationPrompt]: ...


class AnnotationPromptPolicy:
    def frame_prompts(self, annotations, time_range, *, checkpoint_id, boundary_tolerance_sec=0.0):
        return prompts_for_object(
            annotations,
            "rubber_dam_frame",
            time_range,
            checkpoint_id=checkpoint_id,
            boundary_tolerance_sec=boundary_tolerance_sec,
        )

    def dam_prompts(self, annotations, time_range, *, checkpoint_id):
        prompts = prompts_for_object(
            annotations, "rubber_dam", time_range, checkpoint_id=checkpoint_id
        )
        if not 3 <= len(prompts) <= 5 or any(
            item.kind != "point" or not item.positive for item in prompts
        ):
            raise ValueError("cp_11 requires 3-5 positive rubber_dam points")
        return prompts


class TextPromptPolicy:
    def frame_prompts(self, _annotations, time_range, *, checkpoint_id, boundary_tolerance_sec=0.0):
        return [
            SegmentationPrompt(
                object_id="rubber_dam_frame",
                kind="text",
                frame_time_sec=time_range.start_sec,
                text=FRAME_TEXT_PROMPT,
            )
        ]

    def dam_prompts(self, _annotations, time_range, *, checkpoint_id):
        return [
            SegmentationPrompt(
                object_id="rubber_dam",
                kind="text",
                frame_time_sec=time_range.start_sec,
                text=DAM_TEXT_PROMPT,
            )
        ]
```

- [ ] **Step 4: 运行提示策略测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_segmentation_prompt_policy.py tests/test_segmentation_prompts.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/medical_evaluation/segmentation/prompt_policy.py tests/test_segmentation_prompt_policy.py
git commit -m "feat: add CP09 CP11 segmentation prompt policies"
```

### Task 2: 用真实 SAM3.1 请求契约重写后端

**Files:**
- Create: `tests/test_sam3_backend.py`
- Modify: `src/medical_evaluation/segmentation/sam3_backend.py`

- [ ] **Step 1: 写失败测试，固定官方请求与帧映射**

```python
class FakeSam3Predictor:
    def __init__(self) -> None:
        self.requests = []
        self.stream_requests = []

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "session-1"}
        if request["type"] == "add_prompt":
            return {
                "frame_index": 0,
                "outputs": {
                    "out_obj_ids": np.array([7]),
                    "out_binary_masks": np.ones((1, 20, 40), dtype=bool),
                },
            }
        return {"is_success": True}

    def handle_stream_request(self, request):
        self.stream_requests.append(request)
        for local_index in range(2):
            yield {
                "frame_index": local_index,
                "outputs": {
                    "out_obj_ids": np.array([7]),
                    "out_binary_masks": np.ones((1, 20, 40), dtype=bool),
                },
            }


def test_sam3_uses_bounded_sequence_and_maps_source_frames(tmp_path: Path) -> None:
    predictor = FakeSam3Predictor()
    video = make_test_video(tmp_path / "video.mp4", fps=10, seconds=3, size=(40, 20))
    backend = Sam3Backend(tmp_path / "sam3.pt", predictor=predictor)
    prompt = SegmentationPrompt(
        object_id="rubber_dam_frame",
        kind="text",
        frame_time_sec=1.0,
        text="white U-shaped dental frame",
    )

    frames = list(
        backend.track(video, TimeRange(start_sec=1, end_sec=3), [prompt], sample_fps=1)
    )

    start = predictor.requests[0]
    assert Path(start["resource_path"]).name == "frames"
    assert predictor.requests[1] == {
        "type": "add_prompt",
        "session_id": "session-1",
        "frame_index": 0,
        "text": "white U-shaped dental frame",
        "output_prob_thresh": 0.5,
    }
    assert predictor.stream_requests == [{
        "type": "propagate_in_video",
        "session_id": "session-1",
        "propagation_direction": "forward",
        "start_frame_index": 0,
        "max_frame_num_to_track": 2,
        "output_prob_thresh": 0.5,
    }]
    assert [item.frame_index for item in frames] == [10, 20]
    assert all(set(item.masks) == {"rubber_dam_frame"} for item in frames)
    assert predictor.requests[-1]["type"] == "close_session"
    assert Path(start["resource_path"]).exists() is False
```

同时增加四个小测试：拒绝空提示、拒绝非文本或多文本提示、零候选返回空帧序列、多候选无分数抛出 `Sam3AmbiguousTextResult`，以及 add/propagate 异常时仍关闭会话。

- [ ] **Step 2: 运行并确认旧骨架失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_sam3_backend.py -q`
Expected: FAIL，表现为请求仍使用完整视频、绝对帧号或错误 builder。

- [ ] **Step 3: 实现 builder 与 bounded sequence**

```python
class Sam3AmbiguousTextResult(RuntimeError):
    pass


class Sam3Backend:
    def __init__(
        self,
        checkpoint: Path,
        *,
        bpe_path: Path | None = None,
        device: str = "cuda:0",
        output_prob_threshold: float = 0.5,
        predictor: Any | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.bpe_path = bpe_path
        self.device = device
        self.output_prob_threshold = output_prob_threshold
        if predictor is None:
            import torch
            from sam3.model_builder import build_sam3_multiplex_video_predictor

            torch.cuda.set_device(device)
            predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=str(checkpoint),
                bpe_path=str(bpe_path) if bpe_path else None,
                use_fa3=False,
                use_rope_real=False,
                compile=False,
                warm_up=False,
                async_loading_frames=False,
            )
        self.predictor = predictor
```

`track` 必须调用现有 `write_sampled_frame_sequence`，并使用 `sequence.entries[local_index]` 映射回源帧。第一轮严格要求一个 text prompt；从 add_prompt 输出选定对象 ID，单候选直接选，多候选只在 `out_scores` 存在时取最高分，否则抛出歧义错误。传播时只保留选定 ID，最后在 `finally` 发送 close_session。

- [ ] **Step 4: 运行后端与既有契约测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_sam3_backend.py tests/test_segmentation_contract.py -q`
Expected: PASS，并且 SAM2 契约测试无回归。

- [ ] **Step 5: 提交**

```bash
git add src/medical_evaluation/segmentation/sam3_backend.py tests/test_sam3_backend.py
git commit -m "feat: adapt SAM3.1 text video tracking"
```

### Task 3: 增加严格设置和运行时后端选择

**Files:**
- Modify: `src/medical_evaluation/settings.py`
- Modify: `src/medical_evaluation/runtime.py`
- Modify: `tests/test_settings.py`
- Modify: `tests/test_runtime.py`

- [ ] **Step 1: 写失败测试**

```python
def test_settings_resolve_sam3_paths(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        sam_backend="sam3",
        sam3_checkpoint_path=Path("models/sam3.pt"),
        sam3_bpe_path=Path("external/sam3/sam3/assets/bpe.txt.gz"),
    )
    assert settings.sam3_checkpoint_path == (tmp_path / "models/sam3.pt").resolve()
    assert settings.sam3_bpe_path == (
        tmp_path / "external/sam3/sam3/assets/bpe.txt.gz"
    ).resolve()


def test_runtime_builds_sam3_without_validating_sam2(tmp_path: Path) -> None:
    settings = make_settings(tmp_path).model_copy(
        update={
            "sam_backend": "sam3",
            "sam2_checkpoint_path": tmp_path / "missing-sam2.pt",
            "sam3_checkpoint_path": tmp_path / "sam3.pt",
            "sam3_bpe_path": tmp_path / "bpe.gz",
        }
    )
    settings.sam3_checkpoint_path.write_bytes(b"weights")
    settings.sam3_bpe_path.write_bytes(b"vocab")
    calls = []

    def factory(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return FakeBackend()

    build_analysis_pipeline(settings, segmenter_factory=factory)
    assert calls[0][0] == settings.sam3_checkpoint_path
    assert calls[0][1]["bpe_path"] == settings.sam3_bpe_path
```

另加：非法 `sam_backend` Pydantic 校验失败、SAM3 checkpoint/BPE 分别缺失时报可操作错误、sam2 分支的旧构造参数保持不变。

- [ ] **Step 2: 运行并确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_settings.py tests/test_runtime.py -q`
Expected: FAIL，当前 Settings 无 SAM3 路径且 runtime 硬编码 SAM2。

- [ ] **Step 3: 写最小实现**

```python
# settings.py
sam_backend: Literal["sam2", "sam3"] = "sam3"
sam3_checkpoint_path: Path = Path("models/SAM3.1/sam3.1_multiplex.pt")
sam3_bpe_path: Path = Path("external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
sam3_output_prob_threshold: float = Field(default=0.5, ge=0, le=1)
sam3_source_revision: str = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
sam3_checkpoint_sha256: str = (
    "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
)
```

在 `resolve_paths` 中解析两个 SAM3 路径。把 runtime 的默认 `segmenter_factory` 改为 `None`：sam2 分支默认选择 `Sam2Backend` 并只验证 SAM2；sam3 分支默认选择 `Sam3Backend` 并只验证 checkpoint/BPE。sam3 构造参数必须包含 `device` 和 `output_prob_threshold`，不得 fallback。

根据 `settings.sam_backend` 创建 `AnnotationPromptPolicy` 或 `TextPromptPolicy`，并传入 CP09、CP11 和组合提取器。现有 runtime 测试的 `make_settings` 显式设置 `sam_backend="sam2"`。

- [ ] **Step 4: 运行设置和 runtime 测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_settings.py tests/test_runtime.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/medical_evaluation/settings.py src/medical_evaluation/runtime.py tests/test_settings.py tests/test_runtime.py
git commit -m "feat: select SAM2 or SAM3 at runtime"
```

### Task 4: 接入 CP09 文本支架提示

**Files:**
- Modify: `src/medical_evaluation/extractors/cp09.py`
- Modify: `src/medical_evaluation/extractors/cp09_cp11.py`
- Modify: `tests/test_cp09_extractor.py`
- Modify: `tests/test_cp09_cp11_extractor.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_cp09_extractor.py` 新增一个只有 oral_region、没有支架人工点的标注，注入 `TextPromptPolicy`：

```python
def test_text_mode_starts_frame_tracking_at_cp09_start(tmp_path: Path) -> None:
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[BoxPrompt(
            video_id="success", frame_time_sec=0.5, object_id="oral_region",
            x1=0.25, y1=0.2, x2=0.75, y2=0.8,
        )],
    )
    segmenter = FakeSegmenter([])
    extractor = Cp09FeatureExtractor(
        segmenter=segmenter,
        annotations=annotations,
        evidence_root=tmp_path,
        prompt_policy=TextPromptPolicy(),
    )
    extractor.extract(
        tmp_path / "video.avi",
        "cp_09",
        TimeRange(start_sec=0.5, end_sec=1.6),
        dense_fps=2,
        analysis_width=1280,
    )
    assert [(p.kind, p.text, p.frame_time_sec) for p in segmenter.prompts] == [
        ("text", "white U-shaped dental frame", 0.5)
    ]
    assert segmenter.time_range == TimeRange(start_sec=0.5, end_sec=1.6)
```

在组合提取器测试中验证：TextPromptPolicy 下没有人工 frame prompt 仍调用 delegate；AnnotationPromptPolicy 下仍报告 `EvaluationInputMissing`。

- [ ] **Step 2: 运行并确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cp09_extractor.py tests/test_cp09_cp11_extractor.py -q`
Expected: FAIL，构造器还不接收 prompt_policy。

- [ ] **Step 3: 写最小实现**

给两个提取器构造器增加 `prompt_policy: Cp09Cp11PromptPolicy | None = None`，默认使用 `AnnotationPromptPolicy()`。CP09 用：

```python
frame_prompts = self.prompt_policy.frame_prompts(
    self.annotations,
    time_range,
    checkpoint_id=checkpoint_id,
    boundary_tolerance_sec=self.prompt_boundary_tolerance_sec,
)
```

组合提取器的 `_validate_cp09_inputs` 先调用同一策略的 `frame_prompts`；策略失败转为 `EvaluationInputMissing`。oral_region 仍必须是时间容差内唯一 BoxPrompt。不要改中心距离、三帧门槛、阈值或证据输出。

- [ ] **Step 4: 运行 CP09 和 Judge 回归**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cp09_extractor.py tests/test_cp09_cp11_extractor.py tests/test_judges_cp09_cp11.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/medical_evaluation/extractors/cp09.py src/medical_evaluation/extractors/cp09_cp11.py tests/test_cp09_extractor.py tests/test_cp09_cp11_extractor.py
git commit -m "feat: use SAM3 text prompt for CP09 frame"
```

### Task 5: 接入 CP11 文本橡皮布和跨阶段支架

**Files:**
- Modify: `src/medical_evaluation/extractors/cp11.py`
- Modify: `tests/test_cp11_extractor.py`

- [ ] **Step 1: 写失败测试**

增强 FakeSegmenter 记录 `time_range`、提示和 sample_fps，新增：

```python
def test_text_mode_prompts_dam_at_final_start_and_frame_at_cp09_start(tmp_path: Path) -> None:
    annotations = VideoAnnotations(
        video_id="success",
        prompts=[BoxPrompt(
            video_id="success", frame_time_sec=3.0, object_id="nose_region",
            x1=0.0, y1=0.0, x2=0.2, y2=0.2,
        )],
    )
    extractor = Cp11FeatureExtractor(
        segmenter=segmenter,
        annotations=annotations,
        evidence_root=tmp_path,
        frame_reference_time_range=TimeRange(start_sec=0.4, end_sec=0.6),
        min_stage_dam_presence_ratio=0.05,
        prompt_policy=TextPromptPolicy(),
    )
    extractor.extract(
        _video(tmp_path / "attempted.avi", green_stage=True),
        "cp_11",
        TimeRange(start_sec=1.0, end_sec=4.0),
        dense_fps=2,
        analysis_width=1280,
    )
    dam_call, frame_call = segmenter.recorded_calls
    assert (dam_call.time_range.start_sec, dam_call.prompts[0].text) == (
        1.0, "green dental rubber dam"
    )
    assert (frame_call.time_range.start_sec, frame_call.prompts[0].text) == (
        0.4, "white U-shaped dental frame"
    )
    assert frame_call.time_range.end_sec == 4.0
```

注意本例 CP11 长度正好三秒，final_start 为 1.0。再保留两个既有测试，证明无绿色和末尾消失仍不调用 segmenter。

- [ ] **Step 2: 运行并确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cp11_extractor.py -q`
Expected: FAIL，当前仍强制 3–5 个 rubber_dam 正点。

- [ ] **Step 3: 写最小实现**

```python
dam_prompts = self.prompt_policy.dam_prompts(
    self.annotations, final_range, checkpoint_id=checkpoint_id
)
frame_prompts = self.prompt_policy.frame_prompts(
    self.annotations,
    self.frame_reference_time_range,
    checkpoint_id=checkpoint_id,
)
```

删除 CP11 提取器内重复的 3–5 正点校验，因为 AnnotationPromptPolicy 已承担该约束。保持绿色阶段 gate 在提示生成之前；保持 final_range、nose box、Lab 外观、掩膜与绿色交集、三个有效帧和全部特征不变。

- [ ] **Step 4: 运行 CP11、Judge 和评分回归**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cp11_extractor.py tests/test_cp11_judge_states.py tests/test_judges_cp09_cp11.py tests/test_scoring.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/medical_evaluation/extractors/cp11.py tests/test_cp11_extractor.py
git commit -m "feat: use SAM3 text prompts for CP11"
```

### Task 6: 写入任务级分割审计

**Files:**
- Create: `src/medical_evaluation/segmentation/audit.py`
- Create: `tests/test_segmentation_audit.py`
- Modify: `src/medical_evaluation/runtime.py`

- [ ] **Step 1: 写失败测试**

```python
def test_audited_segmenter_records_completed_and_failed_calls(tmp_path: Path) -> None:
    output = tmp_path / "segmentation_metadata.json"
    wrapped = AuditedVideoSegmenter(
        FakeSegmenter(),
        output_path=output,
        static_metadata={
            "backend": "sam3",
            "source_revision": "660a5e9",
            "checkpoint_sha256": "0567debe",
        },
    )
    list(wrapped.track(
        Path("video.mp4"),
        TimeRange(start_sec=10, end_sec=20),
        [SegmentationPrompt(
            object_id="rubber_dam_frame", kind="text",
            frame_time_sec=10, text="white U-shaped dental frame",
        )],
        sample_fps=2,
    ))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backend"] == "sam3"
    assert payload["tracks"][0]["status"] == "completed"
    assert payload["tracks"][0]["prompt_texts"] == ["white U-shaped dental frame"]
    assert payload["tracks"][0]["time_range"] == {"start_sec": 10.0, "end_sec": 20.0}
```

增加一个 delegate 抛异常的测试，确认 finally 仍写 `status="failed"`、异常类型和耗时，但不吞掉原异常。

- [ ] **Step 2: 运行并确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_segmentation_audit.py -q`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 写最小实现**

`AuditedVideoSegmenter.track` 用 `time.perf_counter()` 包住 delegate generator，逐帧计数并原样 yield；finally 使用 `atomic_write_json` 重写累计记录。提示只记录 kind、object_id、frame_time_sec 和 text，禁止序列化 mask/坐标。若 torch/CUDA 可用则记录 `torch.cuda.max_memory_allocated(device) / 1024**2` 为 process_peak_allocated_mib，否则为 null。

runtime 在每个 `extractor_factory(job)` 内创建任务级包装器，输出到 `data/jobs/{job_id}/segmentation_metadata.json`。static metadata 对 sam3 使用 Settings 中锁定的源码提交和完整 checkpoint SHA-256；服务器 smoke 前用 sha256sum 验证实际文件与该值一致。sam2 记录其 model_version。CP09 与 CP11 共享同一包装器。

- [ ] **Step 4: 运行审计和 runtime 测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_segmentation_audit.py tests/test_runtime.py tests/test_pipeline_vertical.py -q`
Expected: PASS，report.json Schema 不变。

- [ ] **Step 5: 提交**

```bash
git add src/medical_evaluation/segmentation/audit.py src/medical_evaluation/runtime.py tests/test_segmentation_audit.py tests/test_runtime.py
git commit -m "feat: audit segmentation backend runs"
```

### Task 7: 增加可复现的 GPU 文本 smoke

**Files:**
- Create: `scripts/run_sam3_text_smoke.py`
- Create: `tests/scripts/test_run_sam3_text_smoke.py`
- Modify: `docs/server-runbook.md`

- [ ] **Step 1: 写 CLI 失败测试**

测试 monkeypatch `Sam3Backend` 返回两个 FrameMasks，调用 `main([...])` 后断言输出目录包含 `00000010.png`、`00000020.png` 和 summary.json；summary 包含提示、帧号、非空面积率、耗时。还要断言已有输出目录时报错，避免覆盖证据。

- [ ] **Step 2: 运行并确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/scripts/test_run_sam3_text_smoke.py -q`
Expected: FAIL，脚本不存在。

- [ ] **Step 3: 实现 CLI**

参数必须为：`--video`、`--start-sec`、`--end-sec`、`--sample-fps`、`--object-id`、`--text`、`--checkpoint`、`--bpe-path`、`--device`、`--output-dir`。CLI 构造一个 text SegmentationPrompt，调用 Sam3Backend.track，用 OpenCV 写每帧布尔 PNG，并通过 atomic_write_json 写 summary.json。输出目录必须不存在。

runbook 写明版本、权重哈希、sam3_medical 环境、A5000 参数、动态选卡、不得停止他人进程和两个固定提示。

- [ ] **Step 4: 运行 CLI 测试和 Ruff**

Run: `.\.venv\Scripts\python.exe -m pytest tests/scripts/test_run_sam3_text_smoke.py -q`
Expected: PASS。
Run: `.\.venv\Scripts\ruff.exe check scripts/run_sam3_text_smoke.py tests/scripts/test_run_sam3_text_smoke.py`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add scripts/run_sam3_text_smoke.py tests/scripts/test_run_sam3_text_smoke.py docs/server-runbook.md
git commit -m "test: add reproducible SAM3 text smoke"
```

### Task 8: 本地完整回归

**Files:**
- Verify only

- [ ] **Step 1: 运行 SAM3 聚焦测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_segmentation_prompt_policy.py tests/test_sam3_backend.py tests/test_segmentation_audit.py tests/test_settings.py tests/test_runtime.py tests/test_cp09_extractor.py tests/test_cp11_extractor.py tests/test_cp09_cp11_extractor.py tests/scripts/test_run_sam3_text_smoke.py -q`
Expected: 全部 PASS。

- [ ] **Step 2: 运行 Judge、评分和 Qwen 保护回归**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_judges_cp09_cp11.py tests/test_cp11_judge_states.py tests/test_scoring.py tests/test_vlm_client.py tests/test_vlm_schema.py -q`
Expected: 全部 PASS。

- [ ] **Step 3: 运行完整测试**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: 全部 PASS；记录真实通过数量，不沿用旧的 188 项数字。

- [ ] **Step 4: 运行完整 Ruff 和差异检查**

Run: `.\.venv\Scripts\ruff.exe check src tests scripts`
Expected: PASS。
Run: `git diff --check`
Expected: 无输出。

- [ ] **Step 5: 处理回归结果**

若有失败，返回引入该行为的 Task，先补充精确失败测试，再修改对应文件、重复 Task 的测试与提交步骤。若没有失败，不创建空提交。

### Task 9: 服务器 GPU smoke 与三视频验证

**Files:**
- Verify server checkout and generated data only

- [ ] **Step 1: 同步代码并核对版本**

先在 Windows 确认 HEAD 和干净的任务文件差异；通过正常 Git 或 bundle 同步服务器。服务器运行 `git rev-parse HEAD`，必须等于本轮最终提交。不得覆盖服务器现有 `src/medical_evaluation/rubric.py` 修改或用户数据。

- [ ] **Step 2: 动态选择 SAM3 GPU**

Run: `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader`
Expected: 查看实时占用，不把历史 GPU 6 当永久分配。随后把实时利用率为 0% 且显存占用最低的卡号保存为任务变量 sam3_gpu，并在执行前再次人工核对该卡。

- [ ] **Step 3: 运行两个短视频文本 smoke**

在 `sam3_medical` 中先验证权重，再分别运行 smoke CLI。服务器 success 视频使用 `videos/橡皮障完整.mp4`，CP09 区间为 175–195 秒，CP11 最后三秒为 255–258 秒：

```bash
echo "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6  models/SAM3.1/sam3.1_multiplex.pt" | sha256sum -c -
CUDA_VISIBLE_DEVICES="$sam3_gpu" conda run --no-capture-output -n sam3_medical python scripts/run_sam3_text_smoke.py --video "videos/橡皮障完整.mp4" --start-sec 175 --end-sec 195 --sample-fps 2 --object-id rubber_dam_frame --text "white U-shaped dental frame" --checkpoint models/SAM3.1/sam3.1_multiplex.pt --bpe-path external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz --device cuda:0 --output-dir data/runs/sam3-text/cp09-success
CUDA_VISIBLE_DEVICES="$sam3_gpu" conda run --no-capture-output -n sam3_medical python scripts/run_sam3_text_smoke.py --video "videos/橡皮障完整.mp4" --start-sec 255 --end-sec 258 --sample-fps 2 --object-id rubber_dam --text "green dental rubber dam" --checkpoint models/SAM3.1/sam3.1_multiplex.pt --bpe-path external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz --device cuda:0 --output-dir data/runs/sam3-text/cp11-success-final
```

Expected: 权重校验为 OK；两次推理 exit 0；每个 summary.json 至少三个非空帧；掩膜尺寸等于源视频。

- [ ] **Step 4: 目视检查短 smoke**

从每个输出选择首帧、中帧、末帧与原视频对应帧叠加查看。CP09 掩膜必须是白色 U 形支架，CP11 必须是绿色橡皮布；若语义不对，停止，不改 Judge 或阈值。

- [ ] **Step 5: 运行 success 完整真实管线**

设置 `MED_EVAL_SAM_BACKEND=sam3`、SAM3 checkpoint/BPE、选定 GPU；Qwen 使用另一张动态确认的 GPU。运行现有真实任务入口，记录 job id。Expected:

- report.json 有 11 项、仅 CP09/CP11 自动评价、final_score=null；
- segmentation_metadata.json 记录三个 track：CP09 支架、CP11 橡皮布、CP09 至 CP11 支架；
- CP09 和 CP11 均至少三个有效帧；
- Qwen source 为 qwen 或明确 fallback，但不能修改 Judge 和分数。

- [ ] **Step 6: 目视检查 success 证据**

检查 CP09、CP11 各最多三张 overlay。确认口腔框、固定中心、鼻部框、支架、橡皮布轮廓和帧时间正确。只有这一项通过后冻结两个提示文本。

- [ ] **Step 7: 依次运行 failure 与 clamp_failure**

保持同一代码、提示、阶段、Judge 和阈值。Expected: 完整保留两种既有 CP11 语义——从未出现为 incomplete/rubber_dam_not_observed，出现后末尾消失为 incorrect/rubber_dam_missing_at_end；这些早退场景不得调用 SAM3 精细分割。CP09 若无稳定语义掩膜应 needs_review，不能人工替代模型输出。

- [ ] **Step 8: 汇总 SAM2/SAM3 对照**

记录每视频、每 CP 的掩膜语义、有效帧数、特征、Judge 状态、耗时、峰值显存、提示数量和错误帧。只描述三个案例，不计算准确率；先确认掩膜差异，再另行提出阈值或混合提示设计。

## 完成标准

- 本地聚焦测试、完整 pytest、完整 Ruff 和 git diff 检查通过；
- A5000 完成两个固定文本的真实视频推理；
- success 完整报告与证据图通过人工目视；
- 两个失败视频保持既有阶段存在性和 Judge 语义；
- SAM2 仍可显式运行；
- SAM3 不读取人工支架/橡皮布提示，不静默 fallback；
- Judge、评分和 Qwen 权限边界无变化；
- 原视频、标注、已有报告及服务器用户修改未被覆盖。
