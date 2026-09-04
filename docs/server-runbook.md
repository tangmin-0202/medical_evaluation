# Ubuntu GPU 服务器运行手册

适用服务器快照：8 张 NVIDIA RTX A5000（每张约 24 GB）、驱动 570.124.06、CUDA 12.8。快照中 GPU 1、2、3、6 基本空闲，但每次启动前必须重新运行 `nvidia-smi`，不要依赖旧占用情况。

## 1. 获取代码和创建环境

```bash
git clone https://github.com/tangmin-0202/medical_evaluation.git
cd medical_evaluation
git rev-parse HEAD
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
pip install -e '.[dev,vision]'
python -m pytest -q
```

项目要求 Python 3.11 或 3.12。视频、权重、标注和报告均留在本机，不上传到外部 API。

## 2. 安装一个 SAM 后端

建议先用 SAM2.1 跑通点/框视频跟踪，再试 SAM3.1 文本提示。把第三方仓库放在项目外的 `external/`，并记录实际提交。

### SAM2.1

```bash
mkdir -p external models
git clone https://github.com/facebookresearch/sam2.git external/sam2
git -C external/sam2 rev-parse HEAD | tee models/sam2-code-commit.txt
pip install -e external/sam2
# 按 Meta 官方 README 下载相应 sam2.1 checkpoint 到 models/
sha256sum models/sam2.1_hiera_large.pt | tee models/sam2.1_hiera_large.pt.sha256
```

SAM2.1 使用 Apache-2.0。配置示例：

```yaml
segmentation:
  backend: sam2
  checkpoint: /absolute/path/models/sam2.1_hiera_large.pt
  device: cuda:0
  analysis_width: 960
  sparse_fps: 1.0
  dense_fps: 5.0
```

### SAM3.1

当前已验证的锁定版本：

- Meta 源码提交：`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`；
- 权重：`models/SAM3.1/sam3.1_multiplex.pt`；
- 权重 SHA-256：`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`；
- 独立 Conda 环境：`sam3_medical`（不得改动 `video_medical` 或 `qwen_medical`）；
- PyTorch `2.10.0+cu128`、`decord2 3.4.0`。普通 Python 包使用清华镜像；CUDA
  PyTorch wheel 仍使用 PyTorch 官方 CUDA 索引。

```bash
git -C external/sam3 rev-parse HEAD
conda run --no-capture-output -n sam3_medical python -m pip check
echo "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6  models/SAM3.1/sam3.1_multiplex.pt" | sha256sum -c -
```

SAM3 权重和代码受各自随附许可证约束，部署前阅读并保留 `external/sam3/LICENSE`
及 `models/SAM3.1/LICENSE`。本项目第一轮只使用官方 multiplex video predictor 的文本
提示模式，并转换为统一 `FrameMasks`；不静默回退 SAM2。

锁定提交的共享 `Sam3BasePredictor.start_session()` 会传入 multiplex `init_state()`
不接受的 `offload_state_to_cpu`。项目适配器按底层方法签名过滤不支持的参数，与
facebookresearch/sam3#543 的修复方式一致；不要直接修改 `external/sam3`。

每次先查看 GPU，选择利用率为 0 且显存占用最低的卡，不终止其他用户进程。下面的
`sam3_gpu` 必须按实时结果填写；进程内设备仍为 `cuda:0`：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
sam3_gpu=6
CUDA_VISIBLE_DEVICES="$sam3_gpu" conda run --no-capture-output -n sam3_medical \
  python scripts/run_sam3_text_smoke.py \
  --video "videos/橡皮障完整.mp4" --start-sec 175 --end-sec 195 --sample-fps 2 \
  --object-id rubber_dam_frame --text "white U-shaped dental frame" \
  --checkpoint models/SAM3.1/sam3.1_multiplex.pt \
  --bpe-path external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --device cuda:0 --output-dir data/runs/sam3-text/cp09-success
```

CP11 最后三秒使用固定文本 `green dental rubber dam`，区间 `255–258`。输出目录必须
不存在，防止覆盖已有证据。分别目视检查首、中、末帧掩膜；CP09 必须覆盖白色 U 形
支架，CP11 必须覆盖绿色橡皮布。语义不对时停止，不改 Judge 或阈值。

## 3. 启动本地 Qwen3-VL

Qwen 官方建议使用 vLLM，Qwen3-VL 需要 `vllm>=0.11.0`。4B BF16 权重约 9 GB，单张 24 GB A5000 通常比 SAM 与 Qwen 共卡更稳妥。

```bash
python -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install -U pip 'vllm>=0.11.0'
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85
```

另一个终端检查 OpenAI 兼容接口：

```bash
curl http://127.0.0.1:8001/v1/models
```

Qwen3-VL-4B-Instruct 模型卡标注 Apache-2.0。项目只向 `127.0.0.1:8001` 发送有限张派生证据图和特征 JSON，不发送原视频。

## 4. GPU 分配建议

根据 2026-08-20 的快照，GPU 0、4、5、7 已有任务。首次部署可暂定：

- GPU 1：SAM2.1/SAM3.1；
- GPU 2：Qwen3-VL-4B；
- Web/队列：CPU。

启动前执行：

```bash
nvidia-smi
watch -n 2 nvidia-smi
```

若 GPU 1/2 已被占用，修改 `CUDA_VISIBLE_DEVICES` 和 `device`，不要终止其他用户进程。

## 5. 启动当前网页 Demo

```bash
source .venv/bin/activate
cp config/models.example.yaml config/models.yaml
export MED_EVAL_VIDEOS_DIR=/absolute/path/to/橡皮障视频/橡皮障视频
export MED_EVAL_DATA_DIR=/absolute/path/to/medical_evaluation/data
export MED_EVAL_MODEL_CONFIG_PATH=/absolute/path/to/medical_evaluation/config/models.yaml
export MED_EVAL_VLM_BASE_URL=http://127.0.0.1:8001/v1
export MED_EVAL_VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct
export MED_EVAL_PIPELINE_MODE=fake
python scripts/run_server.py --host 0.0.0.0 --port 8000
```

浏览器访问 `http://服务器IP:8000/`。若服务器有防火墙，优先使用 SSH 端口转发：

```bash
ssh -L 8000:127.0.0.1:8000 tangm@服务器地址
```

`fake` 模式仍用于纯网页流程验收。当前 `real` 模式只接入 CP09 和 CP11，其他项目保留在 11 项报告中但标记为未评估，不计作零分。CP01 已暂停，不要再运行 `smoke_sam2_cp01_cp11.py` 作为当前验收入口。

### 5.1 启动 CP09/CP11 真实评分与 Qwen 点评

先在独立终端按第 3 节启动 Qwen。再检查 GPU，占用较低的物理 GPU 分配给 SAM2；以下仅以物理 GPU 4 和网页端口 57116 为示例：

```bash
conda activate video_medical
cd ~/medical_evaluation
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

export MED_EVAL_VIDEOS_DIR="$HOME/medical_evaluation/videos"
export MED_EVAL_DATA_DIR="$HOME/medical_evaluation/data"
export MED_EVAL_RUBRIC_PATH="$HOME/medical_evaluation/config/rubric.yaml"
export MED_EVAL_SAM2_CHECKPOINT_PATH="$HOME/medical_evaluation/external/sam2/checkpoints/sam2.1_hiera_large.pt"
export MED_EVAL_SAM2_MODEL_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
export MED_EVAL_SAM_DEVICE="cuda:0"
export MED_EVAL_SAMPLE_FPS=2
export MED_EVAL_ANALYSIS_WIDTH=1280
export MED_EVAL_VLM_BASE_URL="http://127.0.0.1:8001/v1"
export MED_EVAL_VLM_MODEL="Qwen/Qwen3-VL-4B-Instruct"
export MED_EVAL_PIPELINE_MODE=real

CUDA_VISIBLE_DEVICES=4 python scripts/run_server.py --host 127.0.0.1 --port 57116
```

如果 GPU 4 已占用，替换 `CUDA_VISIBLE_DEVICES=4`，不要停止其他用户进程。因为该进程只看到所选物理卡，`MED_EVAL_SAM_DEVICE=cuda:0` 保持不变。Qwen 必须运行在另一个进程和另一张 GPU 上。

本机建立端口转发：

```powershell
ssh -p 22 -L 57116:127.0.0.1:57116 tangm@10.25.64.102
```

打开 `http://127.0.0.1:57116/`，或者通过 API 创建三个任务：

```bash
curl -X POST http://127.0.0.1:57116/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"preset_id":"success"}'
curl -X POST http://127.0.0.1:57116/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"preset_id":"failure"}'
curl -X POST http://127.0.0.1:57116/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"preset_id":"clamp_failure"}'
```

响应中的 `job_id` 对应：

```text
http://127.0.0.1:57116/jobs/{job_id}
http://127.0.0.1:57116/reports/{job_id}
data/jobs/{job_id}/report.json
```

成功视频具备 CP09 支架提示和唯一 `oral_region` 框时应显示“已评估 2/11 项”。当前两个失败视频缺少完整 CP09 输入时，CP09 显示 `automatic_evaluation_input_missing` 且不参与评分，CP11 仍自动扫描绿色橡皮布，页面显示实际的“已评估 1/11 项”。不得用人工标签替代缺少的模型结果。

阶段性分数只在已运行项目内归一化，页面必须同时显示“非最终成绩”。Qwen 服务不可用或返回非法 JSON 时，`ai_commentary.source` 应为 `template_fallback`，但 Judge 的 `status`、`reason_code`、特征和阶段性分数不能变化。

## 6. 标注与验证顺序

1. 打开三个 `/annotate/{video_id}` 页面；
2. 校准各视频自己的 11 个时间段；
3. 填写真实标签和原因；
4. 为关键对象添加点/框提示；
5. 保存并检查 `data/annotations/*.json` 的审计历史；
6. 再运行真实模型一致性验证。

在 33 项人工标签完成前，只能称为流程 Demo，不能报告准确率。三条视频得到的结果只能称为“Demo 集一致率”，不能称为测试集或泛化准确率。

### 6.1 已暂停的 CP01 实验记录

以下 CP01 内容仅保留为历史实验记录，当前真实评分链路不会调用 CP01 提取器或 Judge。CP11 继续使用后文规定的标注和判定方式。

启动标注网页后访问：

```text
http://127.0.0.1:8000/annotate/success
http://127.0.0.1:8000/annotate/failure
http://127.0.0.1:8000/annotate/clamp_failure
```

`success` 的 CP01 保留一次性的 `rubber_dam` 框和 `cp01_reference` 点；每个实际执行 CP01 的视频还必须在笔正在橡皮布上标记的清晰帧提供 `marking_pen` 提示。可以紧框一次笔，也可以在同一帧笔身或笔尖内部打 1–2 个正点；不要混用框和点，不要跨帧打点，也不要把点落在手、橡皮布或黑色标记点上。若人工真值确认整个 CP01 没有执行，则不要伪造笔提示。

CP01 在完整阶段同时跟踪橡皮布和笔。至少两个有效帧检测到笔即确认执行了标记动作，不要求笔和橡皮布掩膜重叠。橡皮布内符合尺寸和形状要求、且跨帧稳定的暗点按局部黑度取最黑的 1–2 个，再选择距离固定参考点最近的点。

CP01 状态顺序为：橡皮布有效帧少于 3 帧是 `needs_review`；没有稳定检测到笔是 `incomplete`；检测到笔但没有稳定黑点是 `incorrect`；有候选后才比较 `max_mark_distance`。

CP11 全阶段从未出现橡皮布时为 `incomplete`；中途出现但末尾消失时为 `incorrect`，这两种情况都不伪造末尾提示。只有末尾仍有橡皮布时，才用 3–5 个 `rubber_dam` 正点并紧框 `nose_region`。正点必须位于绿色橡皮布内部，避开模型皮肤、牙齿、支架和画面边缘。

CP11 的支架提示复用 CP09 `rubber_dam_frame`，并从 CP09 连续跟踪至 CP11 末尾。CP11 末尾的橡皮布面积和鼻部重叠使用“原始 SAM2 橡皮布掩膜与确定性绿色像素掩膜的交集”，避免把模型皮肤计入橡皮布；证据图中的白色轮廓仅表示仍符合 CP09 支架外观的可见支架像素。没有白色轮廓且 `visible_frame_area_ratio` 接近零，表示未检出明显裸露支架。

保存基准点后执行一次校准：

```bash
python scripts/calibrate_cp01_reference.py \
  --video-id success \
  --data-dir "$HOME/medical_evaluation/data"
```

重新标注笔提示前先备份三个 JSON。只删除要重打的 CP01 `marking_pen`，不要删除 `rubber_dam`、`cp01_reference`、CP11 或其他阶段提示：

```bash
BACKUP_DIR="data/annotations/backups-cp01-pen-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BACKUP_DIR"
cp data/annotations/success.json "$BACKUP_DIR/"
cp data/annotations/failure.json "$BACKUP_DIR/"
cp data/annotations/clamp_failure.json "$BACKUP_DIR/"
```

标注后先运行时序阈值诊断；该命令只运行一次 SAM2，再在 CPU 上遍历阈值：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/calibrate_cp01_detector.py \
  --video-id success \
  --checkpoint-path external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --model-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda:0 \
  --sample-fps 2 \
  --videos-dir "$HOME/medical_evaluation/videos" \
  --data-dir "$HOME/medical_evaluation/data"
```

检查 `data/calibration/cp01_detector_sweep.json` 和 `data/calibration/cp01_detector_sweep_overlays/`。必须目视确认灰色点是接触前模板背景、红色点是接触后最黑候选、绿色点是固定参考位置，不能仅凭候选数量选阈值。

选择当前空闲 GPU（示例为物理 GPU 1）后，对每个视频联合运行：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/smoke_sam2_cp01_cp11.py \
  --video-id success \
  --checkpoint-path external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --model-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda:0 \
  --sample-fps 2 \
  --videos-dir "$HOME/medical_evaluation/videos" \
  --data-dir "$HOME/medical_evaluation/data"
```

将 `--video-id` 依次改为 `failure`、`clamp_failure`。每次只检查命令打印的新运行目录，其中应同时存在 `summary.json`、`decisions.json`、`cp_01/evidence.json`、`cp_01/overlays/` 和已执行情况下的 `cp_11/overlays/`。

CP01 最多两张证据图：黄色轮廓为橡皮布，青色为笔，红色为最黑 Top-2，绿色为固定参考点，黄线连接最终学员点和参考点。JSON 至少核对 `pen_presence_detected`、`mark_candidate_count`、`selected_mark_darkness` 和 `mark_reference_distance`。

## 7. CUDA OOM 诊断

出现 OOM 时按顺序处理：

```bash
nvidia-smi
ps -fp <PID>
```

- 确认没有选到已有任务的 GPU；
- 将 `analysis_width` 从 1280/960 降到 768；
- 将 `dense_fps` 从 5 降到 2 或 1；
- SAM 与 Qwen 分配到不同 GPU；
- 降低 vLLM `--gpu-memory-utilization` 或 `--max-model-len`；
- 保留报告 `audit.degradations`，不要隐藏降级或修改人工真值。

管线只自动降级重试一次；第二次 OOM 会使任务失败，以避免在证据不足时继续给分。

## 8. SAM2 CP09 固定口腔参考框纵向切片

该烟雾测试只让 SAM2 分割和跟踪 `rubber_dam_frame`（白色支架）。
`oral_region` 必须是一个人工框出的口腔参考区域；该框不送入 SAM2，而是在整个
CP09 阶段保持固定。系统计算每帧支架掩膜中心相对口腔参考框中心的归一化偏移，
再取阶段中位数。它不会把网页切换到真实模式，也不会覆盖
`data/annotations/`。

### 8.1 标注要求

`data/annotations/success.json` 的 CP09 时间段目前为 `175.0-195.0s`，必须包含：

- `rubber_dam_frame` 的点或框提示；
- 恰好一个 `oral_region` 框。

CP09 对阶段边界使用 `±0.5s` 的提示容差，因此当前
`oral_region@174.686926s` 可以作为固定参考框。该框的时间不会扩大 SAM2
跟踪窗口；跟踪窗口只根据原始阶段和支架提示时间确定。特征、有效帧计数和证据图
只统计原始 `175.0-195.0s` 阶段内的帧。没有口腔框、使用口腔点提示或存在多个
口腔框时，脚本会在 SAM2 推理前要求重新标注。

另开一个服务器终端，保留当前网页服务继续运行。获取实现分支：

```bash
conda activate video_medical
cd ~/medical_evaluation
git fetch origin
git switch -C codex/sam2-cp09-vertical-slice \
  --track origin/codex/sam2-cp09-vertical-slice
```

安装 Meta 官方 SAM2，并记录实际代码提交：

```bash
mkdir -p external models
test -d external/sam2/.git || git clone https://github.com/facebookresearch/sam2.git external/sam2
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -e external/sam2 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple eva-decord
python -c "import decord; print(decord.__version__)"
git -C external/sam2 rev-parse HEAD | tee models/sam2-code-commit.txt
```

当前服务器的 NVIDIA 驱动支持 CUDA 12.8，但系统 `nvcc` 是 10.1，因此用
`SAM2_BUILD_CUDA=0` 跳过可选的小区域后处理扩展。该扩展不影响本切片的视频
分割主流程。`eva-decord` 提供 SAM2 读取 MP4 所需的 `decord` 模块。

下载官方 SAM2.1 checkpoints。本切片使用 large 权重：

```bash
cd ~/medical_evaluation/external/sam2/checkpoints
./download_ckpts.sh
cd ~/medical_evaluation
sha256sum external/sam2/checkpoints/sam2.1_hiera_large.pt \
  | tee models/sam2.1_hiera_large.pt.sha256
```

重新查看当前 GPU 占用。本次验证选择物理 GPU 7；如果该卡已有任务，必须换成空闲卡，
不能终止其他用户进程：

```bash
nvidia-smi
```

运行成功视频 CP09：

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/smoke_sam2_cp09.py \
  --checkpoint-path external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --model-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda:0 \
  --sample-fps 2 \
  --videos-dir "$HOME/medical_evaluation/videos" \
  --data-dir "$HOME/medical_evaluation/data"
```

`CUDA_VISIBLE_DEVICES=7` 会让物理 GPU 7 在该进程内显示为 `cuda:0`，因此
`--device cuda:0` 是正确的。SAM2 后端不会把完整 MP4 搬入显存；它只导出当前
考核阶段在 `--sample-fps` 下的 JPEG 帧，并强制保留人工提示所在帧。若首次发生
CUDA OOM，脚本只自动重试一次，并降到最多 1 FPS。

### 8.2 预期输出

脚本会生成：

- `summary.json`：包含支架提示和口腔参考框各自的 `prompt_counts`；
- `decision.json`：使用 `frame_oral_center_offset` 判定；
- `cp_09/masks/rubber_dam_frame/*.png`：支架二值掩膜；
- 不生成 `cp_09/masks/oral_region/`；
- `cp_09/overlays/*.jpg`：最多三张显示支架掩膜、口腔参考框、两个中心及连线的
  代表帧。

`summary.json` 应包含以下特征：

- `frame_oral_center_offset`：至少 3 个有效支架帧的相对偏移中位数；
- `frame_valid_count`：支架非空掩膜帧数；
- `oral_reference_count`：已验证的口腔参考框数，当前必须为 `1.0`；
- `relative_offset_valid_count`：成功计算相对参考框偏移的支架帧数。

若 `relative_offset_valid_count < 3`，偏移特征必须为 `null`，Judge 必须返回
`needs_review`，不能在证据不足时给出“正确”或“错误”。

### 8.3 人工验收

检查前、中、后三张组合叠加图和支架二值掩膜。只有同时满足以下条件才接受：

- `rubber_dam_frame` 覆盖白色支架，而不是橡皮布、手或背景；
- 叠加图中的 `oral_region` 框、口腔中心、支架中心和连线位置正确；
- 三张代表帧中的支架语义保持稳定；
- 没有生成 `oral_region` 掩膜目录；
- `decision.json` 使用 `frame_oral_center_offset`，且至少有 3 个有效偏移帧。

同时记录 SAM2 代码提交、checkpoint SHA-256、实际 GPU、运行时间、三类有效帧计数、
相对偏移和任何 OOM 降级记录。当前 `0.50` 是依据成功视频人工确认结果设置的临时
Demo 阈值；仍必须运行两个失败视频检查区分度，再决定是否继续校准。在人工
验收前，网页服务继续保持：

```bash
export MED_EVAL_PIPELINE_MODE=fake
```

不能把这一次烟雾测试描述为 11 项自动评分完成，也不能把成功视频单样本结果描述为准确率。
