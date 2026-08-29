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

```bash
git clone https://github.com/facebookresearch/sam3.git external/sam3
git -C external/sam3 rev-parse HEAD | tee models/sam3-code-commit.txt
pip install -e external/sam3
# 从 Meta 官方许可页面取得 checkpoint 后放到 models/
sha256sum models/sam3.1_multiplex.pt | tee models/sam3.1_multiplex.pt.sha256
```

SAM3 权重和代码受仓库中的 SAM License 约束，部署前阅读并保留许可证。SAM3.1 官方视频接口使用 session/request 流程，支持文本提示及点/框细化；本项目适配器把它转换为统一 `FrameMasks`。

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

当前 `fake` 模式用于网页流程验收。切换 `real` 前必须实现面向 11 个 Judge 的实际 `FeatureExtractor`、校准 rubric 阈值，并在应用创建时注入 `AnalysisPipeline`；仅安装权重后直接设置 `real` 会被应用明确拒绝，避免产生伪真实报告。

## 6. 标注与验证顺序

1. 打开三个 `/annotate/{video_id}` 页面；
2. 校准各视频自己的 11 个时间段；
3. 填写真实标签和原因；
4. 为关键对象添加点/框提示；
5. 保存并检查 `data/annotations/*.json` 的审计历史；
6. 再运行真实模型一致性验证。

在 33 项人工标签完成前，只能称为流程 Demo，不能报告准确率。三条视频得到的结果只能称为“Demo 集一致率”，不能称为测试集或泛化准确率。

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

## 8. SAM2 CP09 支架纵向切片

该烟雾测试只验证成功视频的 CP09 支架分割、证据图和居中规则，不会把网页切换到真实模式，也不会覆盖 `data/annotations/`。

另开一个服务器终端，保留当前网页服务继续运行。获取尚未合并的实现分支：

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
pip install -e external/sam2
git -C external/sam2 rev-parse HEAD | tee models/sam2-code-commit.txt
```

下载官方 SAM2.1 checkpoints。本切片使用 large 权重：

```bash
cd ~/medical_evaluation/external/sam2/checkpoints
./download_ckpts.sh
cd ~/medical_evaluation
sha256sum external/sam2/checkpoints/sam2.1_hiera_large.pt \
  | tee models/sam2.1_hiera_large.pt.sha256
```

重新查看当前 GPU 占用。下面示例选择物理 GPU 1；如果该卡已有任务，必须换成空闲卡，不能终止其他用户进程：

```bash
nvidia-smi
```

运行成功视频 CP09：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/smoke_sam2_cp09.py \
  --checkpoint-path external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --model-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda:0 \
  --sample-fps 2 \
  --videos-dir "$HOME/medical_evaluation/videos" \
  --data-dir "$HOME/medical_evaluation/data"
```

`CUDA_VISIBLE_DEVICES=1` 会让物理 GPU 1 在该进程内显示为 `cuda:0`，因此命令中的 `--device cuda:0` 是正确的。脚本会打印 `summary.json` 和叠加图目录。

至少检查 CP09 前、中、后三张叠加图，确认彩色掩膜覆盖的是橡皮障支架，而不是手、面部、橡皮布或背景。同时记录：

- SAM2 代码提交和 checkpoint SHA-256；
- 实际使用 GPU、运行时间和有效掩膜帧数；
- `frame_center_offset`、Judge 状态和任何 OOM 降级记录。

在叠加图人工验收前，网页服务继续保持：

```bash
export MED_EVAL_PIPELINE_MODE=fake
```

不能把这一次烟雾测试描述为 11 项自动评分完成，也不能把成功视频单样本结果描述为准确率。
