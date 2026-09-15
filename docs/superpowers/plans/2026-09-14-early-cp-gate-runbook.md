# CP01 / CP02 分割门槛运行说明

## 本地已准备

- CP01 模板坐标特征和独立 Judge，旧人工框路径保留但未启用。
- CP02 孔盘几何与清理独立 Judge，不覆盖历史 judge_cp02。
- scripts/run_early_cp_sam3_gate.py：CP01 模板板/橡皮布/笔，CP02 打孔器/清理器具/橡皮布；逐提示独立 SAM3 会话。
- 标注仅取 steps 时间，不读取 prompts 或 label；不使用截图红圈坐标。
- 输出对象各提示目录中的 raw/masks/overlays、summary.json；完成不代表语义正确，visual_review_status=pending，score=null。

## 服务器空闲后

1. 查询 nvidia-smi，选择无其他计算任务且可用显存至少覆盖既有约 10.5 GiB reserved 峰值并留余量的卡。
2. Windows 测试后仅提交本次 CP01/CP02 文件；不要混入暂停的 CP08 修改、bundle 和数据。
3. 推送功能分支、服务器 fast-forward；前后核对用户 rubric.py 哈希，不覆盖脏文件。
4. 在服务器项目目录执行，下方索引以实际空闲卡替换：

```bash
EARLY_CP_GPU_INDEX=实际空闲索引
env CUDA_VISIBLE_DEVICES="$EARLY_CP_GPU_INDEX" PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/tangm/miniconda3/envs/sam3_medical/bin/python scripts/run_early_cp_sam3_gate.py --video-id success --checkpoint-id cp_01 --output-dir data/runs/cp01-object-gate-first --checkpoint models/SAM3.1/sam3.1_multiplex.pt --bpe-path external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz --device cuda:0
env CUDA_VISIBLE_DEVICES="$EARLY_CP_GPU_INDEX" PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/tangm/miniconda3/envs/sam3_medical/bin/python scripts/run_early_cp_sam3_gate.py --video-id success --checkpoint-id cp_02 --output-dir data/runs/cp02-object-gate-success-first --checkpoint models/SAM3.1/sam3.1_multiplex.pt --bpe-path external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz --device cuda:0
env CUDA_VISIBLE_DEVICES="$EARLY_CP_GPU_INDEX" PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/tangm/miniconda3/envs/sam3_medical/bin/python scripts/run_early_cp_sam3_gate.py --video-id failure --checkpoint-id cp_02 --output-dir data/runs/cp02-object-gate-failure-first --checkpoint models/SAM3.1/sam3.1_multiplex.pt --bpe-path external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz --device cuda:0
```

已有非空目录会被拒绝，不覆盖旧实验。失败记录异常；逐对象打印进度，长时无输出立即检查资源与日志。
success 清理器具可缺失，不将它的分割零掩膜当整个 gate 失败；failure 重点验证清理工作端。
先目视核对原图、对象完整性、细工作端与遮挡，再写视觉复核结果；掩膜面积过门槛不代表对象或动作正确。
CPU 单元测试不能代替真实 GPU 运行。当前三个样本只能做 Demo 验证。
