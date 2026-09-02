# 橡皮障隔离术视频智能考核 Demo

本项目把 11 个考核点、逐视频时间段、SAM2.1/SAM3.1 分割接口、确定性规则评分、本地 Qwen3-VL 点评和网页报告组织成一条可审计流程。

## 当前能运行的内容

- 网页选择三个内置视频或上传视频；
- 每个视频独立调整 11 个时间段，不要求和成功视频完全同步；
- 在暂停帧上添加归一化点/框提示，作为 SAM 输入；
- 任务持久化、进度页、11 项报告结构、阶段性评分和人工复核；
- SAM2.1/SAM3.1 可切换协议、掩膜特征函数和全部 11 个规则 Judge；
- 本地 OpenAI 兼容 Qwen3-VL 客户端，失败时自动回退中文模板且不能改分。

默认 `pipeline_mode=fake`，因此无需 GPU 就能演示网页、队列、报告和人工复核。当前真实纵向链路只运行 CP09 和 CP11：具备 CP09 必需提示的成功视频显示已评估 2/11 项；缺少 CP09 输入的视频只运行 CP11，并显示实际评估数量。CP01 暂时搁置，和其他未实现项目一样不调用模型、不计作零分。当前不能把 fake 报告或阶段性分数当成完整考核成绩或模型准确率。

## Windows 本地启动网页 Demo

```powershell
cd D:\Medical_evaluation
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
$env:MED_EVAL_VIDEOS_DIR="D:\Medical_evaluation\橡皮障视频\橡皮障视频"
python scripts\run_server.py --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/`。运行测试：

```powershell
python -m pytest -q
ruff check src scripts tests
```

## 输入与输出

输入包括：一个 MP4/MOV/AVI/MKV 视频、`config/rubric.yaml`、该视频自己的 11 段时间标注，以及目标对象的点/框/掩膜或文本提示。成功视频 Excel 时间只是初始模板。

输出保存在 `data/`：

- `annotations/{video_id}.json`：时间段、标签、SAM 提示和修改审计；
- `jobs/{job_id}/job.json`：任务状态；
- `jobs/{job_id}/report.json`：11 项结论、特征、证据、评分和运行审计；
- `runs/{job_id}/`：可删除的派生掩膜和证据图。

Ubuntu GPU 部署、SAM/Qwen 安装和显存分配见 [服务器运行手册](docs/server-runbook.md)。

## 评分原则

完整考核仍计划采用 11 项等权，每项约 9.09 分。当前页面只对实际完成自动评估的 CP09/CP11 归一化显示“阶段性得分”：两项均运行时各占阶段性 50 分，只有一项运行时该项占阶段性 100 分；`needs_review` 形成阶段性区间。正式总分保持为空。Qwen 点评保存在独立字段，只解释证据和建议，不得覆盖 Judge 状态、原因码或分数。

## 尚需人工完成

两个失败视频的每项真实开始/结束时间、正确性和原因不能根据文件名推断。需要在 `/annotate/{video_id}` 页面完成 22 项人工标注，再进行“三视频 Demo 集一致率”验证；这不等同于泛化准确率。
