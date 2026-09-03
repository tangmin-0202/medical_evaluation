# SAM2 至 SAM3 项目汇报材料 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份约 4–6 页、可直接用于阶段性项目汇报的 Word 文档，准确呈现已完成成果、SAM2 瓶颈和 SAM3 迁移计划。

**Architecture:** 以当前 `AGENTS.md`、已确认的三视频结果和相关设计文档为唯一事实来源。使用一个工作区内的 Python 构建脚本生成 DOCX，采用 `standard_business_brief` 版式与 `editorial_cover` 首页面板，并通过 LibreOffice 渲染为逐页 PNG 完成视觉检查。

**Tech Stack:** bundled Python、python-docx、Word OOXML、LibreOffice DOCX renderer

---

### Task 1: 整理汇报正文事实

**Files:**
- Read: `AGENTS.md`
- Read: `docs/superpowers/specs/2026-09-03-sam2-to-sam3-project-report-design.md`
- Read: `docs/superpowers/specs/2026-09-02-cp09-cp11-scoring-llm-design.md`

- [ ] **Step 1: 提取已确认信息**

只采用已确认的流程、CP09/CP11 算法、三个任务结果、SAM2 问题和 SAM3 未验证状态。明确写出“2/11 项”“阶段性结果”“三个样例不代表泛化能力”。

- [ ] **Step 2: 完成内容边界检查**

确认正文不包含运行命令、源码、长参数列表、虚构准确率，以及“已经完成 SAM3 接入”等未发生事项。

### Task 2: 生成 Word 汇报材料

**Files:**
- Create: `scripts/build_sam2_to_sam3_project_report.py`
- Create: `artifacts/Medical_Evaluation_SAM2_to_SAM3_Project_Report.docx`

- [ ] **Step 1: 标记文档创建操作**

使用 bundled Node 运行：

```powershell
node container_tools/mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 1 --output-format docx
```

预期：命令成功退出。

- [ ] **Step 2: 编写 DOCX 构建脚本**

脚本应创建封面、六个正文部分、一张端到端流程图、一张三视频结果表、一张 SAM2/SAM3 对比表和简短结论。页脚包含“Medical Evaluation | 阶段性项目汇报”和动态页码。

- [ ] **Step 3: 运行构建脚本**

```powershell
<bundled-python> scripts/build_sam2_to_sam3_project_report.py
```

预期：生成 `artifacts/Medical_Evaluation_SAM2_to_SAM3_Project_Report.docx`，文件大小大于 0。

- [ ] **Step 4: 结构审计**

用 python-docx 重新打开成品，检查标题、章节数量、两个表格、流程图、页眉页脚和段落内容；扫描 `TODO`、`TBD`、内部工具标记和错误的“准确率”表述。

### Task 3: 渲染和逐页检查

**Files:**
- Read: `artifacts/Medical_Evaluation_SAM2_to_SAM3_Project_Report.docx`
- Create: `artifacts/.qa/sam2-to-sam3-report/page-*.png`

- [ ] **Step 1: 渲染 DOCX**

```powershell
<bundled-python> <documents-skill>/render_docx.py artifacts/Medical_Evaluation_SAM2_to_SAM3_Project_Report.docx --output_dir artifacts/.qa/sam2-to-sam3-report
```

预期：生成 4–6 张页面 PNG。

- [ ] **Step 2: 逐页目视检查**

逐张打开页面 PNG，检查中文字体、标题层级、流程箭头、表格换行、页码、裁切、重叠和异常空白。

- [ ] **Step 3: 修复并重新渲染**

如发现任何缺陷，修改构建脚本并重新生成、渲染，直到全部页面清晰无误。

- [ ] **Step 4: 最终交付检查**

确认只向用户交付 DOCX，不交付内部构建脚本或 QA PNG；保留现有用户文件，不加入无关未跟踪文件。
