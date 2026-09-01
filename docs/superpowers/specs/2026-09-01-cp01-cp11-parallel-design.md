# CP01 与 CP11 并行纵向切片设计

## 已确认修订（覆盖下文冲突的早期 CP01 描述）

- CP01 的 36 牙标准点只在 `success` 基准视频上人工标注一次，对象名为 `cp01_reference`；与同帧 `rubber_dam` 框换算为橡皮布局部坐标后持久化到 `data/calibration/cp01_reference.json`。
- 学员实际打出的点不由人工标注。系统使用 SAM2 跟踪 CP01 全阶段橡皮布，在掩膜内检测小面积、近圆形、跨帧稳定的暗点。
- 任何点首次出现时均不立即判定。必须等 CP01 阶段结束后再处理最终稳定点集合：1 点时直接作为打孔点；2 点且恰有 1 点位于橡皮布 15% 角区时，角落点视为辅助点、另一点为打孔点；2 点均在角落、均不在角落或超过 2 点时进入人工复核。时间顺序不参与身份判断，也不得按“离标准点最近”选择。
- 学员视频的 CP01 只需一个 `rubber_dam` 框。清晰有效阶段没有稳定点为 `incomplete`；候选歧义或画面不足为 `needs_review`。
- CP01 与 CP11 使用一个联合 smoke 入口，共用一次 SAM2 加载，并在同一运行目录输出 `summary.json`、`decisions.json`、`cp_01/` 和 `cp_11/`。

## 目标

同时实现 CP01 和 CP11 的真实特征提取、确定性判定、证据图与三视频 smoke。两个考核点独立实现并验证，共用现有视频采样、SAM2 后端和运行产物结构。

## CP01：橡皮布局部相对坐标

36 牙对应的是橡皮布自身坐标系中的固定相对位置，不是视频画面里的固定像素位置。橡皮布在画面中平移或缩放后，正确相对位置保持不变。

每个视频在 CP01 操作完成且标记清晰的同一帧标注：

- `rubber_dam`：一个框，覆盖当时展开的橡皮布；
- `mark`：一个正点，位于学员实际标记的中心。

每个视频不标 `target_mark_reference`。标准位置 `(reference_u, reference_v)` 由专家标定一次并写入 rubric，所有视频共同使用。

设橡皮布框为 `(x1, y1, x2, y2)`，实际标记为 `(x, y)`：

```text
mark_u = (x - x1) / (x2 - x1)
mark_v = (y - y1) / (y2 - y1)
mark_reference_distance = sqrt((mark_u - reference_u)^2 + (mark_v - reference_v)^2)
```

- 距离不超过 `max_mark_distance`：`correct`；
- 距离超过阈值：`incorrect`；
- 框无效、标记在框外或提示缺失：`needs_review`。

证据图显示橡皮布框、实际点、标准点、连线和距离。当前三个 Demo 视频使用方向一致、近似水平展开的框；明显旋转或透视变化留给后续四角局部坐标升级。

## CP11：撑开覆盖支架且鼻部无遮挡

正确完成后，绿色橡皮布被撑开并覆盖支架，所以末尾可能检测不到支架。CP11 先扫描整个阶段判断操作是否发生，再分析最后约 3 秒判断最终状态是否正确。全阶段默认按 1 FPS 检查橡皮布是否出现；末尾窗口默认按 2 FPS 分析，并至少需要 3 个有效帧。

在 CP11 末尾清晰帧标注：

- `rubber_dam`：3–5 个分散正点；必要时增加负点排除皮肤、手套和背景；
- `nose_region`：一个紧框，只覆盖应保持无遮挡的可见鼻部区域。

CP11 不要求在末尾给被覆盖的支架打点。支架候选区域复用 CP09 的 `rubber_dam_frame` 提示和外观参考。末尾只统计候选区域内仍与 CP09 可见支架外观匹配的像素；已经变成绿色橡皮布的区域不计为可见支架。这样可以避免把 SAM2 在遮挡后的持续预测误当作裸露支架。

人工标注为未执行的样例不要求伪造 `rubber_dam` 提示。全阶段橡皮布存在性由确定性的绿色橡皮布外观检查产生；清晰画面中橡皮布面积为零是“未出现”的有效证据，不是分割失败。

全阶段先计算：

```text
dam_stage_presence_ratio =
    confidently_visible_rubber_dam_frames / readable_stage_frames
```

末尾每个有效帧再计算：

```text
dam_area_ratio = rubber_dam_mask_area / frame_area
nose_overlap = area(rubber_dam_mask intersect nose_region) / area(nose_region)
visible_frame_area_ratio = visible_rubber_dam_frame_area / frame_area
```

阶段特征取有效末尾帧的中位数。小于 `max_visible_frame_area_ratio` 在规则含义上等同于“检测不到明显支架”，小的非零上限用于容忍分割噪声。

“有效帧”是能够正常读取且画面足以判断对象存在性的帧，不等同于正确帧。清晰看到没有橡皮布或没有支架仍是有效证据，相应面积记为零。只有画面损坏、严重无关遮挡或存在性检测不可信时才排除该帧。

首先根据全阶段证据判断是否执行：

```text
dam_stage_presence_ratio < min_stage_dam_presence_ratio
→ incomplete
```

只在确认阶段内出现过橡皮布后，以下条件全部满足时为 `correct`：

```text
dam_area_ratio >= min_dam_area_ratio
nose_overlap <= max_nose_overlap
visible_frame_area_ratio <= max_visible_frame_area_ratio
```

阶段内出现过橡皮布，但任一可靠末尾特征违反阈值时为 `incorrect`。只有确认阶段内出现过橡皮布后才要求鼻部框；画面损坏、存在性检测不可信、鼻部框缺失或有效末尾帧不足时为 `needs_review`。

证据记录全阶段橡皮布存在比例；另最多保存 3 张代表性末尾帧，显示橡皮布掩膜、鼻部框、实际可见支架和三个末尾比例。

## 校准、并行边界和验证

CP01 标准相对位置由专家确认的标准样例标定一次。`min_stage_dam_presence_ratio`、`min_dam_area_ratio`、`max_nose_overlap`、`max_visible_frame_area_ratio` 和其他阈值全部写入 `config/rubric.yaml`。实现后先输出三个视频的原始特征和证据图，再依据各 CP 的人工真值选择 Demo 阈值；缺少负例的阈值须明确标记为工程初值。

CP01 与 CP11 使用独立提取器、测试和 smoke 参数。公共代码只承载无 CP 语义的采样、掩膜统计和证据写出。用户标注 JSON 不加入 Git，覆盖或批量修改前必须备份并保留审计。

每个切片均须完成针对性测试、完整 pytest、Ruff、Git 提交、bundle 同步、服务器完整测试、真实 GPU smoke 和证据图目视检查。三个视频分别留下 summary、decision、运行目录和人工结论。
