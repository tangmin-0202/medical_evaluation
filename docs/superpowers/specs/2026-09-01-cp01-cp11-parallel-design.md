# CP01 与 CP11 并行纵向切片设计

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

正确完成后，绿色橡皮布被撑开并覆盖支架，所以末尾可能检测不到支架。只分析 CP11 最后约 3 秒，默认 2 FPS，并至少需要 3 个有效帧。

在 CP11 末尾清晰帧标注：

- `rubber_dam`：3–5 个分散正点；必要时增加负点排除皮肤、手套和背景；
- `nose_region`：一个紧框，只覆盖应保持无遮挡的可见鼻部区域。

CP11 不要求在末尾给被覆盖的支架打点。支架候选区域复用 CP09 的 `rubber_dam_frame` 提示和外观参考。末尾只统计候选区域内仍与 CP09 可见支架外观匹配的像素；已经变成绿色橡皮布的区域不计为可见支架。这样可以避免把 SAM2 在遮挡后的持续预测误当作裸露支架。

每个有效末尾帧计算：

```text
dam_area_ratio = rubber_dam_mask_area / frame_area
nose_overlap = area(rubber_dam_mask intersect nose_region) / area(nose_region)
visible_frame_area_ratio = visible_rubber_dam_frame_area / frame_area
```

阶段特征取有效末尾帧的中位数。小于 `max_visible_frame_area_ratio` 在规则含义上等同于“检测不到明显支架”，小的非零上限用于容忍分割噪声。

以下条件全部满足时为 `correct`：

```text
dam_area_ratio >= min_dam_area_ratio
nose_overlap <= max_nose_overlap
visible_frame_area_ratio <= max_visible_frame_area_ratio
```

任一可靠特征违反阈值时为 `incorrect`。橡皮布分割失败、鼻部框缺失或有效帧不足时为 `needs_review`。

最多保存 3 张代表性末尾帧，显示橡皮布掩膜、鼻部框、实际可见支架和三个比例。

## 校准、并行边界和验证

CP01 标准相对位置由专家确认的标准样例标定一次。所有阈值写入 `config/rubric.yaml`。实现后先输出三个视频的原始特征和证据图，再依据各 CP 的人工真值选择 Demo 阈值；缺少负例的阈值须明确标记为工程初值。

CP01 与 CP11 使用独立提取器、测试和 smoke 参数。公共代码只承载无 CP 语义的采样、掩膜统计和证据写出。用户标注 JSON 不加入 Git，覆盖或批量修改前必须备份并保留审计。

每个切片均须完成针对性测试、完整 pytest、Ruff、Git 提交、bundle 同步、服务器完整测试、真实 GPU smoke 和证据图目视检查。三个视频分别留下 summary、decision、运行目录和人工结论。
