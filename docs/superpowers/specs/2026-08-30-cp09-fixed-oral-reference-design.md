# CP09 固定口腔参考框设计

## 目标

CP09 评价“橡皮障支架是否相对整个口腔居中”。SAM2 只分割和跟踪
`rubber_dam_frame`；人工标注的一个 `oral_region` 框不再送入 SAM2，而是作为整个
CP09 时间段内固定的口腔空间参考。这样避免橡皮障布遮挡口腔后，SAM2 把
`oral_region` 错分成支架或橡皮障布。

本设计不使用目标牙齿作为 CP09 参考。目标牙齿是否选择正确由前序考核点评价，
避免操作者围绕错误牙齿安装支架后仍通过“居中”检查。

## 已确认的范围

- 复用现有 `oral_region` `BoxPrompt`，不修改 JSON 标注格式。
- Demo 中每个视频的 CP09 必须且只能有一个位于阶段边界容差内的 `oral_region` 框。
- 该框的归一化坐标在整个 CP09 时间段内保持固定。
- SAM2 只接收 `rubber_dam_frame` 提示并只输出支架掩膜。
- 暂不实现多个口腔框插值、口腔关键点跟踪或专用口腔检测模型。

## 输入验证

CP09 提取器读取一个或多个 `rubber_dam_frame` 点/框提示送入 SAM2，并读取恰好
一个 `oral_region` 框用于几何参考。`oral_region` 沿用阶段边界容差 `0.5` 秒，
因此当前 `174.686926` 秒的框可用于从 `175.0` 秒开始的 CP09。没有框、只有点
提示或有多个框时，必须在调用 SAM2 前提示重新标注。

## 数据流

```text
rubber_dam_frame 点/框 ──> SAM2 ──> 每帧支架掩膜 ──> 支架外接框中心
oral_region 单个框 ──────────────────────────────> 固定口腔框中心和宽高
支架中心 + 口腔框中心和宽高 ──> 每帧归一化偏移 ──> 阶段中位数 ──> CP09 Judge
```

SAM2 跟踪时间窗只根据 CP09 原始时间段和支架提示时间扩展。口腔框时间只用于确认
其属于本阶段，不会额外解码或跟踪阶段外视频。

## 偏移计算

```text
(sx, sy) = 支架掩膜外接框中心的归一化坐标
(rx, ry) = ((x1 + x2) / 2, (y1 + y2) / 2)
rw       = x2 - x1
rh       = y2 - y1
offset   = sqrt(((sx - rx) / rw)^2 + ((sy - ry) / rh)^2)
```

口腔框宽高归一化可降低分辨率和拍摄缩放的影响。`frame_oral_center_offset` 是阶段
内所有有效逐帧偏移的中位数。至少需要三个有效支架掩膜；少于三帧时特征为
`None`，Judge 返回 `needs_review`。

## 输出特征

- `frame_oral_center_offset`：有效逐帧偏移中位数，证据不足时为 `None`；
- `frame_valid_count`：阶段内有效支架掩膜帧数；
- `oral_reference_count`：固定为 `1.0`；
- `relative_offset_valid_count`：成功计算相对偏移的帧数。

删除 `oral_region_valid_count` 和 `paired_valid_count`。现有
`max_oral_center_offset: 0.08` 暂时保留，但必须根据成功和失败视频的新结果校准。

## 证据文件

每次最多选择首、中、末三个有效帧。只保存
`cp_09/masks/rubber_dam_frame/<frame>.png` 和 `cp_09/overlays/<frame>.jpg`，
不再创建 `cp_09/masks/oral_region/`。叠加图显示支架掩膜、口腔参考框、两个
中心以及中心连线。证据规则名改为
`rubber_dam_frame_relative_to_oral_reference`。

## 判定与错误处理

- 偏移不大于阈值：`correct`；
- 偏移大于阈值：`incorrect`；
- 少于三个有效支架掩膜：`needs_review`；
- 口腔参考框缺失、类型错误或不唯一：推理前失败并提示重新标注。

判定文案使用“口腔参考区域”，不能声称系统成功分割了被遮挡的完整口腔。

## 测试与验收

自动化测试覆盖仅向分割器发送支架提示、口腔框不扩大跟踪窗、已知偏移计算、证据
目录、错误参考框、证据不足和 smoke summary 新字段。最后运行完整测试与 Ruff。

服务器先用成功视频重新运行 CP09 smoke，人工检查三张叠加图并记录新偏移；再对
两个失败视频运行相同流程，比较三组数据后校准阈值，不能只凭成功视频单点调参。

