# CP01 标记笔出现与最黑点判定设计

## 目标

CP01 的评分对象是橡皮布上的黑色牙位标记。标记笔只用于确认学员执行过标记动作，不参与位置评分，也不要求笔掩膜与橡皮布掩膜重叠。

## 标注与跟踪

- `rubber_dam` 保持一个框提示。
- `marking_pen` 接受一个框，或同一帧 1–2 个正点。
- 没有 `marking_pen` 提示表示没有观察到标记笔；提取器只跟踪橡皮布。
- 有合法提示时，SAM2 在完整 CP01 阶段跟踪笔。至少两个有效帧得到非空笔掩膜时，`pen_presence_detected=true`。

## 黑点选择

在所有橡皮布有效帧中检测尺寸和形状合格的暗色连通域，并按橡皮布包围盒转换为相对坐标。跨至少 `min_mark_observed_frames` 个不同帧、局部位置稳定的观察组成一个候选轨迹。不再按笔接触时刻拆分背景和新增点，也不记录模板背景点。

所有稳定候选按局部黑度从深到浅排序，只保留最黑的 1–2 个。在这 1–2 个候选中，选择与保存的 CP01 固定参考点相对距离最小者作为学员标记；`mark_reference_distance <= max_mark_distance` 为正确，否则为错误。

## 状态优先级

1. 橡皮布有效帧少于 3：`needs_review`。
2. 没有稳定检测到标记笔：`incomplete`。
3. 检测到笔但没有稳定黑点：`incorrect`。
4. 有稳定黑点但无法计算距离：`needs_review`。
5. 距离超过阈值：`incorrect`；否则 `correct`。

## 输出与证据

输出 `pen_valid_frame_count`、`pen_presence_detected`、`mark_candidate_count`、最终点坐标、黑度和参考距离。证据图显示橡皮布轮廓、笔轮廓、最黑 Top 2、固定参考点以及最终距离连线。旧的 `pen_dam_overlap_ratio`、`pen_contact_frame_count`、`pen_contact_detected`、`preexisting_mark_candidate_count` 和 `new_mark_candidate_count` 不再参与 CP01。

## CP11 样例数据修正

`clamp_failure` 视频实际时长为 193.835 秒。其 CP11 人工时间段改为有效的末尾 `190.0–193.0s`，标签保持 `incomplete`，且不添加对象提示。该修改属于服务器未跟踪的人工标注数据，不提交 Git。
