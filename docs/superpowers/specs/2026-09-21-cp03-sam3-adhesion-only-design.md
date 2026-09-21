# CP03 SAM3 单孔无粘连判定设计

## 目标与评分边界

CP03 只评价打孔后的目标孔是否存在橡皮布粘连。上下提拉动作、孔的圆度、孔径、椭圆轴比、CP01/CP02 结果均不参与评分。

确定性 Judge 只输出以下四类结果：

- `correct/criteria_satisfied`：目标孔被可靠观察，连续清晰帧均未发现与孔边相连的橡皮布薄片或残留。
- `incorrect/hole_adhesion_detected`：目标孔被可靠观察，并发现与孔边相连的橡皮布薄片、翻边或残留。
- `incomplete/hole_not_observed`：可靠扫描范围内未观察到目标孔。
- `needs_review`：SAM3 分割、孔观察、连续帧或遮挡条件不足，无法可靠判断。

圆度、面积、轴比等几何量可以用于排除噪声候选或判断画面是否清晰，但不得成为 `correct` 或 `incorrect` 的评分条件，也不得以“不够圆”作为失败原因。

## 时间窗口

先处理人工标注 CP03 阶段的最后 4 秒，按 2 FPS 取样，通常得到 8 帧。若其中没有至少 3 帧连续、无遮挡且可判断的目标孔画面，则向前扩展到 CP03 阶段最后 8 秒；不继续扫描更早画面。

最后 8 秒仍无法形成 3 帧连续可靠证据时输出 `needs_review`，不得选择更早的单张最佳画面强判。

## 视觉流程

1. `VideoSegmenter` 只加载一次 SAM3 模型，并以 `rubber_dam` 文本提示分割所选时间窗口中的绿色橡皮布。
2. 从 SAM3 橡皮布掩膜与对应原始 BGR 帧建立有效橡皮布区域，排除掩膜外部、图像边缘、手套和器械明显遮挡帧。
3. 在橡皮布内部寻找同一目标孔候选。候选必须跨连续帧保持相近位置和尺度；单帧亮点、黑色笔迹、SAM3 掩膜漏点和折痕不得直接当作孔。
4. 在原始分辨率孔 ROI 内检查孔边缘。若绿色橡皮布薄片或翻边从孔边连续伸入孔内，则记为粘连；没有此类连续连接证据则记为无粘连。
5. 汇总至少 3 个连续清晰帧。任一可靠帧出现持续粘连证据即判有粘连；全部可靠帧无粘连才判无粘连；帧间冲突且无法消除遮挡影响时输出 `needs_review`。

SAM3 负责橡皮布区域分割，OpenCV 负责在原始像素上定位孔并检查绿色连接结构。不会要求 SAM3 单独识别数像素级粘连残片。

## 特征与证据

正式提取器只向 Judge 提供最小判定特征：

- `final_scan_reliable`
- `hole_observed`
- `hole_clear_consecutive_frames`
- `hole_adhesion_free`
- `adhesion_observed_frame_count`

每次真实运行保存：原始帧、SAM3 橡皮布掩膜、孔 ROI、孔边及粘连候选叠加图、逐帧判定表、最终原因码。允许保存定位用的面积或形状调试值，但这些值不进入评分规则。

## 接入范围

- 新增正式 `Cp03FeatureExtractor`，复用当前 pipeline 中的 SAM3 `VideoSegmenter`。
- 将 `pipeline.py` 的 CP03 Judge 从旧动作 Judge 切换到 `judges/cp03_hole.py`。
- 在 `runtime.py` 中构建并注入 CP03 提取器，将 CP03 加入启用项。
- 更新 `config/rubric.yaml`，删除上下提拉、圆度和完整度评分文字，只保留“目标孔无粘连”。
- 保留 CP01 的全部源码、测试、证据和 bundle，不修改 CP01。

## 错误处理

- SAM3 未输出可靠橡皮布掩膜：`needs_review/unreliable_final_scan`。
- 可靠橡皮布画面中未发现孔：`incomplete/hole_not_observed`。
- 孔存在但不足 3 帧连续清晰证据：`needs_review/insufficient_clear_hole_frames`。
- 孔边被手或器械遮挡、帧间粘连结果冲突：`needs_review/unreliable_adhesion_observation`。

## 测试与真实验收

按测试驱动方式先修改 Judge 测试，使“不圆但无粘连”仍为 `correct`、粘连为 `incorrect/hole_adhesion_detected`。再为孔定位、连接薄片、孤立噪声、遮挡和连续帧汇总写失败测试，之后实现最小代码。

本地运行 CP03 聚焦测试、相关 pipeline/runtime 测试、完整 pytest 和 Ruff。随后同步到 `10.25.64.102`，核对 GPU5 专用 MPS 通道后，用 SAM3 对 success、failure、clamp_failure 三段真实 CP03 窗口运行；逐张目视检查保存的掩膜和 overlay。三个样本只用于 Demo 链路验证，不用于宣称准确率或泛化能力。
