# SAM3 CP09/CP11 纯文本提示迁移设计

日期：2026-09-04

状态：待用户审阅

范围：只为 CP09、CP11 增加可选 SAM3 文本后端；不重做 SAM2、Judge、评分和 Qwen 链路。

## 1. 目标与边界

现有 CP09、CP11 已由 SAM2 跑通，但对象分割依赖人工点或框。本轮验证 SAM3 文本语义发现能否减少人工提示，同时保持 VideoSegmenter、FrameMasks、特征、Judge、阈值、评分和 Qwen 权限边界不变。

阶段时间仍为人工标注；CP09 的 oral_region 框和 CP11 的 nose_region 框仍是固定几何参考。CP01、CP02–CP08、CP10 不在范围内。三个样例只用于功能与案例验证，不代表准确率或泛化能力。

## 2. 已验证基线

- Meta 源码：external/sam3，提交 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7；
- ModelScope：facebook/sam3.1@master；
- 权重：models/SAM3.1/sam3.1_multiplex.pt，3,502,755,717 字节；
- SHA-256：0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6；
- 环境：sam3_medical，Python 3.12，PyTorch 2.10.0+cu128；
- A5000 已真实加载：873,185,516 参数，约 63.68 秒，峰值已分配显存约 3.7 GiB；
- 兼容参数：use_fa3=False、use_rope_real=False、compile=False、warm_up=False、async_loading_frames=False。

以上只证明模型可加载，尚未证明视频文本提示和项目后端已跑通。

## 3. 选定方案

第一轮只使用 SAM3 文本提示：

- CP09：white U-shaped dental frame
- CP11：green dental rubber dam
- 不向 SAM3 发送 SAM2 的人工对象点、框或掩膜；
- 不做文本加视觉提示或人工修正的混合模式；
- SAM2 保留为显式可选基线，现有行为不变；
- SAM3 失败不得静默回退到 SAM2。

提示集中配置并写入审计信息，不针对单个视频调词。success 通过后冻结提示，再跑 failure 和 clamp_failure。

## 4. 组件设计

新增小型提示策略层：sam2 策略继续从标注读取对象点或框；sam3 策略生成固定文本提示，时间为跟踪区间首帧。oral_region 和 nose_region 不属于分割提示，继续从标注读取。提取器只消费策略产生的 SegmentationPrompt，不按后端名称分支。测试必须证明 SAM3 模式没有读取支架或橡皮布的人工提示。

## 5. CP09 数据流

1. 读取人工 CP09 时间范围；
2. 读取且只接受一个 oral_region 固定框，并保留固定中心；
3. 忽略 rubber_dam_frame 人工点或框；
4. CP09 第一帧发送 white U-shaped dental frame；
5. 从 CP09 第一帧跟踪支架至 CP09 结束；
6. 逐帧计算支架包围盒中心相对口腔框中心的宽高归一化距离；
7. 至少三个有效帧，取中位数得到 frame_oral_center_offset；
8. Judge 仍使用 max_oral_center_offset: 0.50；
9. 保存最多三张支架掩膜、口腔框、中心和连线 overlay。

缺少唯一口腔框、无目标、候选歧义或有效帧少于三帧时进入现有 needs_review，不猜测结果。

## 6. CP11 数据流

### 6.1 整阶段存在性

完整 CP11 继续按 1 FPS 使用 OpenCV 绿色面积规则。阶段比例低于 0.05 仍为 incomplete/rubber_dam_not_observed；中间出现但最后三秒比例低于 0.5 仍为 incorrect/rubber_dam_missing_at_end。只有末尾仍存在才运行 SAM3 精细分割。

### 6.2 最后三秒橡皮布

1. 读取且只接受一个 nose_region 固定框；
2. 不读取人工 rubber_dam 正点；
3. 最后三秒第一帧发送 green dental rubber dam；
4. 跟踪至 CP11 结束；
5. SAM3 掩膜继续与确定性绿色像素掩膜取交集；
6. 至少三个有效末尾帧，继续计算 dam_area_ratio 和 nose_overlap 中位数；
7. Judge、阈值和原因码不变。

### 6.3 支架跨阶段跟踪

CP11 的裸露支架特征使用 white U-shaped dental frame，在 CP09 第一帧发送提示并跟踪到 CP11 结束。继续用 CP09 支架 Lab 外观估计 CP11 末尾 visible_frame_area_ratio，阈值仍为 0.005。

首轮中 CP09 提取与 CP11 跨阶段提取各自建立独立 SAM3 会话；CP11 会话本身仍从 CP09 首帧开始，故判定含义不变。暂不增加跨提取器缓存。

## 7. SAM3 后端

Sam3Backend 适配真实 SAM3.1 multiplex 视频 predictor：

1. 显式加载本地 checkpoint，离线推理；
2. 每次 track 独立建立和释放会话，异常时也清理；
3. 文本提示作用于请求区间第一帧；
4. 只输出 time_range 和 sample_fps 对应的 FrameMasks；
5. 正确映射局部帧、原视频帧号和时间；
6. 若官方 API 不能限制传播区间，则创建仅覆盖请求区间的临时帧资源；
7. 布尔掩膜尺寸与原视频帧一致；
8. 初始化失败、无目标或目标不稳定必须显式暴露。

短视频 smoke 先记录真实输出字段、对象 ID、置信度和时序行为。若有候选分数，首帧选择最高置信候选并保持同一对象 ID；只有真实输出证明多个掩膜属于同一语义对象时才合并。无法消歧时进入人工复核，不增加未经验证的空间启发式。

## 8. Runtime 与输出

- sam_backend=sam2 只校验和创建 SAM2；
- sam_backend=sam3 只校验 SAM3 源码、checkpoint、tokenizer/BPE 资源和配置；
- 未知后端启动即报错；
- runtime 不再硬编码 Sam2Backend；
- SAM3 使用独立 sam3_medical，不修改 video_medical、qwen_medical；
- SAM2/SAM3 输出目录可区分，不覆盖已有掩膜、overlay、特征和报告；
- 每个任务新增独立的 segmentation_metadata.json 审计文件，记录后端、源码提交、权重哈希、提示、采样率、时间范围、候选选择、耗时和峰值显存，不改变现有报告 Schema。

## 9. 保持不变

人工阶段时间、CP11 整阶段绿色存在性规则、特征定义、中位数聚合、Judge 状态和原因码、全部阈值、阶段性评分、11 项报告、Qwen Schema 与禁止改分约束、现有 SAM2 后端均不修改。首次实验不同时修改模型、Judge、阈值、阶段和提示词。

## 10. 错误处理

- SAM3 初始化失败：明确失败并记录，不静默回退；
- 文本无目标或候选歧义：记录缺失特征并进入 needs_review；
- 有效帧少于三帧：沿用 needs_review；
- GPU 不足：运行前动态选卡，不终止其他用户进程，最多一次受控重试；
- 原视频、标注、已有证据和报告均不覆盖。

## 11. 测试与验收

采用失败测试、最小实现、回归的顺序。本地测试覆盖文本提示契约、Sam3Backend 会话/传播/帧映射/清理、time_range 与 sample_fps、runtime 后端选择、CP09/CP11 不读取人工对象提示、CP11 从 CP09 首帧跟踪，以及 Judge、评分、Qwen 回归。聚焦测试通过后运行完整测试和 Ruff。

服务器按顺序执行：动态查 GPU；短视频验证两个文本提示；验证帧号和时间；先完整运行 success 并目视 overlay；冻结提示后运行 failure、clamp_failure；比较 SAM2/SAM3 掩膜、有效帧、特征和错误帧。确认掩膜差异后才讨论阈值。

首轮通过条件：

- A5000 完成真实视频文本提示推理；
- SAM3 未使用支架或橡皮布人工点框；
- success 视频的 CP09、CP11 各有至少三个语义正确且稳定的目标掩膜；
- CP11 支架从 CP09 首帧跟踪到末尾；
- 固定口腔中心和鼻部框继续生效；
- SAM2 仍可显式选择；
- Judge、评分、Qwen 未被改写；
- 自动测试、GPU smoke、证据图目视检查通过。

## 12. 实施顺序

1. 为真实 SAM3.1 API 和文本输出补失败测试；
2. 修正 Sam3Backend 并做短视频 GPU smoke；
3. 实现提示策略和 runtime 选择；
4. 接入 CP09；
5. 接入 CP11；
6. 跑聚焦测试、完整测试和 Ruff；
7. 服务器先跑 success；
8. 目视通过后冻结提示，再跑两个失败视频；
9. 汇总 SAM2/SAM3 对照，再决定后续扩展。
