# 模型总表（2026-10-13）

## 1. 各模型的基座与训练方式

四个模型都是**监督微调**：输入是 (state, 问题, 候选集)，监督信号是候选上的目标分布 y，没有强化学习，没有从 Jev 蒸馏。差别在基座、哪些参数在训、目标函数和标签来源。

### tde-general-v0.1（150M）
- 基座：ModernBERT-base 架构，权重取自 GLiClass-base 的编码器（词表 50,370）。
- 可训练参数：全部（编码器 149.6M + pointer 头 <1M），全参微调。
- 结构：序列 `[CLS] state [SEP] 问题 [SEP] [MASK] 候选1 [MASK] 候选2 … [MASK]`，每个候选从它前面的 [MASK] 隐状态（加候选 span 均值）读出，与末尾 [MASK] 做点积得到 K 个 logit，softmax 成分布。
- 目标函数：soft 交叉熵 + 1.0 × RPS（有序题）+ 0.3 × 换序 KL（同题打乱候选再前向一次，约束两次分布一致）。
- 标签来源：公开数据集的金标签改写成 one-hot（有人工投票的用投票分布）；规则世界由程序算出；typed-decisions 用其教师软标签。
- 超参：1 epoch，micro-batch 2 × 累积 16，lr 骨干 2e-5 / 头 1e-4，layer decay 0.9，bf16 autocast，梯度检查点，state 上限 1,024 token，每数据集 ≤15,000 条，共 18.6 万条。
- 算力：RTX 4060 8GB，约 2.7 小时。

### tde-typed-specialist-v0.1（150M）
- 起点：上面的 checkpoint。
- 训练：只用 typed-decisions 训练集 5,400 条，8 epoch，lr 减半（5e-6 / 2e-5），按校准集 NLL 在第 1,350 步选点；其余同上。3 个 seed 准确率完全一致。

### tde-general-large-v0.1（395M）
- 基座：ModernBERT-large 架构，权重取自 GLiClass-large 的编码器。
- 可训练参数：全部，全参微调。
- 结构：同 150M，但候选用 **set 拓扑**：每个候选块位置编号从同一位置起算，注意力掩码让候选只看前缀和自己，候选标记之间互看；输出对候选置换严格等变，换序 KL 恒为零，不再使用。
- 目标函数：soft 交叉熵 + 1.0 × RPS。
- 标签来源：同 150M，另加 synth_rubric（带描述候选 + 指令规则，程序标签）。
- 超参：1 epoch，token 预算组批（每批 ≤64 行且行数 × 最长行 ≤ 20,000 token），不开梯度检查点，lr 同上，state 上限 1,024，共 18.9 万条。
- 算力：RTX 6000 Ada 48GB，19 分钟。

### tde-qwen3.5-4b-v0.1（4B）
- 基座：Qwen/Qwen3.5-4B 指令版（revision 851bf6e），bf16，冻结。
- 可训练参数：LoRA，r = 64，α = 128，挂在每一个线性层（注意力和 MLP），fp32 适配器，1.3 亿参数；训完合并进 bf16 权重。
- 结构：decider 的读出：把 state、问题、选项渲染成一段文本，选项标 (A)(B)(C)…，在答案位取各字母 token 的 logit 做 softmax；score 型每档单独问一次 yes/no 再归一化。零生成。
- 目标函数：交叉熵（one-hot 或程序标签），decider 训练器的损失，无 RPS、无换序项。
- 标签来源：decider 公开混合（95 个公开集金标签、其 27B 教师写的题并经双答一致过滤、rules 生成器程序标签）+ LegalBench 金标签 + typed-decisions 教师软标签取 argmax + rubric 程序标签。
- 超参：1 epoch，lr 1e-4，warmup 50，cosine 衰减，micro-batch 8,192 token × 累积 4，长批（>3,072 token）开梯度检查点，共 51.8 万条、1.82 亿 token。
- 温度：训完在自有校准集（1,504 行）按题型用 NLL 拟合：choice 0.94、noul 0.86、score 1.0（样本不足保持 1）。
- 算力：H100 80GB，5.6 小时。

## 2. 同一评测设置下的基线与结果

### 2.1 JevBench 公开题（231 = easy 48 / standard 72 / hard 111；每题算对错，不加权；本机跑，原生概率，无温度，除注明外）

| 系统 | easy | standard | hard | 合计 | 备注 |
|---|---|---|---|---|---|
| decider-4b v2.1（第三方，Qwen3.5-4B-Base 全参 + LoRA） | 48 | 71 | 73 | 192 | 其引擎，其温度 |
| Qwen3.5-4B 原版（零训练，字母读出） | 48 | 64 | 71 | 183 | decider 引擎，chat 布局，T=1 |
| **tde-qwen3.5-4b-v0.1** | 48 | 70 | 66 | 184 | identity check：PyPI decider-ai[serve] 1.5.0 + Hub 权重 + 原版 typesafe 适配器 |
| **tde-general-large-v0.1**（395M） | 48 | 56 | 40 | 144 | tde-serve，typesafe 适配器 |
| **tde-general-v0.1**（150M） | 48 | 44 | 38 | 130 | 同上 |
| Laya（第三方，421M） | — | — | 34.1%（含保留题） | 公开 58.4% | 榜单公布值 |
| openJev-verdict 1.4（第三方，150M） | — | — | 37.7% | 57.6% | 榜单公布值 |

### 2.2 typed-decisions 公开测试集（2,000 题，LLM 写的自然语言决策；noul 600 / choice 600 / score 800）

| 系统 | 模式 | 准确率 | Brier | ECE |
|---|---|---|---|---|
| decider-4b v2.1 | wire，T=1 | 0.681 | 0.148 | 0.020 |
| Qwen3.5-4B 原版 | wire，T=1 | 0.583 | 0.272 | 0.183 |
| **tde-qwen3.5-4b-v0.1** | wire，T=1 | 0.773 | 0.115 | 0.019 |
| **tde-typed-specialist-v0.1** | 专家模式 | 0.763 | 0.063 | 0.14 |
| **tde-general-large-v0.1** | 混合模式 | 0.695 | 0.094 | — |
| **tde-general-v0.1** | 混合模式 | 0.666 | 0.109 | — |
| 数据卡 ModernBERT-base 专家基线（第三方） | 专家 | 0.646 | 0.119 | — |
| openJev-verdict-2.0（第三方自报） | 专家 | 0.771 | 0.064 | 0.014 |
| Jev 1.13（数据卡） | 通才 | 0.727 | 0.148 | 0.144 |

口径：准确率 = argmax 命中教师软标签的 argmax；Brier 对整个分布；"混合模式" = 训练集作为 13 路混合之一，"专家模式" = 额外在该训练集上微调。注意 4B 与两个编码器的训练混合都含该数据集的训练集（6k），因此此表对它们是"同一生成器的未见题"，不是分布外。

### 2.3 全标签集意图路由（一次前向，K = 全部标签；准确率 / 5% 错误率下的覆盖率）

| 系统 | banking77（K=77） | clinc150（K=150） |
|---|---|---|
| tde-general-v0.1 | 0.758 / 0.51 | 0.865 / 0.82 |
| tde-general-large-v0.1 | 0.853 / 0.74 | 0.928 / 0.96 |

## 3. 分数聚合口径

- **JevBench（官方）**：v1.7 起每系统 1,500 题（1,200 密封 + 300 公开）；智力 = 机会校正的分层加权准确率并按公开-密封差距调整；校准 = 对概率分布的评分；开源榜头条 Capability = (智力 + 校准)/2，仅当成本 ≤ 0.0646 美元/千题且调整后 p50 ≤ 1.23 s 才参与排名；Composite = 四轴加权调和平均 × 低于 50 的平方惩罚。我们的三个提交均未出官方分。
- **本机 JevBench 公开题**：231 题逐题对错计数，不加权，不含保留题与密封题，只用于"没训坏"的判断。
- **本仓库测试集**：每报告单元 ≥2,000 题；配对 bootstrap 10,000 次按源样本重采样；ECE 15 等质量桶 + 噪声下限；AURC；cov@risk。

## 4. 训练任务覆盖范围

| 任务类型 | 150M / 395M | 4B |
|---|---|---|
| 意图 / 主题分类（banking77、clinc150、ag_news） | 是 | 是（decider 混合内） |
| 情感 / 情绪（sst5、go_emotions） | 是 | 是 |
| NLI / 布尔问答（snli、boolq） | 是 | 是（含 MNLI 等更多） |
| 规则生成：策略核对、多跳 JSON、时间数值 | 是 | 否（实测在 4B 上损害 hard 层，剔除） |
| 带描述候选 + 指令规则（rubric：adequacy / intent / ordinal / routing / policy） | 395M 是，150M 否 | 是（4k） |
| LegalBench 规则应用（118 任务，22 任务级留出） | 是 | 是（8k） |
| typed-decisions 训练集 | 是 | 是 |
| decider 的 rules 生成器、教师自然语言题、HelpSteer 偏好、仇恨言论分级、客服路由等 174 个任务 | 否 | 是 |
| 多模态 / 多语言 | 否 | 否 |

## 5. 时延与成本（同机对比，一次一题，loopback）

| 系统 | 硬件 | p50 | p95 | 说明 |
|---|---|---|---|---|
| tde-qwen3.5-4b-v0.1 | H100，decider-ai 引擎（CUDA graph） | 15 ms | 92 ms | identity check 实测 |
| decider-4b v2.1 | 同上 | 28 ms | — | 同机实测 |
| tde-general-large-v0.1 | H100，tde-serve（无 CUDA graph） | 62 ms | 70 ms | identity check 实测 |
| tde-general-v0.1 | Apple M5 Pro，MPS | 未单独计时 | — | — |

成本：全部自托管，未做提供商定价；按榜单对 4B 档的估价约 0.02 美元/千题，150M 到 400M 编码器档约 0.003 到 0.007 美元/千题（Laya、verdict 的公布值）。同等质量下的对比：在 typed-decisions 上 395M 编码器（0.695）与 4B（0.773）相差 8 个点，推理成本相差约一个数量级；JevBench hard 层上两者差 26 题，编码器的成本优势抵不过。
