# 论文骨架（工作题目）

**Reading Decisions out of a Masked Language Model: Readout Position, Candidate Topology, and Objective in a 150M Typed-Decision Encoder**

一句话主张：把预训练编码器变成「state + question + candidates → 校准分布」的决策模型时，起决定作用的是读出位置，而不是损失函数或额外结构；数据能买到分布内与任务级迁移，买不到需要世界知识的推理。

## 摘要要点
- 统一原语：noul / choice / score 全部是 P(candidate | state, question, candidate set)，一次前向，任意候选数，不生成文本。
- 发现 1：[MASK] 标记 + span 池化读出，比新增标记读出高 28 个点（58.6 → 86.2，6,000 题）；机制解释（MLM 只把 [MASK] 训练成上下文汇总）；跨底座验证（DeBERTa-v3-base）。
- 发现 2：proper scoring rule 家族等价、后验温度 ≈1、置信头无增益；唯一有效应的是换序一致性正则，且只作用于一致性。
- 发现 3：候选拓扑（Exp 016）。顺序排布的候选对候选数外推差（−30 点）；把每个候选块绑定到同一位置并用集合掩码（pointwise / set），一次前向在 K=77/150 上比顺序排布高 19.2 个点（CI [+17.6, +20.8]），换序一致率严格 1.000（结构保证，取代 perm-KL），分布内非劣；候选互看（set）在分布内 +0.5 点、在外推上 −1.9 点，符合闭世界 IIA 的预测。训练时见大 K 和推理端分块锦标赛是次一级的补救，二者不叠加。
- 数据边界：定向数据把 typed-decisions 从 36% 提到 76%（专家阶段），22 个未见法律规则任务 82%；六个数据版本在 JevBench hard 层 33～35%，同尺寸开源簇 34～40%，给出 150M 的经验上限。
- 开源：权重、数据生成器、评测协议（噪声下限 ECE、AURC、配对 bootstrap、控制块、污染检查）。

## 1 引言
- 背景：System-One / typed decision 接口（Jev 及社区复现）；为什么小编码器值得研究（延迟、成本、校准）。
- 贡献列表（上面五条），明确不主张架构新颖性，明确不与 Jev 比较智能上限。

## 2 相关工作
- NLI 零样本、TARS、GLiClass、label-embedding 双塔、cross-encoder 重排；
- 开源 Jev 复现：verdict、Laya、kev、SemIf、NanoJev（各自读出方式）；
- 校准与选择性预测：proper scoring rules、温度缩放、AURC、ECE 小样本偏差。

## 3 模型
- 3.1 统一原语与三种题型的编码（图 1：序列布局，[MASK] 标记位置）。
- 3.2 读出：pointer 打分 q·k/√d；等级序号 embedding；三种读出变体（joint / branch / bi-encoder）。
- 3.3 候选拓扑：seq / pointwise / set 的位置编号与注意力掩码（图 1b）；命题：位置绑定 + 集合掩码 ⇒ 输出对候选置换严格等变；pointwise ⇒ 每个候选 logit 与其他候选无关（IIA），闭世界下子集条件概率即全集重归一化。
- 3.4 目标函数：soft-CE + RPS（+ 仅 seq 拓扑需要的 perm-KL），声明均为已有成分并引用来源。

## 4 数据与评测协议
- 4.1 数据：v0.1 七个公开集；v0.2+ 规则生成（policy / multihop / temporal，程序标签）、LegalBench（任务级留出）、typed-decisions 训练集；切分按源样本；模板独立于标签；负例按构造平衡。
- 4.2 协议：每单元 ≥2,000 题、配对 bootstrap、McNemar、15 等质量桶 ECE 与噪声下限、AURC、cov@risk；控制块（无 state、打乱 state、候选换序、留出模板）；13-gram 污染检查。
- 4.3 外部基准：typed-decisions（专家 / 通才模式分开）、JevBench 公开题（并说明保留题由维护者运行）。

## 5 实验
- 5.1 读出位置（表 2：2×2 标记 × span，3 seeds，ModernBERT；DeBERTa 复核）→ 已有：Exp 017（[MASK] 0.865±0.002 vs 新增标记 0.600±0.015；span 池化无影响）；Exp 018 进行中。
- 5.2 读出结构（表 3：joint / branch / bi-encoder，含训练成本）→ 已有：Exp 001。
- 5.3 目标函数（表 4：五个 arm 的准确率、NLL、ECE、AURC、一致性、温度）→ 已有：Exp 003。
- 5.4 候选拓扑与候选数（表 5：seq / pointwise / set 在 K≥77 全集与分布内的配对比较；一次前向 / 分块 / 锦标赛 × 训练 K 上限）→ 已有：Exp 016（单 seed，待 3 seeds）、Stage 1、Exp 006、010。
- 5.5 数据能买到什么（图 2：JevBench hard 与 typed-decisions 随数据版本的曲线；表 6：LegalBench 留出任务）→ 已有：Exp 005–015。
- 5.6 外部对照（表 7：typed-decisions 专家模式 vs Jev / verdict；JevBench 公开题与官方分）→ 待：官方分。

## 6 分析
- 读出位置的机制：[MASK] 位置表示与候选 span 的相似度分析；新增标记的学习曲线（Mac 上 3,200 条 = v1 的 16,000 条）；新增标记 seed 方差大 10 倍（Exp 017），说明差距来自初始化而非容量。
- 哪些 hard 题型失败及为什么（按 family 的准确率与置信度）。
- 校准来自哪里：Brier 分解（可靠性 / 分辨率）。

## 7 局限与未来
- 150M 的知识上限；有序打分弱；大标签集；英文；第二期：相干性约束、位置绑定候选块、不对称块注意力、同参数量编码器 vs 解码器（需 24GB）。

## 图表清单
图 1 序列布局与读出；图 2 数据版本曲线；图 3 风险覆盖曲线（joint / bi-encoder / branch）；表 1 数据集与许可；表 2–7 如上；附录：控制块全表、合成数据模板、污染检查、JevBench 按题型。

## 待补实验（进行中）
- ~~Exp 017~~ 完成，见 RESULTS_exp005.md。
- Exp 018：DeBERTa-v3-base 上 [MASK]+span 与新标记两种读出。
- Exp 016 的 3 seeds 复核（pointwise / set / seq），以及 pointwise / set 的 typed-decisions 专家阶段（选项经设计时互看是否必要）。
- JevBench 官方综合分（issue #39 排队中）。
