# 发布候选 v0.1.0-rc1（冻结于 2026-09-23）

## 模型

- 架构：ModernBERT-base 编码器（149.6M）+ pointer 决策头（<1M），[MASK] 候选标记 + span 池化，一次前向输出任意候选集的分布。
- 底座初始化：knowledgator/gliclass-modern-base-v3.0 的编码器权重（Apache-2.0），词表 50,370。
- 训练：Stage A = Exp 010（v0.5 混合，186k 样本，1 epoch，1024 token state）；Stage B = Exp 014（typed-decisions 训练集 5,400 条，8 epoch，lr 5e-6 / 2e-5，第 1,350 步按校准集 NLL 选点）。
- 候选发布 checkpoint：`runs/exp014_typed_specialist_long/best.pt`（sha256 前 16 位 `37618128057ac54f`）；通用版（未做专家阶段）：`runs/exp010_fullk_v05/best.pt`（sha256 前 16 位 `d9877b493f104874`）。主机上冻结副本：runs/release_v0.1.0-rc1_specialist、runs/release_v0.1.0-rc1_general。
- 损失：soft-CE + RPS 1.0 + perm-KL 0.3；后验温度 ≈1，不做温度缩放。

## 冻结指标（单 seed）

| 评测 | 结果 | 参照 |
|---|---|---|
| typed-decisions 测试集 2,000 题（专家模式，Exp 014） | 准确率 0.763，Brier 0.063，NLL 0.882 | Jev 0.727 / 0.148；verdict-2.0 0.771 / 0.064 |
| JevBench 公开题 231（Exp 008 通用版） | easy 100%，standard 65.3%，hard 35.1% | Laya hard 34.1%；verdict 38.2%；Jev 74.1% |
| 全标签集（Exp 010 通用版，一次前向） | clinc150 0.865（cov@5% 0.82）；banking77 0.758（0.51） | — |
| v0.1 分布内测试集（Exp 010） | 87.5% | Stage 1 88.7% |

## 外发前必须补齐

1. 已完成：Exp 014 三 seed 准确率均为 76.3%（seed 方差 <0.1 点），Brier 0.0627 ± 0.0003；测试集抽样区间 [74.4%, 78.2%] 覆盖 verdict 77.1%，措辞用「齐平」。
2. 模型卡：数据来源与许可（AG News 仅限研究用途；HotpotQA CC-BY-SA-4.0；LegalBench CC-BY-4.0；typed-decisions 见其数据卡；合成数据 Apache-2.0 随生成器发布），污染检查结果，专家 / 通才两种模式分开报告。
3. 措辞：基于公开资料的数学推导与小模型实验得到的开放设计；不探测、不调用、不蒸馏 Jev；Jev 数字全部引自第三方。
4. 由用户执行：上传 Hugging Face（权重 + tokenizer + config），向 fstandhartinger/jevbench 提 PR（适配器 integrations/jevbench/tde_local.py）。

## 已知限制

- JevBench hard 层 35%：multi_hop、trap、temporal 类题仍接近随机；150M 编码器的知识上限。
- 全标签集 banking77 5% 错误率下只有 51% 可自动执行。
- 有序打分（score 原语）准确率 52%～72%，校准最差的切片。
- typed-decisions 的 ECE 0.14：对教师软标签的 argmax 定义过度自信，以 Brier / NLL 为准。
