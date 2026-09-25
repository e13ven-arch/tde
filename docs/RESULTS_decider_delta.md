# decider-4b 继续训练（delta 阶段）v1 —— 2026-09-25，H100 80GB

起点：`Mapika/decider-4b`（Hub 当前快照 eb5fbdfc，decider_config 版本 4b-v2.1，Apache-2.0）。训练：decider 自带 `decider.train`（全参数 AdamW，bf16，lr 8e-6，warmup 150，8192 token × 4 累积，1 epoch），混合 8.4 万条：我们 v0.7 的规则世界 3.6 万（policy / multihop / temporal / rubric）、LegalBench 8 千、typed-decisions 6 千、分类集回放 1.4 万，加 decider 自带 rules 生成器 2 万回放。552 步，1,520 万 token，装好 causal_conv1d 内核后 11,000 token/s，训练 30 分钟。自建留出集：rubric 0.96、temporal 0.97、legal 0.88（step 552）。

## JevBench 公开题（231，decider 自带服务 + 原版 typesafe 适配器）

| 模型 | easy | standard | hard | 合计 |
|---|---|---|---|---|
| decider-4b v2.1（起点） | 48/48 | 71/72 | 73/111 | 192 |
| delta v1（本次） | 48/48 | 70/72 | 72/111 | 190 |

hard 层按题型（起点 → 本次）：temporal_numeric 5 → 3 /15，tradeoff 2 → 3 /6，judge_hard 9 → 8 /17，long_policy 10 → 9 /19，ambiguous 5 → 6 /7，probability 8 → 9 /10，其余不变。配对翻转：修好 4 题，弄坏 5 题。p50 15 ms。

## 结论

- 没有可测的变化（±9 点区间内的 −2），我们的规则生成数据对这个 4B 模型的公开题零迁移。temporal_numeric 反而掉 2 题，尽管 synth_temporal 自身测试 97%。这与编码器上的结论一致（Exp 005 到 019）：模板化的程序生成题在自己的分布上学得很好，对自然语言写的基准题不迁移。
- decider 从 v1 到 v2 的提升来源，按其披露，是 27B 教师模型带思考写的 1.1 万条"文档 + 问题"并用两次独立作答一致来过滤。程序生成只占 8 千行。要在这个模型上再往上走，杠杆是自然语言、有文档依托、答案经核验的题，不是更多模板题。
- 本次权重不提交。

## 模型汤（权重平均，零训练，2026-09-25）

| 汤 | easy | standard | hard | 合计 | hard 配对翻转 |
|---|---|---|---|---|---|
| v2.1（起点） | 48 | 71 | 73 | 192 | — |
| avg(v2, v2.1) | 48 | 71 | 72 | 191 | 修 3 坏 4 |
| avg(v1, v2, v2.1) | 48 | 71 | 73 | 192 | 修 3 坏 3 |

同血统三个版本的权重平均对公开题没有可测变化，不采用。

## 冻结的 Qwen3.5-4B 基线（decider 引擎，chat 布局，T = 1.0，无训练）

231 题 183 对（48 / 64 / 71）。hard 层与 decider v2.1 的差别：judge_hard 14 对 9，multi_hop 11 对 15，probability 5 对 8；配对修 13 坏 15。一个没训过的 4B 在公开 hard 上就有 64%，decider 全部训练在公开题上的净收益约 +9 题，主要在 multi_hop 与 probability。

## 密封题代理：typed-decisions 测试集 2,000 题（wire 格式，T = 1，`ops/wire_eval.py`）

| 模型 | acc | Brier | ECE | choice | noul | score |
|---|---|---|---|---|---|---|
| decider-4b v2.1 | 0.681 | 0.148 | 0.020 | 0.680 | 0.742 | 0.636 |
| Qwen3.5-4B 原版（chat 布局） | 0.583 | 0.272 | 0.183 | 0.622 | 0.632 | 0.516 |

公开 231 题两者只差 9 题，代理集差 10 个点、Brier 差近一倍，和榜单密封分的差距（34.7 对 25.6）方向一致。以后用这个集合判断"密封题会不会涨"，公开题只用来确认没训坏。

## decider-4b + LoRA（fp32 适配器，r64，lr 1e-4，同一 8.4 万条混合）

231 题 184 对（48 / 71 / 65）。hard 层 73 → 65：temporal_numeric 5 → 1，long_policy 10 → 8，trap 8 → 7，ambiguous 5 → 4；配对修 2 坏 10。

结论：精度假设成立了一半——LoRA 确实学进去了（纯 bf16 全参那次几乎没动），但学进去的是坏东西。我们的规则生成数据在这个 4B 上不是"不迁移"，是**反向迁移**：synth_temporal 训得越好，基准的 temporal_numeric 掉得越多。模板题教会模型一种狭窄的题面形式，自然语言的同类题反而答错。这与编码器上的结论（不迁移）一致，在更强的模型上表现为损害。下一次混合里规则世界要么去掉，要么只留极小比例。

## Qwen3.5-4B 原版 + LoRA（同一 8.4 万条混合，lr 5e-5）

231 题 180 对（48 / 71 / 61）。对比冻结原版 183（48 / 64 / 71）：standard 层 +7（rubric / 格式对齐起作用），hard 层 −10（temporal_numeric 5 → 0，long_policy 10 → 7，multi_hop 15 → 12），配对修 6 坏 18。与 decider + LoRA 的结论一致：规则世界的数据在 4B 上有害，带描述候选的 rubric 数据对 standard 层有益。

## 大混合（qwen_big_v1）配方
Qwen3.5-4B + LoRA r64 / alpha 128，lr 1e-4，1 epoch，H100 无梯度检查点。数据 51.8 万条：decider 公开混合（delta 模式，`decider.data.mixture`，174 个任务）随机抽 50 万 + LegalBench 8 千 + typed-decisions 训练集 6 千 + synth_rubric 4 千；规则世界（policy / multihop / temporal）全部不用。

## qwen_big_v1 结果（2026-09-26，H100，7,083 步，1.82 亿 token，5.6 小时）

| 模型 | 公开 easy / standard / hard | 合计 | 代理集 acc | Brier | ECE |
|---|---|---|---|---|---|
| Qwen3.5-4B 原版 | 48 / 64 / 71 | 183 | 0.583 | 0.272 | 0.183 |
| **qwen_big_v1** | 48 / 70 / 66 | 184 | **0.773** | **0.115** | 0.019 |
| decider-4b v2.1 | 48 / 71 / 73 | 192 | 0.681 | 0.148 | 0.020 |

hard 层按题型（原版 → 本次）：tradeoff 2 → 5，judge_hard 14 → 10（原版 14，decider 9），probability 5 → 3，temporal_numeric 4 → 3，multi_hop 11 → 13；配对修 14 坏 21（对 decider）。代理集大幅上升，但 typed-decisions 训练集（6 千条）在混合里，代理集对本模型是同一生成器的未见题，不是完全分布外；decider 混合是否含该集未知。权重：`tdelab/tde-checkpoints/runs/qwen_big_v1`。
