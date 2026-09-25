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
