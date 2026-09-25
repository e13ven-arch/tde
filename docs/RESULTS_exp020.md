# Exp 020 / 021：ModernBERT-large + set 拓扑 + 数据 v0.7（目标：JevBench 公开题全面超过同尺寸对照）

分支 `exp/laya`。配方 = Exp 010 发布配方，只改三处：骨干 GLiClass-large 编码器（395M，`scripts/extract_gliclass_encoder.py --repo knowledgator/gliclass-modern-large-v3.0`）、set 拓扑（换序 KL 恒为零，关掉）、数据 v0.7（v0.6 + `synth_rubric`：带描述的候选 + 指令里的判定规则，五个子族，2 万题，`tde/data/synth_rubric.py`）。训练布局：token 预算组批 `--max_tokens 20000 --batch_size 64`，无梯度检查点，RTX 6000 Ada 48GB，3,073 步 / epoch，0.36 s/步，19 分钟。Exp 021 是同数据、同拓扑的 base（150M）对照，用来分离数据和尺寸的贡献。

## JevBench 公开题（231，本机原生概率，无温度）

| 模型 | easy (48) | standard (72) | hard (111) | 全部 |
|---|---|---|---|---|
| Exp 010 发布版（base, seq, v0.5） | 48/48 | 0.611 (44) | 0.342 (38) | 0.563 |
| Exp 019 base, set, v0.5 | 48/48 | 0.611 (44) | 0.306 (34) | 0.545 |
| Exp 021 base, set, **v0.7** | 47/48 | 0.653 (47) | 0.297 (33) | 0.550 |
| **Exp 020 large, set, v0.7** | 48/48 | **0.778 (56)** | **0.369 (41)** | **0.628** |

standard 层按题型（Exp 010 / 021 / 020，各 12 题）：adequacy 5 / 5 / 5，extraction 11 / 7 / 11，intent 5 / 8 / 10，ordinal 7 / 10 / 12，policy 8 / 8 / 7，routing 8 / 9 / 11。
hard 层按题型（Exp 020）：judge_hard 10/17（发布版 5/17），trap 5/8，adversarial 5/6，routing_hard 3/5，long_policy 4/19，multi_hop 3/18，temporal_numeric 3/15，probability 4/10，tradeoff 2/6，ambiguous 2/7。

## 其余评测

| 评测 | Exp 010 | Exp 021 base v0.7 | Exp 020 large v0.7 |
|---|---|---|---|
| v0.7 测试集（2 万） | — | 0.780 / NLL 0.500 | **0.870 / 0.303** |
| v0.1 回归 | 0.875 | 0.869 | **0.897** |
| banking77 全 77 类一次前向 | 0.758 | 0.757 | **0.853**（cov@5% 0.74） |
| clinc150 全 150 类一次前向 | 0.865 | 0.852 | **0.928**（cov@5% 0.96） |
| typed-decisions 混合模式 acc / Brier | 0.666 / 0.109 | 0.643 / 0.115 | **0.695 / 0.094** |
| synth_rubric 自身测试 | — | — | 0.967 |

## 结论（单 seed，seed 1、2 复核进行中）

- 目标线（standard ≥ 0.75、hard ≥ 0.34）过了：0.778 / 0.369。standard 层的提升集中在 intent、ordinal、routing 三类，正是 v0.7 针对的形式；adequacy 12 题仍只对 5，policy 没变。
- 数据和尺寸的分工：v0.7 在 base 上只带来 standard +3 题，且 hard 掉 1 题；换到 large 之后 standard +9、hard +8。也就是说带描述的候选和指令里的规则这种形式，150M 学不动，395M 学得动。judge_hard 从 5 到 10 也是尺寸带来的。
- 所有回归项同时上升，全标签集是历史最好（banking77 +9.5 点，5% 风险覆盖 51% → 74%）。
- 与 Laya 的直接对比要等官方跑：Laya 公开 534 题 58.4%、密封 30.8%；我们 231 公开题 62.8%。
