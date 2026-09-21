# JevBench 公开题结果（231 题：easy 48 / standard 72 / hard 111）

运行方式：`scripts/run_jevbench.py`，本项目的 `integrations/jevbench/tde_local.py` 适配器，原生概率，本地推理。JevBench 的 133 道保留题需要由基准维护者运行，这里只有公开题；第三方参照取自该仓库 `results/v1.2/jevbench-v1.2-results.json` 的 `hard.accuracy`（保留 + 公开的 hard 题合计）。

| 模型 | 训练数据 | easy | standard | hard | 全部 | Brier | ECE(10 桶) |
|---|---|---:|---:|---:|---:|---:|---:|
| joint v2，Stage 0（7 个分类集，1 epoch） | v0.1，每集 1 万 | 97.9% | 61.1% | 26.1% | 51.9% | 0.587 | 0.097 |
| joint v2，Stage 1（同上，2 epoch） | v0.1，每集 2 万 | 97.9% | 61.1% | 28.8% | 53.2% | 0.574 | — |
| Exp 005 数据广度（待跑） | v0.2 | | | | | | |

hard 层第三方参照：Jev 1.13 为 74.1%，SemIf（Qwen3.5-4B）59.5%，kev 0.6B 为 40.0%，openJev verdict（ModernBERT-base）38.2%，GLiNER2.5 multi 37.7%，Laya 421M 为 34.1%。

## Stage 0 按题型（hard 层最弱项）

tool_selection 12/12、fact 12/12、intent 21/24、extraction 19/24、ordinal 9/12；long_policy 3/19、multi_hop 3/18、trap 0/8、routing_hard 1/5、temporal_numeric 4/15、judge_hard 7/17（ECE 0.51，严重过度自信）。

结论：分类型题已饱和，缺的是「按规则逐条核对」「沿引用链多跳」「时间数值运算」三种技能的监督信号；Stage 1 加倍同分布数据没有改变这一点。Exp 005 用 v0.2（规则生成 + LegalBench + HotpotQA + typed-decisions 训练集）针对这三类补数据，主终点为 hard 层准确率。
