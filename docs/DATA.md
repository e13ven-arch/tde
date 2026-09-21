# v0.1 数据集清单（Stage 0 构建结果）

由 `python scripts/build_data.py --stage 0` 生成；切分单元为源样本 id，比例 80/10/10；训练分区每条源样本 ≤4 个改写，评测分区各 1 个训练模板改写 + 1 个留出模板改写。

| 数据集 | 许可 | 用途 | 源样本 | 样例数 | train / calibration / test | 原语 |
|---|---|---|---:|---:|---|---|
| banking77 | CC-BY-4.0 | train | 13,069 | 47,060 | 41,844 / 2,622 / 2,594 | noul:26,743, choice:20,317 |
| clinc150 | CC-BY-3.0 | train | 23,850 | 83,654 | 74,057 / 4,777 / 4,820 | choice:57,380, noul:26,274 |
| ag_news | research/non-commercial (AG corpus terms) | train | 40,000 | 144,010 | 128,020 / 8,006 / 7,984 | noul:67,912, choice:76,098 |
| sst5 | research | train | 11,855 | 40,544 | 38,252 / 1,164 / 1,128 | noul:16,178, score:24,366 |
| go_emotions | Apache-2.0 | train | 15,775 | 56,664 | 50,228 / 3,262 / 3,174 | choice:40,270, noul:16,394 |
| boolq | CC-BY-SA-3.0 | train | 12,697 | 32,997 | 30,450 / 1,256 / 1,291 | noul:32,997 |
| snli | CC-BY-SA-4.0 | train | 79,832 | 325,014 | 298,230 / 13,299 / 13,485 | choice:203,773, noul:121,241 |

合计：train 661,081 · calibration 34,386 · test 34,476

## 尚未拉取（Stage 1）

- **chaos_nli**（校准评测专用，100 人票数分布）：需手动下载 https://github.com/easonnie/ChaosNLI ，把 `chaosNLI_snli.jsonl`、`chaosNLI_mnli_m.jsonl` 放到 `data/external/chaosnli/`（或设置 `TDE_CHAOSNLI_DIR`）。
- **yahoo_topics**（OOD Tier 1）、**typed_decisions**（教师标签，仅可比性）：`python scripts/build_data.py --datasets yahoo_topics typed_decisions`。
- OOD Tier 2（When2Call / MetaTool / Mind2Web cross-domain）：适配器待写。

## 注意

- snli 占训练集近半；Stage 0 训练时用 `train_limit` 或后续加入按数据集均衡采样。
- sst5 测试分区只有 ~1.1k 条（数据集本身只有 11.9k 句），低于 2k/单元的目标，报告时需标注。
- ag_news 许可为研究用途；banking77 为 CC-BY-4.0；go_emotions 为 Apache-2.0；boolq CC-BY-SA-3.0；snli CC-BY-SA-4.0；clinc150 CC-BY-3.0。
