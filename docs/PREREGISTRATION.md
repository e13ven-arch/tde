# 预注册终点（v0.1）

每个实验在跑之前把主终点写进本文件并提交；提交时间戳即预注册时间。主终点以外的指标一律标为探索性。

| 实验 | 主终点 | 次要终点 | 判定规则 | 预注册提交 |
|---|---|---|---|---|
| Exp 001 读出方式 | calibration 分区、`templates=held_out_templates` 切片上的 Brier（joint vs branch vs biencoder，同 finetune=full） | 准确率、每决策吞吐（决策/秒，batch 32）、`order_argmax_agreement` | branch 的 Brier 与 joint 的配对 bootstrap 95% CI 上界 < +0.01 且吞吐 ≥ 2× 才保留 branch 为默认 | 待填 |
| Exp 002 编码器 vs 解码器 | Tier 1（yahoo_topics）与 Tier 2 零样本 Brier，ModernBERT-base vs Qwen3-0.6B+pointer，同训练数据 | 准确率、每决策 FLOPs、p50 延迟 | CI 重叠或编码器更好，且延迟更低 → H1 成立 | 待填 |
| Exp 003 校准因子 | go_emotions 人类票数切片（`labels=human_distribution` 或 go_emotions 数据集切片）上的 NLL，soft-CE 基线 vs 各 arm，3 seeds | AURC、OOD ECE（含噪声下限）、cov@risk5 | 至少一个 arm 的 NLL 配对差 CI 不含 0 | 待填 |
| Exp 004 OOD 与 few-shot | Tier 2 零样本准确率 vs deberta-v3-base-zeroshot-v2.0；100-shot 准确率 vs SetFit | Brier、AURC、偏移下 ECE | 零样本 ≥ deberta 基线且 100-shot ≥ SetFit → go | 待填 |

统计规则：每个报告单元 ≥ 2,000 条；配对 bootstrap 10,000 次、按源样本重采样；精确 McNemar；ECE 报 15 等质量桶并附噪声下限。
