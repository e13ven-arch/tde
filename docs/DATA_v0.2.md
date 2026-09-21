# v0.2 数据集清单（2026-09-21，Exp 005 数据广度实验）

v0.1 的 7 个分类数据集不变（见上文 v0.1 表），新增下列来源。切分单元仍为源样本；合成数据按「世界 / 规则组合」切分；LegalBench 另有 22 个任务整体留出作任务级 OOD 测试。合计：train 865,818 · calibration 53,698 · test 61,884。

| 数据集 | 许可 | 用途 | 源样本 | 样例 | train / cal / test | 原语 |
|---|---|---|---:|---:|---|---|
| synth_policy | generated in this repo (Apache-2.0) | train | 8,000 | 32,000 | 24,172 / 3,764 / 4,064 | noul:16,000, choice:8,000, score:8,000 |
| synth_multihop | generated in this repo (Apache-2.0) | train | 8,000 | 40,000 | 32,170 / 3,905 / 3,925 | noul:16,000, choice:16,000, score:8,000 |
| synth_temporal | generated in this repo (Apache-2.0) | train | 8,000 | 48,000 | 37,980 / 4,818 / 5,202 | noul:32,000, choice:8,000, score:8,000 |
| legalbench | CC-BY-4.0 (per-task; see tde/data/legalbench_tasks.json) | train | 47,256 | 47,256 | 32,113 / 3,910 / 11,233 | choice:5,277, noul:41,979 |
| hotpot_qa | CC-BY-SA-4.0 | train | 30,000 | 78,201 | 72,302 / 2,915 / 2,984 | choice:73,379, noul:4,822 |

## 来源与生成方法

- **synth_policy / synth_multihop / synth_temporal**：本仓库 `tde/data/synth.py` 程序生成，标签由规则引擎、图遍历和日期数值运算得出；模板独立于标签采样，yes/no 按构造平衡；生成器、种子和模板随代码发布，可一键复现。设计只依据公开文档对题型的描述，不参考任何基准题；刻意排除退款 / 收据类政策场景。
- **legalbench**：CC-BY-4.0，118 个规则应用子任务（97 个 yes/no、21 个小多分类），问题文本取自官方 base prompt 的任务说明；每任务最多 1,500 行；22 个任务按哈希整体留出。
- **hotpot_qa**：CC-BY-SA-4.0（衍生数据同样按 SA 条款发布），30,000 行：比较题的 yes/no 作为 noul，桥接题改为 Choice（金标答案 + 语料标题 / 其他答案作干扰项）；state 为支撑段落在前、1～3 个干扰段落在后。
- **typed_decisions**：LLM 教师标签；HF 训练集 6,000 条决策进入训练（与 verdict-2.0 同做法），HF 测试集 2,000 条只放在 `eval_only/` 作「专家模式」报告，永不并入主测试集。

## 污染检查

`scripts/contamination_check.py`（13-gram 重叠）：JevBench 231 道公开题与 v0.2 训练 state **无任何重叠**；typed-decisions 测试集与其自身训练集有 89% 的题共享 13-gram，来源是同一工作流的 JSON 字段骨架，精确 state 重叠为 0（上一行输出）。
