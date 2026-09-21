# TDE v0.1 方案（修订版）

> Typed Decision Encoder：一个 150M～400M 参数、非自回归、以「state + question + candidates → 校准概率分布」为唯一原语的决策模型研究项目。
> 本文是 2026-09-21 可行性评审后的修订版，替代对话稿中的原始 v0.1。项目名 TDE 为占位，可改。

## 0. 评审结论摘要（为什么要改）

| 原方案主张 | 核验结果 | 处理 |
|---|---|---|
| 「从零验证一种新架构」「第一个原生决策模型」 | 已有 openJev-verdict-2.0（ModernBERT-base 149.6M）、Laya（421M）、kev、decider 等原生小模型开源，且在公开集上打平或超过 Jev；GLiClass、TARS、NLI zero-shot 覆盖了架构本身 | 去掉新颖性叙事，改为**同参数量对照研究** |
| 「在 Jev 开源之前」 | TypeSafe 从未提过开源，CEO 只说「讨论过写论文」 | 删除该前提 |
| 150M 微调 ≈ 4B 零样本 | 专家 vs 通才，不可比；Level 1 与训练集重叠 | 拆成两轨评测（§5） |
| 用 JevBench / typed-decisions 测校准 | 标签全是 LLM 共识，ECE 测的是与教师的一致度 | 校准主指标改用人类分布或可验证标签 |
| 前沿模型 ensemble 蒸馏软标签 | Claude / Gemini / GPT-6 Astra 不返回 logprobs；ToS 禁止；1M 样本 $15k～68k | 改用开源教师 logits（Qwen3 8B/32B、gpt-oss-120b） |
| 冻结 80% backbone + 20M 头 | 正是已发表的 OOD 坍塌配方（minojev 域内 95.8% / 未见域 31.7%） | 全参微调，从有零样本迁移能力的 checkpoint 初始化 |
| 「每条真值改写成几十种接口」 | 问题和候选全由正标签生成，模板泄漏答案 | 按源样本切分、模板配负例、每条 ≤8 个改写 |
| 242 题 benchmark、coverage@99% | 分辨不了 <7 个点差距；coverage@99% 需 ≥300 零错样本 | 每单元 ≥2000 题，配对 bootstrap CI |
| Exp 003：CE / CE+Brier / +temperature | 温度是单调变换，动不了 risk-coverage；CE+Brier 与 CE 同最优解 | 因子设计，AURC 为主 |
| 自己跑 Jev 当基线 | TypeSafe MCA（2026-09-19）禁止用输出蒸馏 / 开发竞品 | 只引用第三方已发布行 |

## 1. 研究问题

- **H1（架构）**：在相同决策训练数据、相同参数量下，双向编码器 + 候选打分头（无自回归解码）在未见任务族上的准确率 / Brier 不劣于因果解码器 + pointer head，且吞吐更高。
- **H2（泛化）**：<400M 的非生成模型能恢复 LLM 零样本决策泛化的多少，作为预训练任务广度和 few-shot 预算的函数。
- **H3（校准）**：直接以 proper scoring rule 训练的概率，在分布偏移和候选改写下，选择性预测（AURC）是否优于从 LM logits 读出的概率。

不做的主张：不声称复现 Jev 架构，不声称「Decision Foundation Model」，不与 Jev 比延迟（不同硬件和网络）。

## 2. 模型

### 2.1 统一原语

```
P(candidate_i | state, question, {candidate_j})
noul   = 2 个候选（yes / no，可附 criteria 文本）
choice = 2～255 个候选（名称 + 描述）
score  = 2～10 个有序等级（描述 + 等级序号 embedding），score = Σ i·p_i
```

### 2.2 两种读出方式（Exp 001 对照）

| 模式 | 结构 | 用途 |
|---|---|---|
| `joint` | `[CLS] state [SEP] question [SEP] <opt> desc_1 <opt> desc_2 … <decide>`，一次编码器前向，从每个 `<opt>` 标记的 hidden 与 `<decide>` 做 pointer 打分 | 基线；等价于 verdict-2.0 / Laya 的做法 |
| `branch` | state 只编码一次得到 H；每个问题一条分支序列 `question + <opt> desc … <decide>`，经 2～4 层「自注意力 + 对 H 的交叉注意力」，同样 pointer 读出 | 多问题共享 state；候选与问题同分支，保留 listwise 交互 |
| `biencoder`（消融） | state+question → z，候选单独编码 → e_i，s_i = zᵀW e_i | 验证候选间交互值多少 |

Backbone：`answerdotai/ModernBERT-base`（149M，8k 上下文，Apache-2.0），初始化优先用 `knowledgator/gliclass-modern-base-v3.0`（已有零样本迁移）。备选 `MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c`（184M，512 上下文，仅用于短 state 任务）。

解码器对照（Exp 002）：`Qwen/Qwen3-0.6B` + LoRA + pointer head，同样的数据与读出。

### 2.3 训练目标

```
L = CE_soft(p, y)                        # y 可为 hard one-hot、人类票数分布或教师分布
  + λ_ord · RPS(p, y)                    # 仅 score：ranked probability score（有序 proper score）
  + λ_perm · KL(p_π1 ‖ p_π2)             # 候选乱序两次前向的对称 KL，抑制顺序偏置
```

不再单独加 Brier（与 CE 同最优解，仅作 Exp 003 的一个 loss arm）。

### 2.4 校准与置信

- 后验温度：按 (primitive, K 桶) 在**独立校准集**上拟合，所有 arm 一律应用。
- 置信信号：默认 max-prob；Exp 003 对照一个独立训练的 confidence head（预测本题是否答对）。
- 选择性预测：split-conformal 阈值在校准集上拟合，给出 risk@coverage 的有限样本保证。

## 3. 数据

### 3.1 切分规则（先于任何改写）

- 切分单元 = 源样本 id（dataset + row），哈希到 train / calibration / test，比例 80 / 10 / 10。
- 校准集 ≥ 5k 源样本，覆盖每个 primitive 和 K 桶；测试集每个报告单元 ≥ 2k。
- 改写只在分区内进行。20% 的问题模板和候选词表整体留出，只在测试集使用。

### 3.2 源数据集（v0.1 Stage 0）

| 数据集 | 许可 | 原语 | 用途 |
|---|---|---|---|
| Banking77 | CC-BY-4.0 | choice (77) | 意图；Level 1 |
| CLINC150 (plus, 含 OOS) | CC-BY-3.0 | choice (150) + noul(OOS) | 意图；天然弃答目标 |
| AG News | 研究用途 | choice (4) | 主题；Level 1 |
| Yahoo Answers Topics | 研究用途 | choice (10) | OOD Tier 1（同族新数据集，训练时不用） |
| SST-5 | 研究用途 | score (5) | 有序 |
| GoEmotions (raw) | Apache-2.0 | choice (28) | 情绪；含标注者票数 → 人类软标签 |
| BoolQ | CC-BY-SA-3.0 | noul | 阅读理解式布尔判断 |
| SNLI | CC-BY-SA-4.0 | choice (3) | NLI |
| ChaosNLI | 研究用途 | choice (3) | **校准评测专用**：100 人票数分布 |
| typed-decisions (LocalLLaMA) | 见数据卡 | 混合 | 与现有开源行可比（标签为 LLM 教师，单独列表） |
| When2Call / MetaTool / Mind2Web(cross-domain) | 各自 | choice | OOD Tier 2（新任务族），训练时不用 |

### 3.3 模板与负例

- 每个数据集 ≥ 6 个问题模板；模板**独立于标签**采样。
- noul 模板对 k≥1 个不同标签的 state 施加，强制每模板 yes/no 50/50。
- choice 候选集：真标签 + 同数据集其它标签（随机子集，2～K 个），乱序；每条源样本训练改写 ≤ 8 个。
- 控制块（每层评测必报）：仅问题+候选（无 state）、打乱 state、候选顺序一致性、标签同义改写、加干扰候选、留出模板准确率。

### 3.4 教师软标签（Stage 1，可选）

开源教师（Qwen3-8B / 32B、gpt-oss-120b）经 vLLM 约束选项读 logits，≥4 次候选乱序平均；前沿 API 只用于 5k～20k 金标 / 评测，且只用暴露 token logprobs 的非推理模式。不用 Jev 输出做任何训练、选模或调参。

## 4. 实验矩阵

| 编号 | 问题 | 因子 | 主指标 |
|---|---|---|---|
| **Exp 001 读出与微调方式** | 哪种读出 / 哪种微调 | {joint, branch, biencoder} × {全参, LoRA-r16, 冻结 80%} | 留出数据集 Brier、准确率、吞吐 |
| **Exp 002 编码器 vs 解码器** | H1 | ModernBERT-base vs Qwen3-0.6B+pointer，同数据 | Tier 1/2 Brier、准确率、每决策 FLOPs 与延迟 |
| **Exp 003 校准因子** | H3 | {soft-CE, +RPS, +perm-KL, +Brier} × {TS 无/有} × {maxprob, conf head} × 3 seeds | 人类标签集 NLL / Brier；次要 AURC、OOD ECE |
| **Exp 004 OOD 与 few-shot** | H2 | Tier 1/2/3 × {0, 100, 1k, 10k} shot（LoRA） | 准确率、Brier、偏移下 ECE、AURC |

OOD 三层：Tier 1 同族新数据集（AG News → Yahoo）；Tier 2 新任务族（When2Call / MetaTool / Mind2Web）；Tier 3 新领域 + 新接口，程序生成标签（规则工单集，声明噪声率）。

## 5. 基线（分两轨，永不同表）

**轨 A（同数据同参数量）**：ModernBERT-base / large；Qwen3-0.6B + LoRA + pointer；Gemma-3-270M + pointer。

**轨 B（零样本，在无人训练过的任务族上）**：deberta-v3-base/large-zeroshot-v2.0；gliclass-base/large-v3.0；Qwen3-0.6B / 1.7B / 4B prefill 选项 logprob；Qwen3-Reranker-0.6B；bge-large 双塔；verdict-2.0；Laya；SetFit 与 LR-on-bge-small（每个 shot 数）；Jev 仅引用第三方已发布行（JevBench、elcronos、AbdelStark、scienthoon）。

## 6. 评测协议

- 指标：准确率、Brier（主）、NLL（概率下限 1e-6）、ECE（15 等质量桶，报告噪声下限）、smooth ECE、AURC / E-AURC、AUROC(对/错)、risk@coverage{50,80,90}、coverage@risk{1,5,10}；有序任务另报 MAE、RPS。
- 统计：每个对比报配对差异 + 10k 次配对 bootstrap 95% CI + 精确 McNemar；共享源文档的改写按源文档重采样。
- 每个实验预注册一个主终点（写入带时间戳的文件），其余为探索性。
- JevBench（v1.2.7，534 题，LLM 出题）只作次要参考。

## 7. 预算与时间

| 项目 | 估计 |
|---|---|
| 工程时间 | 8～11 周（数据与评测占一半以上） |
| 训练算力 | 150M 全参微调：100k 样本/epoch 在 4090 上数分钟；本机 M5 Pro (MPS) 可跑 Stage 0，Stage 1 起建议租 4090/A100 |
| API 费用 | < $300（开源教师本地跑；前沿 API 只做金标） |
| 显存 | 150M 全参 batch 32 × 512 约 13 GB；300M 需梯度检查点 |

## 8. 阶段与 go/no-go

- **Stage 0（1～2 周）**：≤10k 源样本，Exp 001 跑通，确认 branch 模式不劣于 joint（Brier 差 <0.01）且吞吐 ≥2×。否则退回 joint。
- **Stage 1（3～5 周）**：全部 3.2 数据集，Exp 002 + 003。go 条件：编码器在 Tier 1/2 上与解码器 CI 重叠或更好，且延迟更低；校准因子中至少一个 arm 在人类标签集上 NLL 显著优于 soft-CE 基线。
- **Stage 2（3～4 周）**：Exp 004 + 教师软标签。go 条件：Tier 2 零样本 ≥ deberta-zeroshot-v2.0 基线；100-shot 后 ≥ SetFit。
- 任一 no-go：写出负结果报告，不进入 v0.2（stochastic latent / latent diffusion）。

## 9. 仓库结构

```
tde/            模型、数据、损失、校准、评测代码
scripts/        build_data / train / evaluate / smoke_test
configs/        每个实验一个 yaml
tests/          切分、模板泄漏、指标、模型形状
docs/           本方案、评测协议、预注册文件
data/           生成的 JSONL（git 忽略）
runs/           checkpoint 与日志（git 忽略）
```

## 10. 明确不做的事

- 不逆向、不推测 Jev 内部架构，不把 Jev 输出用于训练或选模。
- v0.1 不做 diffusion / stochastic latent。
- 不在 README 里使用「首个」「新架构」「Foundation Model」措辞。

## 11. 两期路线（2026-09-21 确定）

**第一期：开源版。** 目标是用 ≤150M 的模型在典型决策路由任务上达到公开可比的齐平或领先水平，然后开源权重、数据管线和评测。验收标准（全部用第三方已发布的 Jev 数据对比，本项目不调用 Jev）：

| 基准 | 目标 | 现有参照 |
|---|---|---|
| typed-decisions 测试集（2,000 题） | 准确率与 Brier 同时优于 Jev 1.13 与 verdict-2.0 | Jev 72.7% / 0.148；verdict-2.0（149.6M）77.1% / 0.064 |
| JevBench v1.2.7（534 题，经其 adapter 提交） | 超过 Laya 421M，逼近 Jev | Laya 70.1；SemIf 74.7；Jev 75.4 |
| 意图路由全标签集（banking77 77 类、clinc150 150 类，训练只见 ≤12 候选子集） | 5% 错误率下自动执行 ≥90%，并通过相干性与换序不变性测试 | 本项目 Exp 001 子集设置下 93%～95% |

随开源一起发布的实验结论：(a) 监督 proper score 训练不弱于 RLCD 式 proper-reward 策略梯度（同最优点，方差更小；含 correctness-only REINFORCE 负对照）；(b) 读出位置（[MASK] + span）带来的收益及 2×2 消融；(c) 校准来自目标函数，各桶温度 ≈1。

措辞约束：本项目是基于公开资料的数学推导与小模型实验得到的近似设计，不探测、不调用、不蒸馏 Jev；发布时描述为「复现同类接口与行为的开放设计，在公开基准上用第三方数据对比」。

**第二期：研究版。** 从结构和数学上论证更优的设计并持续实验：
1. 相干性框架：统一能量 E(s,q,a) 生成三种原语，证明相干与多原语训练一致性，定义并测量相干误差（含对 Jev 第三方数据的对照）。
2. 结构三旋钮：不对称块注意力（state 可缓存、多问题并行）、位置绑定候选块（严格置换等变，perm-KL 恒为 0，pointwise/listwise 开关）、晚期交互深度 k；给出质量对成本前沿。
3. 子集泛化定理：闭世界下 pointwise 读出的子集条件概率免费正确；实验为候选数外推与无关候选注入。
4. 选择性预测的有限样本保证（conformal / Learn-then-Test）。
5. 受算力限制的项目（395M 底座、解码器同参对照、长 state、开源教师蒸馏、3 seeds 全矩阵、agent 轨迹集）在算力扩充后执行。
