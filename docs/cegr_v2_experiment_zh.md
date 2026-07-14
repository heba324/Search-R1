# Search-R1 CEGR V2 改进实验与云端运行手册

> 状态：CEGR V1 已冻结；CEGR V2 代码与离线门禁已完成；GPU 训练结果尚未产生。
>
> 分支：`codex/search-r1-cegr-v2`
>
> V1 基准提交：`8672aad0f4089f0fca388601cd9ce20fc9b8b776`
>
> 硬件目标：单张 A800 80GB，CPU BM25，沿用现有 `Search-R1` 与 `Search-R1-retriever` Conda 环境。

> 时间紧、希望在 2 步冒烟后直接训练单个 EFF120 新模型时，使用独立的[紧急 2+120 路线手册](cegr_v2_direct120_urgent_zh.md)。该路线复用但重新评测已有 baseline；它不替代本文的 Grouped-EM/EFF 双臂因果对照。

## 1. 先说结论

CEGR V1 是可信的负结果，不是运行失败。固定 700 题上，EM 从 `0.2400` 降到 `0.2071`，差值 `-0.0329`，Bootstrap 95% CI 为 `[-0.0629, -0.0029]`，exact McNemar `p=0.03975`。因此不能把 V1 写成“准确率基本持平”。

GPT 给出的“保留 V1、在新分支做 V2、删除普通搜索与长度偏好、先小测再正式评测”方向是对的，但它漏掉了一个更关键的代码事实：当前 trainer 先生成同题 5 条 rollout，随后却给每一条分配独立 `uid`。GRPO 按 `uid` 分组，结果每组只有一条，advantage 近似原始 reward，而不是真正的同题组内标准化。

这会让 V1 中 F1 或 `answer_in_context` 部分命中的错误答案直接得到正 advantage。继续把权重改成 `0.65 EM + 0.25 F1 + 0.10 evidence` 并不能解决这个问题。V2 因此不采用那组直加权，而是先恢复同题共享 `uid`，再比较两个等预算候选：

1. `Grouped-EM`：修复 grouping 后继续使用纯 EM。
2. `EFF`：EM-First, F1-Fallback。只在同题 5 条 rollout 全部 EM 为 0 时使用 token-F1 排序。

两者都从同一个冻结 baseline `global_step_120` 追加 40 步。先用与最终 700 题完全不重叠的 140 题 pilot 选择候选，再一次性进入最终评测。

## 2. 对 GPT 原判断的辨别

| 原判断 | 结论 | 原因 |
|---|---|---|
| V1 必须永久保留，V2 独立分支、目录和 run name | 正确 | 防止负结果与 checkpoint 被覆盖，也保证报告可审计 |
| V1 的 EM/F1 显著下降 | 正确 | 700 条配对结果、Bootstrap 与 McNemar 均支持 |
| V1 因“普通搜索惩罚和长度惩罚过强”而下降 | 证据不足 | V1 没有直接长度惩罚；1 至 3 次正常有效搜索也不扣分，惩罚主要针对非法、重复和第 4 次有效搜索 |
| 删除搜索成本项是低风险方向 | 基本正确 | 它能减少变量并避免把效率代理重新放进训练目标，但不能单独保证 EM 上升 |
| 使用固定 `EM + F1 + evidence - penalty` 新权重 | 不采用 | singleton `uid` 下错误轨迹仍会得到直接正 advantage；同类中间检索奖励的经验研究也显示高权重可能伤害结果 |
| 只要静态上“正确分数高于错误”就能保证训练有效 | 错误 | 静态排序不等于组内 advantage、后续 policy 分布或最终 EM 保证 |
| 先小规模测试再投入完整预算 | 正确 | V2 将它落实为离线性质测试、2 步信号 smoke、独立 140 题候选选择和最终 700 题四层门禁 |

所以，本次不是把 CEGR 改回原始 baseline。V2 仍然包含新的组级稠密反馈，但先修复了它赖以成立的 GRPO 分组前提，并保留一个纯 EM 候选作为风险较低的退路。

## 3. 文献调研得到的设计约束

完整综述见 [`docs/research/cegr_v2_literature_review.md`](research/cegr_v2_literature_review.md)。这里仅列与决策直接相关的结论。

### 3.1 Search-R1 与 GRPO

Search-R1 使用最终答案 EM 作为结果奖励，并让模型在推理过程中自主调用检索。GRPO 的组相对信号要求同一 prompt 的多条输出属于同一组。DeepSeekMath 与 DAPO 也都把“同题多输出形成可比较组”作为核心前提；全 0 或全 1 组没有有效相对梯度。

本仓库当前 singleton `uid` 路径破坏了这个前提。因此，恢复共享 prompt identity 是 V2 的第一优先级，而不是再增加一个奖励项。

### 3.2 F1 部分信用

R1-Searcher 与 ReSearch 表明，token-F1 可以缓解二元 EM 的稀疏性。但 F1 也可能鼓励词面重叠或更长答案。LeTS 进一步报告，小模型在仅使用结果奖励时可能发生搜索策略坍缩。由此得到两个约束：

1. F1 不能在已有 exact winner 的组内改变 EM 的胜负关系。
2. 响应长度、搜索次数和单跳退化必须作为评测护栏，而不是再次混入 reward。

### 3.3 中间检索奖励

Search-R1 后续经验研究发现，“gold answer 是否出现在检索文本中”的中间奖励收益有限，权重增大时性能会下降。V1 的 `evidence_answer_coverage` 与它高度相似，而且它只能证明答案字符串出现，不能证明 supporting fact、来源或推理链正确。

因此 V2 删除训练期 evidence reward。`evidence_coverage` 仍保留为诊断指标，不再充当模型优化目标。

### 3.4 暂不采用的方法

Potential-based reward shaping、TIPS 式 turn-level potential、约束强化学习和词典序多目标优化都有研究价值，但需要额外 teacher forward、状态势函数或优化器。它们会在当前 1.5B、单卡、课程预算中引入过多变量，保留为 V3 备选，不与本次 grouping 修复混在一起。

## 4. V2 的精确定义

对问题 `q` 的 5 条 rollout，定义：

```text
e_i = official_exact_match(final_answer_i, gold_q)  in {0, 1}
f_i = token_f1(final_answer_i, gold_q)               in [0, 1]
```

两个候选的 reward 是：

```text
Grouped-EM:
C_i = e_i

EFF:
if sum(e_j for j in group_q) == 0:
    E_i = f_i
else:
    E_i = e_i
```

随后仍由现有 GRPO 实现计算：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + 1e-6)
```

### 4.1 可以严格保证的性质

1. 同一问题的 5 条 rollout 使用同一 `uid`。
2. 任一组只要存在 exact winner，EFF 与 Grouped-EM 的 reward vector 逐元素完全相同。
3. 全错组中，Grouped-EM advantage 全 0；EFF 只有在 F1 存在差异时才产生正负相对信号。
4. 错误大写标签、缺失 `<answer>` 标签与正式 parser 一样得到 0。
5. 正常搜索次数、响应长度、evidence 字符串和重复搜索不会直接改变 V2 reward。

### 4.2 不能保证的事情

上述性质不能保证最终 EM 必然上涨。全错组中的 F1 更新会改变模型参数和后续 rollout 分布；CUDA、采样与小模型能力也带来方差。任何人在真实训练前承诺具体涨点都不科学。

本项目能保证的是：失败会尽早、明确、低成本地暴露；最终只有通过独立数据门禁的方法才会被写成有效改进。

## 5. 实验矩阵与归因边界

| 名称 | 起点 | 追加步数 | grouping | reward | 作用 |
|---|---|---:|---|---|---|
| Frozen baseline | baseline step 120 | 0 | 历史 singleton 路径 | EM | 已完成基线 |
| Grouped-EM | 同一 baseline step 120 | 40 | 同题共享 `uid` | 纯 EM | 候选 A，同时是 EFF 的等预算对照 |
| EFF | 同一 baseline step 120 | 40 | 同题共享 `uid` | 全错组 F1，否则 EM | 候选 B |

两条新训练臂固定：Qwen2.5-1.5B-Instruct、NQ+HotpotQA、batch 32、group size 5、BM25 Top-3、最多 4 轮搜索、driver seed 42、vLLM engine seed 42、学习率 `5e-7`、40 步。V2 专用 worker 将 engine seed 传给 vLLM 构造器；它不把 seed 塞进每条请求的 `SamplingParams`，因此不会让同题 5 条 rollout 因共享请求 seed 而失去采样多样性。

`EFF - Grouped-EM` 可以用于判断 F1 fallback 的额外贡献。`Grouped-EM - Frozen baseline` 同时包含 grouping 修复和额外 40 步训练，不能进一步声称差值完全由 grouping 单独造成。报告中应称它为“group-corrected refinement 复合候选”。

## 6. 四层门禁

### Phase 0：离线性质测试

必须通过：

```text
同题 rollout 数量恰好为 5
混合正确组中 EFF reward == EM reward
严格标签 parser 与官方 scorer 一致
V1 完整 checkpoint、代码和结果 SHA-256 未改变
search_r1/ 与 verl/ 相对 V1 提交零差异
```

### Phase 1：2 步机制 smoke

smoke 只使用 disjoint pilot 数据，不读取最终 700 题。除了无 OOM、NaN、保存失败外，还要求：

```text
group_size > 1
所有混合正确组 reward 与 EM 无不一致
全错组中至少 10% 存在可排序的 F1 差异
```

完成门禁还会复核完整 `train.log` 中不存在独立 token 形式的 `NaN/Inf`、训练指标 step 连续、最终 checkpoint 存在且非空，并核对 completion marker 中的方法、group size 与 vLLM engine seed。它能排除日志中可见的非有限训练失败，但不能把“没有记录到 NaN”夸大成所有底层 CUDA 梯度都得到形式化证明。

最后一项不满足时，说明 EFF 在真实 rollout 上几乎没有额外信号，应停止，不启动两条 40 步训练。

### Phase 2：140 题盲 pilot 选择

pilot 为七个数据集各 20 题，并与最终七个数据集各 100 题的索引完全不重叠。选择规则预先固定：

1. EFF 相对 frozen baseline 的 EM 至少 `+0.01`，相对 Grouped-EM 也至少 `+0.01`；F1 与单跳表现相对两者都不下降，并满足 evidence、搜索、重复与长度护栏时，选择 `eff`。
2. 否则，如果 Grouped-EM 相对 frozen baseline 的 EM 至少 `+0.01`，同时满足同类护栏，选择 `grouped_em`。
3. 两者都不满足时，`selected_candidate=null`，停止实验，不查看最终 700 题。

140 题上的 EM 只能按 `1/140` 变化，所以 `+0.01` 实际要求至少净增加 2 个正确样本。它是筛选阈值，不是显著性证明。

### Phase 3：固定 700 题最终检验

最终主比较把冻结 baseline checkpoint、Grouped-EM 与 EFF 三个模型都放入同一个 V2 严格评测入口，使用同一固定数据、batch 28、driver seed 42、vLLM engine seed 42 与记录器。V1 已冻结的旧 baseline 轨迹仍会离线严格重评分并报告 parser mismatch 数，但它只作为历史一致性审计，不进入 V2 主比较。pilot gate 绑定三份 pilot JSONL 的 SHA-256；最终入口只能验证既有锁，不能重算或覆盖候选选择。

主结论只能针对 pilot 已锁定的候选，且最终有意义改进阈值统一为相关主比较 `Delta EM >= +0.02`：

```text
预声明成功：最终 EM 至少增加 0.02，并满足预设护栏
统计支持：paired bootstrap 95% CI 下界 > 0，且 exact McNemar p < 0.05
无效：EM 不满足条件，即使搜索更少或响应更短也按失败报告
```

## 7. 文件隔离

V1 保持：

```text
分支：codex/search-r1-improvement
baseline checkpoint：search-r1-course-qwen2.5-1.5b-grpo-bm25/.../global_step_120
CEGR V1 checkpoint：search-r1-cegr-qwen2.5-1.5b-grpo-bm25/.../global_step_120
结果：artifacts/improvement
```

V2 只写：

```text
分支：codex/search-r1-cegr-v2
Grouped-EM checkpoint：search-r1-cegr-v2-em-control-qwen2.5-1.5b-grpo-bm25
EFF checkpoint：search-r1-cegr-v2-qwen2.5-1.5b-grpo-bm25
结果：artifacts/improvement-v2
代码：scripts/improvement_v2
```

V2 不修改 `search_r1/`、`verl/`、`scripts/improvement/`，也不覆盖任何 V1 run name。资产链接器要求 V1 checkout 的 `HEAD` 精确等于 `8672aad0f4089f0fca388601cd9ce20fc9b8b776` 且无 tracked diff；冻结清单覆盖两个 step120 checkpoint、V1 结果、V1 代码以及 baseline/CEGR 两套原评测目录，并检测冻结目录中后来新增的文件。

## 8. 云端操作步骤

下面按新手可直接执行的顺序编写。命令成功后再进入下一小节。

### 8.1 找到保存完整 V1 的目录

先执行：

```bash
for d in /root/autodl-tmp/Search-R1-improvement /root/autodl-tmp/Search-R1; do
  if [ -s "$d/verl_checkpoints/search-r1-course-qwen2.5-1.5b-grpo-bm25/actor/global_step_120/config.json" ]; then
    echo "V1_ROOT=$d"
  fi
done
```

记住输出的目录。下文假设它是：

```bash
export V1_ROOT=/root/autodl-tmp/Search-R1-improvement
```

如果实际输出是 `/root/autodl-tmp/Search-R1`，就把上面一行改成那个路径。

### 8.2 克隆独立 V2 分支

```bash
cd /root/autodl-tmp
git clone --branch codex/search-r1-cegr-v2 \
  git@github.com:heba324/Search-R1.git Search-R1-cegr-v2
cd /root/autodl-tmp/Search-R1-cegr-v2
git status --short --branch
```

应看到当前分支为 `codex/search-r1-cegr-v2`，且没有未提交修改。

### 8.3 只链接 V1 输入，不共享 V2 输出目录

```bash
V1_ROOT="$V1_ROOT" bash scripts/improvement_v2/link_v1_assets.sh
```

该脚本只链接训练数据、模型资源、BM25 资源、两个 V1 checkpoint 和冻结的 V1 结果。V2 checkpoint、pilot 和 `artifacts/improvement-v2` 都留在新目录。

检查：

```bash
ls -l data
ls -l verl_checkpoints
ls -l artifacts
```

### 8.4 建立冻结清单并运行全部离线检查

```bash
bash scripts/improvement_v2/prepare_experiment.sh
```

成功标志：

```text
CEGR V1 freeze verified
全部 unittest 通过
CEGR V2 offline preparation passed
```

这一步会生成：

```text
artifacts/improvement-v2/v1-frozen-manifest.json
data/improvement_v2/pilot.parquet
data/improvement_v2/pilot_manifest.json
artifacts/improvement-v2/preflight/
```

任一失败都不要启动训练。

### 8.5 启动 CPU BM25 服务

在第一个终端执行：

```bash
cd /root/autodl-tmp/Search-R1-cegr-v2
bash scripts/course_reproduction/launch_bm25_retriever.sh
```

保持这个终端运行。在第二个终端执行：

```bash
cd /root/autodl-tmp/Search-R1-cegr-v2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
python scripts/course_reproduction/check_retriever.py
```

只有检索检查通过才继续。

### 8.6 运行 2 步 smoke

```bash
cd /root/autodl-tmp/Search-R1-cegr-v2
bash scripts/improvement_v2/run_smoke.sh
```

成功后检查：

```bash
cat artifacts/improvement-v2/search-r1-cegr-v2-smoke/reward_metrics.json
cat artifacts/improvement-v2/search-r1-cegr-v2-smoke/training_completed.txt
```

如果脚本因 `informative fallback rate` 低于 `0.10` 退出，这是科学上的 no-go，不应通过删除门禁强行继续。

### 8.7 运行两条 40 步候选训练

```bash
bash scripts/improvement_v2/run_pilot.sh
```

脚本先跑 Grouped-EM，再跑 EFF。每条都在 20、40 步保存 checkpoint。若某条已经完整完成，重新执行时会重新验真后保留并跳过；若 marker、日志、指标或 checkpoint 任一不完整，会停止并要求先检查，不会覆盖。

训练时另开终端监控：

```bash
watch -n 2 nvidia-smi
```

训练完成后应同时存在：

```bash
test -s verl_checkpoints/search-r1-cegr-v2-em-control-qwen2.5-1.5b-grpo-bm25/actor/global_step_40/config.json
test -s verl_checkpoints/search-r1-cegr-v2-qwen2.5-1.5b-grpo-bm25/actor/global_step_40/config.json
echo $?
```

最后应输出 `0`。

### 8.8 运行独立 pilot 评测并锁定候选

```bash
bash scripts/improvement_v2/evaluate_pilot.sh
```

查看：

```bash
cat artifacts/improvement-v2/pilot-evaluation/step-40/pilot-gate.json
```

关键字段：

```text
"passed": true
"selected_candidate": "eff"
```

或者：

```text
"passed": true
"selected_candidate": "grouped_em"
```

若 `passed` 为 `false`，脚本会非零退出。此时停止，不运行最终评测，并把 gate、三份 JSONL 和训练日志保留为负结果。

通过时，`pilot-gate.json` 内含三份轨迹的 SHA-256 和锁定候选。文件已存在时脚本只允许内容完全一致，不会覆盖；后续任一轨迹或候选字段变化都会使最终入口失败。

### 8.9 仅在 pilot 通过后运行最终 700 题

```bash
bash scripts/improvement_v2/evaluate_final.sh
```

最终报告入口：

```bash
cat artifacts/improvement-v2/final-evaluation/step-40/final-analysis.json
cat artifacts/improvement-v2/final-evaluation/step-40/baseline-rescore.json
head artifacts/improvement-v2/final-evaluation/step-40/historical-baseline-rescored.jsonl
```

重点读取：

```text
effectiveness.selected_candidate
effectiveness.predeclared_success
effectiveness.statistically_supported_for_selected_candidate
effectiveness.claim_level
```

### 8.10 归档证据

```bash
bash scripts/improvement_v2/collect_evidence.sh
```

得到：

```text
artifacts/improvement-v2/evidence/search-r1-cegr-v2-evidence.tar.gz
artifacts/improvement-v2/evidence/search-r1-cegr-v2-evidence.tar.gz.sha256
```

## 9. 成本计算

本方案的新训练更新总数是：

```text
smoke 2 步 + Grouped-EM 40 步 + EFF 40 步 = 82 步
```

训练成本可用你已经完成的 V1 实测速度估计：

```text
新训练时长约等于 V1 120 步训练时长 × 82 / 120
```

另加 140 题 pilot 的三次评测，以及最终三次 700 题评测。最终 baseline 也重新走 V2 严格评测入口，以消除新旧评测执行与 vLLM seed 的混杂；旧 baseline 轨迹仅做低成本离线 parser 审计。因此预算必须包含三套模型的最终生成，不能沿用旧版“两次 700 题”的估算。

准确费用必须以 smoke 的真实耗时计算：

```text
费用 = 总运行小时 × 租机页面当前每小时价格
```

不要在 smoke 前承诺固定金额。

## 10. 最终实验报告结构

GPU 结果产生后，报告按以下顺序完成：

1. 研究背景：搜索增强推理、结果奖励稀疏性与 Search-R1。
2. 原方法复现：课程资源版配置、baseline 训练和七数据集评测。
3. CEGR V1：动机、公式、实现、负结果与统计检验。
4. 失败诊断：singleton `uid`、parser 漂移、代理目标反转及因果边界。
5. 文献调研：Search-R1、GRPO/DAPO、R1-Searcher、ReSearch、LeTS、中间检索奖励与 reward misspecification。
6. CEGR V2：Grouped-EM 与 EFF 公式、局部保证、删除的奖励项。
7. 实验设计：冻结起点、双臂、seed、disjoint pilot、候选选择和最终 700 题。
8. 结果：训练信号、pilot gate、最终 EM/F1、搜索行为、单跳/多跳、Bootstrap 与 McNemar。
9. 消融与归因：`EFF - Grouped-EM` 判断 F1 fallback；`Grouped-EM - frozen` 只称复合 refinement 效果。
10. 局限：单 seed、1.5B、BM25、40 步 warm-start、最终样本规模与 CUDA 非完全确定性。
11. 结论：严格按照 `claim_level` 写；pilot 的 `+0.01` 只是筛选阈值，最终 `+0.02` 才是预声明的有意义改进门槛，不把效率变化冒充准确率提升。

报告中的数值必须从 `pilot-gate.json`、`final-analysis.json` 和训练日志读取。若候选没有通过，不改门槛、不换表述掩盖失败，而是把它作为第二个负结果并据此转向 V3。

## 11. 关键参考文献

- Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning, 2025. <https://arxiv.org/abs/2503.09516>
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, 2024. <https://arxiv.org/abs/2402.03300>
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale, 2025. <https://arxiv.org/abs/2503.14476>
- R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning, 2025. <https://arxiv.org/abs/2503.05592>
- ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning, 2025. <https://arxiv.org/abs/2503.19470>
- Search Wisely: Mitigating Over- and Under-searching in RAG, 2025. <https://arxiv.org/abs/2505.17281>
- LeTS: Learning to Search for LLMs, 2025. <https://arxiv.org/abs/2505.17447>
- An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents, 2025. <https://arxiv.org/abs/2505.15117>
- Ng, Harada, Russell. Policy Invariance under Reward Transformations, 1999. <https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf>
- Gao et al. Scaling Laws for Reward Model Overoptimization, 2023. <https://proceedings.mlr.press/v202/gao23h.html>
