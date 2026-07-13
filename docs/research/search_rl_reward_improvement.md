# Search-R1 改进调研：多跳检索强化学习中的奖励设计

更新日期：2026-07-13

## 1. 调研问题与边界

本调研聚焦一个可在现有单卡 A800、Qwen2.5-1.5B-Instruct、GRPO、CPU BM25 条件下验证的问题：

> Search-R1 仅使用最终答案 Exact Match（EM）时，能否通过低成本、可解释、抗投机的课程式奖励，改善稀疏反馈和多跳搜索行为？

范围覆盖开放域多跳问答中的迭代检索、推理与检索交错、可学习搜索代理、结果奖励、过程奖励和课程奖励。检索器升级、扩大模型、扩大语料虽然可能提高绝对分数，但会改变多个实验变量，不作为本轮首选改进。

## 2. 领域演进

### 2.1 从一次检索到迭代查询

开放域多跳问题的后续检索目标通常不会直接出现在原问题中。GoldEn Retriever 通过“读当前上下文，再生成缺失实体查询”的方式迭代检索，说明自然语言查询本身可以成为可解释的中间动作。[Qi et al., EMNLP 2019](https://aclanthology.org/D19-1261/)

IRCoT 进一步把 CoT 推理句与检索交错：已推导内容决定下一次检索，而新证据反过来更新推理。论文在 HotpotQA、2WikiMultiHopQA、MuSiQue 和 IIRC 上同时观察到检索和问答提升，奠定了 Search-R1 所采用的“推理—搜索—观察—继续推理”交互范式。[Trivedi et al., ACL 2023](https://aclanthology.org/2023.acl-long.557/)

### 2.2 从提示式代理到自适应检索

Self-RAG 训练模型生成检索和批判 token，使模型可以按需检索、判断段落相关性并反思生成质量。它说明固定次数、无条件检索并非总是最优，模型还需要学习“是否搜”和“证据是否有用”。[Asai et al., ICLR 2024](https://openreview.net/forum?id=hSyW5go0v8)

Search-o1 将搜索嵌入长推理过程，并增加 Reason-in-Documents 模块压缩冗长检索结果，指出原始文档中的冗余和噪声会破坏推理连贯性。[Li et al., 2025](https://arxiv.org/abs/2501.05366)

ChainRAG 将多跳失败归纳为“lost-in-retrieval”：子问题分解遗漏关键实体会使后续链条中断。它采用渐进检索和上下文感知重写补全实体，并在 MuSiQue、2Wiki 和 HotpotQA 上验证。[Zhu et al., ACL 2025](https://aclanthology.org/2025.acl-long.1089/)

DEC 同样强调轻量模型上的子问题分解、上下文重写和关键词提取，表明“更精确的查询”在资源受限场景比盲目增加上下文更合适。[Ji et al., Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.846/)

### 2.3 用强化学习获得搜索行为

Search-R1 不依赖搜索轨迹监督数据，而让模型在多轮 rollout 中生成 `<search>` 与 `<answer>` 动作。检索结果 token 被 loss mask 排除，训练只优化模型自己产生的 token；原文消融显示 masking 对七数据集结果很重要。其奖励仅判断最终答案 EM，不使用格式奖励或神经奖励模型。[Jin et al., COLM 2025 / arXiv v5](https://arxiv.org/abs/2503.09516)

ReSearch 也使用无推理步骤标注的强化学习，但最终答案采用 token-F1，并附加格式奖励；论文以 MuSiQue 训练，在四个多跳数据集评测，说明连续的部分正确反馈可以替代纯二元 EM。[Chen et al., 2025](https://arxiv.org/abs/2503.19470)

R1-Searcher 采用两阶段强化学习：第一阶段鼓励正确调用检索和格式，第二阶段去掉检索奖励并加入答案奖励。其奖励消融报告 F1 奖励优于 CEM 和 EM，最高相对 EM 奖励提升 52.6%，这是本项目引入 token-F1 的最直接证据。[Song et al., 2025](https://arxiv.org/abs/2503.05592)

RQR 将小模型查询重写表述为强化学习，并用半规则奖励训练 Qwen2.5-1.5B/7B；在 BM25 条件下，小模型也能改善 reasoning-intensive retrieval。这与我们的 1.5B + BM25 条件高度相关。[Qin et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1078/)

### 2.4 结果监督、过程监督与课程奖励

结果监督便宜、客观，但多跳轨迹长且最终正确率低时，大量 rollout 得到完全相同的零奖励。Search-R1 的纯 EM 属于这一类，优点是目标清晰，缺点是难以区分“答案差一点”“已经检到答案但读取失败”和“搜索完全无关”。

RAG-Gym 使用步骤级偏好和过程奖励模型，偏好标准包括相关性、进展和非冗余；实验说明高质量过程监督能改善信息搜索代理，但 GPT-4o 标注、奖励模型训练和额外 critic 对课程复现过于昂贵。[Xiong et al., 2025](https://arxiv.org/abs/2502.13957)

EVO-RAG 将相关文档命中、子查询重叠、步骤成本和答案正确性等七个信号组合，并从 Discovery 阶段逐渐转向 Refinement 阶段：早期鼓励探索，后期提高准确性与效率权重。其核心思想适合本项目，但原方法需要 gold document、查询 embedding、外部 verifier 和多头偏好模型，不能原样低成本复现。[Ji et al., 2025](https://arxiv.org/abs/2505.17391)

## 3. 候选方向比较

| 方向 | 预期收益 | 新资源 | 与基线公平性 | 主要风险 | 决策 |
|---|---:|---:|---:|---|---|
| 换 3B/7B 模型 | 高 | 高 | 低 | 无法区分规模收益 | 不选 |
| BM25 换 E5/混合检索 | 中到高 | 中到高 | 低 | 改变检索器和显存条件 | 后续消融 |
| 查询分解 SFT | 中 | 中 | 中 | 需要轨迹或教师数据 | 后续工作 |
| 神经过程奖励模型 | 潜在高 | 高 | 中 | 标注和 critic 成本高 | 不选 |
| 只把 EM 换成 F1 | 中 | 几乎无 | 高 | 可能只优化长答案 | 作为消融 |
| CEGR 课程式混合奖励 | 中 | 几乎无 | 高 | 代理信号可能被投机 | **首选** |

## 4. 现有结果对方向的支持

现有资源受限基线的七数据集平均 EM 从 0.104 提升到 0.250，说明 Search-R1 主流程有效。但不同任务增益不均衡：NQ、TriviaQA、PopQA 提升 0.194、0.235、0.260；MuSiQue、Bamboogle 只提升 0.011、0.019。后两者更依赖组合式多跳检索，因此当前瓶颈更像是搜索过程与稀疏反馈，而不是“模型完全不会按格式调用搜索”。

该诊断支持奖励改进，但不能预先保证显著提升。尤其训练只含 NQ + HotpotQA，MuSiQue/Bamboogle 仍是分布外任务；CEGR 若只改善训练集代理信号，也可能不泛化。

## 5. CEGR 方法

CEGR 全称 Curriculum Evidence-Guided Reward。对第 `t` 次参数更新的轨迹 `y`，定义：

```text
A_t = beta_t * EM(y, a*) + (1 - beta_t) * F1(y, a*)
R_t = (1 - eta_t) * A_t + eta_t * H(y, a*) - lambda_t * P(y)
```

其中：

- `EM` 是与基线相同的规范化精确匹配；
- `F1` 是预测答案与多个 gold alias 的最大 token-F1；
- `H` 是证据覆盖：任一 `<information>` 块包含信息量足够的 gold alias 时为 1，否则为 0；`yes/no/true/false` 等短或高频答案禁用该项；
- `P` 是搜索行为惩罚，由空/畸形查询、规范化后完全重复查询和第四次搜索组成，并归一化到 `[0,1]`；
- `beta_t` 从 `0.60` 线性增加到 `0.90`；
- `eta_t` 从 `0.15` 线性下降到 `0.05`；
- `lambda_t` 从 `0.02` 线性增加到 `0.08`。

设计含义是：早期用 F1 与证据命中打破大量零奖励，允许模型发现搜索链；后期把优化重点逐步还给严格 EM，并压制重复和无效调用。正确 EM 轨迹即使没有证据命中也始终高于仅命中证据但答案错误的轨迹，避免过程代理目标盖过最终任务。

## 6. 新意与限制

CEGR 不是声称发明 F1、证据奖励或课程调度，而是面向 Search-R1 资源受限复现做的组合与约束：

1. 不引入教师模型、reward model、dense retriever 或 gold supporting-document ID；
2. 直接从 Search-R1 已有 rollout 中计算信号，额外计算成本近似为零；
3. 奖励权重保证最终答案优先，并对短答案证据泄漏进行过滤；
4. 训练复合奖励与最终评测指标分离，仍以固定七数据集 EM 为主指标。

主要限制：答案字符串出现在文档中只是“检到答案”的弱代理，不等价于检到完整支持证据；英文 token-F1 不度量语义等价；单 seed、每数据集 100 条的课程实验统计功效有限；奖励塑形可能使输出变长或复制证据。因此必须同时报告 EM、F1、证据覆盖、搜索次数、重复/无效搜索和响应长度。

## 7. 可证伪假设与判断标准

主假设 H1：相同 120 步预算下，CEGR 的七数据集宏平均 EM 高于原 Search-R1-EM 基线。

机制假设 H2：CEGR 提高 token-F1 与 evidence coverage，并降低 duplicate/invalid search rate；多跳四数据集的平均增益高于单跳三数据集。

效率假设 H3：后期行为惩罚不会增加平均有效搜索次数和响应 token 数。

预先规定判断：

- **有意义改进**：宏平均 EM 至少 `+0.02`，且四个多跳集平均不下降；
- **较强证据**：宏平均 EM `+0.03` 以上，配对 bootstrap 95% CI 不跨 0，且 McNemar 检验支持；
- **仅机制改善**：EM 未达 `+0.02`，但 F1/证据覆盖明显提高且搜索成本不升；
- **失败**：EM 下降，或只提高代理奖励而重复搜索、长度显著增加。失败结果也应如实报告，而不是重新定义成功指标。

## 8. 公平实验协议

Baseline 与 CEGR 保持：Qwen2.5-1.5B-Instruct、GRPO、BM25、NQ+HotpotQA、batch 32、group 5、120 次更新、max turns 4、top-k 3、学习率、KL、mask、seed 42 和同一固定七数据集评测子集。唯一主动变量为训练奖励。

需要三组结果：

1. `Pre-RL`：已有初始模型结果，不必重训；
2. `Search-R1-EM`：已有 120 步 checkpoint，重新跑一次逐样本评测；
3. `Search-R1-CEGR`：从同一个初始模型训练 120 步并逐样本评测。

建议消融（预算允许时各 40 或 60 步）：`F1-only`、`EM+F1`、`CEGR without behavior penalty`。完整 120 步只要求 baseline 和 CEGR，避免把预算拆得过散。

## 9. 结论

CEGR 的预期效果不是来自增加参数或检索资源，而是让 GRPO 在早期获得更可区分的轨迹排序，并在后期恢复对严格答案和搜索效率的偏好。R1-Searcher 的 F1 消融、ReSearch 的 F1 奖励、RAG-Gym 的非冗余过程偏好和 EVO-RAG 的 Discovery-to-Refinement 调度共同提供了文献依据；现有基线在多跳集上的弱增益提供了项目内证据。它具有低成本、单变量、可解释和可证伪四个优点，适合作为课程报告的第一项改进。
