# CEGR V2 文献调研：准确率优先的 Search-R1 奖励重设计

> 更新日期：2026-07-15
> 适用设置：Qwen2.5-1.5B、GRPO、BM25；从冻结的 Search-R1 baseline `global_step_120` warm-start，默认追加 40 步；同预算 Grouped-EM/EFF 双臂；先评测与最终集不重叠的 `7 x 20` pilot，过门后才解锁固定 `7 x 100` 最终评测。实现入口见[V2 契约](../../scripts/improvement_v2/contract.py)、[warm-start 启动器](../../scripts/improvement_v2/train_refinement.sh)、[V2 manager](../../scripts/improvement_v2/cegr_v2_manager.py)与[pilot 数据构造](../../scripts/improvement_v2/prepare_pilot_data.py)。

## 0. 结论先行

**推荐 CEGR V2 先恢复真正的同题 GRPO grouping，再比较两个完全等预算的 warm-start arm：Grouped-EM control 与“EM 优先、全零组 F1 兜底”（EM-First, F1-Fallback，简称 EFF）。** V2 manager 依据 `data_source + extra_info(split,index)` 给同题 5 条 rollout 重写同一个 `uid`，不修改 `verl/`；两臂都使用该 grouping 和同一正式答案 parser。Grouped-EM 始终给纯 EM，EFF 只在同题 5 条全部 EM=0 时改用 token-F1；只要组内出现 EM winner，EFF 的 reward vector 就逐元素等于 Grouped-EM。

**[代码事实]** 当前课程 trainer 在 `n_agent=5` repeat 后为每一行生成独立 `uuid`，而 GRPO advantage 以该 `uid` 分组；singleton 分支固定 `mean=0, std=1`，所以 outcome advantage 实际为 `raw_reward/(1+1e-6)`，而不是同题 5 条内的标准化。baseline/V1 因而并未获得预期的组相对信号。[repeat 与 UUID 生成](../../verl/trainer/ppo/ray_trainer.py)、[singleton advantage](../../verl/trainer/ppo/core_algos.py)、[DataProto repeat 语义](../../verl/protocol.py)

**[设计推断]** EFF 是当前最小、最值得检验的 accuracy-first 奖励，但 grouping 修复本身也会改变优化。因此只有 **EFF-40 对 Grouped-EM-40** 的差才能归因于 F1 fallback；EFF 对冻结 step120 的差同时包含继续训练、grouping 修复与奖励变化。V2 的严格 parser 是恢复 baseline 官方计分语义，而不是额外 treatment。外部依据仍是：Search-R1 使用最终答案 EM；R1-Searcher 与 ReSearch 支持答案 F1 作为部分正确信号；Search-R1 后续经验研究则发现 answer-in-retrieval 中间奖励收益有限且加大权重会降低性能。[Search-R1](https://arxiv.org/abs/2503.09516)、[R1-Searcher](https://arxiv.org/abs/2503.05592)、[ReSearch](https://arxiv.org/abs/2503.19470)、[Search-R1 经验研究](https://openreview.net/forum?id=IQNZIBspz5)

**保证边界：EFF 不保证最终 EM 涨点。** 它只保证：在两臂使用相同 parser、相同共享 `uid` 的前提下，凡组内存在 EM winner，EFF 与 Grouped-EM 的组内 reward vector（进而本次 advantage 计算的输入）完全相同。全零组上的 F1 更新会改变后续 policy 与 rollout 分布，最终准确率只能由盲 pilot 和最终 700 条证伪或支持。

本轮**不推荐**把真正的 potential-based shaping、词典序/约束优化或 Pareto 优化直接并入。它们在理论上更整洁，但都比“共同修复 grouping 后比较 Grouped-EM/EFF”多出状态势函数、分段打分、约束优化器或多策略选择等变量；应作为 EFF pilot 失败后的下一阶段方案。[Ng et al., 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)、[TIPS, ICLR 2026](https://openreview.net/pdf?id=eBMOr6a84z)、[Lexicographic MORL](https://arxiv.org/abs/2212.13769)、[CPO](https://proceedings.mlr.press/v70/achiam17a.html)

## 1. 证据边界与标记

- **[项目事实]**：本次任务给出的 700 条评测结果。工作树中没有对应预测或统计产物，因此本文只转录，不声称独立复算；指标定义与配对检验流程以[项目评测协议](../improvement_experiment_zh.md)为准。
- **[代码事实]**：可由本 worktree 的代码直接核对，均链接到本地一手实现。
- **[文献事实]**：来自原论文、会议论文或官方仓库，均在主张旁给出可点击的一手来源。
- **[设计推断]**：本文根据项目结果与文献作出的 V2 判断，不冒充论文结论。
- **[实验约定]**：为了在有限预算内做 go/no-go 决策而预先设定的门槛，不是文献公认常数。

## 2. V1 结果与因果边界

### 2.1 给定结果

**[项目事实]** 下表来自任务给定数据；700 条样本的固定集合、逐样本配对和统计口径见[评测协议第 10 节](../improvement_experiment_zh.md)。

| 指标 | Search-R1 baseline | CEGR V1 | V1 - baseline |
|---|---:|---:|---:|
| EM | 0.2400 | 0.2071 | **-0.0329** |
| token-F1 | 0.3044 | 0.2645 | -0.0399 |
| `evidence` | 0.3000 | 0.2714 | -0.0286 |
| 平均搜索次数 | 1.1457 | 1.0314 | -0.1143 |
| 重复搜索率 | 0.0514 | 0.0214 | -0.0300 |
| 平均响应长度 | 778.83 | 673.14 | -105.69（-13.57%） |

**[项目事实]** EM 差的 95% CI 为 `[-0.0629, -0.0029]`，exact McNemar `p=0.03975`，且单跳数据集退化更明显。按预注册式判定，这不是“代理指标改善但主指标持平”，而是主指标出现有统计支持的负向结果；原研究笔记也明确把“EM 下降”定义为失败。[V1 研究笔记的成功/失败标准](search_rl_reward_improvement.md)

### 2.2 V1 实际优化了什么

**[代码事实]** V1 将 EM/F1 答案质量、`evidence_answer_coverage` 和行为惩罚做加权和；训练中 EM 权重上升，evidence 权重下降，行为惩罚权重上升。所谓 evidence 只检查任一 `<information>` 块是否包含 gold alias 字符串；行为项处罚空/畸形查询、规范化后的重复查询和第 4 次及以后搜索。[V1 奖励实现](../../scripts/improvement/cegr_reward.py)

**[代码事实]** 总奖励只写到每条 response 的最后一个有效 token。V1 manager 只解码 response，使用带 `re.IGNORECASE` 的 `<answer>...</answer>` 正则并接受单个标签；它逐轨迹打分，不执行组级门控或逐搜索步骤 credit assignment。[V1 reward manager](../../scripts/improvement/cegr_manager.py)、[V1 宽松 parser](../../scripts/improvement/cegr_reward.py)

### 2.3 隐蔽的 grouping 失效：现有 advantage 实际是 raw reward

**[代码事实]** `ray_trainer` 先用 `n_agent=5` 对 batch 做 interleaved repeat；`DataProto.repeat` 会把 `data_source`、`extra_info` 等 non-tensor metadata 一并复制。搜索 rollout 完成后，trainer 却对 repeat 后的每一行各生成一个独立随机 `uuid`，再把 `uid` 交给 GRPO advantage 计算。[训练循环](../../verl/trainer/ppo/ray_trainer.py)、[repeat 实现](../../verl/protocol.py)

**[代码事实]** `compute_grpo_outcome_advantage` 按 `uid` 汇总分数；每个 `uid` 只有一条时，显式设 `mean=0`、`std=1`，然后计算 `(score-mean)/(std+1e-6)`。在课程当前 `n_agent=5`、每条 rollout 独立 `uuid` 的路径上，advantage 因而是 `raw_reward/(1+1e-6)`，数值上等同 raw reward，而非同题 5 条的中心化/标准化。[advantage 调用处](../../verl/trainer/ppo/ray_trainer.py)、[singleton 分支](../../verl/trainer/ppo/core_algos.py)

**[代码事实 + 推断]** baseline 的错误答案 reward=0，至少不会因答案奖励得到正 advantage；V1 的错误轨迹只要 F1/coverage 部分命中后总 reward 仍大于 0，就会把该正值直接作为正 advantage 送入 policy update，而不是被同题正确轨迹压成相对负值。这使“部分命中错误答案被正向强化”成为 V1 退化的具体机制候选。[V1 奖励公式](../../scripts/improvement/cegr_reward.py)、[V1 manager](../../scripts/improvement/cegr_manager.py)

**[设计推断]** 这项发现否定了旧版综述的一个前提：现有 baseline/V1 不能被当作真正的五轨迹组内 GRPO。V2 必须先恢复共享 `uid`；但 grouping 修复本身会改变所有混合成功组的 advantage，所以必须让 Grouped-EM control 与 EFF 同时采用它，不能把 EFF 对历史 baseline 的全部变化归因于 fallback reward。

### 2.4 正式 EM parser 与 V1 parser 不一致

**[代码事实]** 官方 Search-R1 `RewardManager` 解码的是 **full prompt + response**；QA parser 用大小写敏感的正则 `r'<answer>(.*?)</answer>'`，要求全文至少出现两个 match 后取最后一个。课程 prompt 本身含一个 `<answer> Beijing </answer>` 示例，因此模型 response 必须再给出一个大小写完全正确的标签才会被正式计分。[官方 RewardManager](../../verl/trainer/main_ppo.py)、[官方 QA EM parser](../../verl/utils/reward_score/qa_em.py)、[课程 prompt](../../scripts/data_process/nq_search.py)

**[代码事实]** 官方 `normalize_answer` 依次小写化答案文本、去标点、去英文冠词并压缩空白；标签匹配是 case-sensitive，但标签内答案经过 normalization 后不区分大小写。V1 虽复制了相似 normalization，却在 response-only 文本上用 case-insensitive regex 接受单个或错误大小写标签，可能给正式 parser 会判 0 的轨迹正奖励。[官方 normalization](../../verl/utils/reward_score/qa_em.py)、[V1 parser](../../scripts/improvement/cegr_reward.py)

**[V2 契约]** Grouped-EM 与 EFF 必须共用同一严格语义：大小写敏感地从 response 抽取最后一个正式 `<answer>...</answer>`，并与官方 full prompt+response scorer 做 fixture parity；EM 与 token-F1 复用和官方 `normalize_answer` 等价的 normalization，并由 parity 测试防止漂移。固定课程 prompt 恰有一个合法示例标签，因此上述 response-final 提取与官方“全文至少两个 match、取最后一个”在 valid/缺失/错误大小写场景下等价。无合法 final answer 时 EM/F1 都为 0。它修正的是 V1 parser 漂移，并恢复 frozen baseline 的官方计分语义。[V2 manager](../../scripts/improvement_v2/cegr_v2_manager.py)、[V2 reward](../../scripts/improvement_v2/cegr_v2_reward.py)、[parser parity 测试](../../tests/test_cegr_v2_reward.py)

### 2.5 能说与不能说的因果结论

**[设计推断]** 数据支持“效率代理变好而任务准确率反向下降”这一**代理目标反转症状**：搜索更少、更短、更少重复，但 EM、F1 和 `evidence` 同时下降。singleton advantage 使正的 F1/coverage 错误轨迹可被直接强化，为该症状增加了代码层机制；但 V1 同时改变了 F1 混合、evidence 奖励、行为惩罚、parser 和课程权重，单次对照仍不能识别各项贡献，也不足以证明模型进行了有意的 reward hacking。该现象与 imperfect proxy 过度优化后 gold objective 下降的已知结果相容。[Gao et al., ICML 2023](https://proceedings.mlr.press/v202/gao23h.html)、[Amodei et al., 2016](https://arxiv.org/abs/1606.06565)

**[设计推断]** 单跳退化与平均搜索次数下降相容于“过早停止/少搜”的解释，但不是其证明；也可能来自答案风格、F1 混合或训练方差。要确认机制，必须查看 baseline-correct/V1-wrong 翻转样本的搜索轨迹，并按单跳/多跳分层报告，而不能只凭全局均值定因。

**[设计推断]** 后文把当前 `evidence` 更准确地称为 **`answer_in_context`**。答案字符串出现只说明 lexical answer presence，不说明检索片段支持推理链、来源正确或回答中的主张得到归因；HotpotQA、KILT 与 ALCE 分别使用句级 supporting facts、provenance 和 citation correctness/completeness 来单独评价这些性质。[HotpotQA](https://aclanthology.org/D18-1259/)、[KILT](https://aclanthology.org/2021.naacl-main.200/)、[ALCE](https://aclanthology.org/2023.emnlp-main.398/)

## 3. 一手文献给出的约束

### 3.1 搜索型推理 RL：最终答案奖励仍是最稳的锚

**[文献事实]** Search-R1 让模型在推理中自主生成搜索动作，检索返回 token 不参与 policy loss；训练奖励只做最终答案 exact-match，不使用格式奖励或额外神经奖励模型。它因此提供了本项目最直接的 accuracy-aligned baseline。[论文](https://arxiv.org/abs/2503.09516)、[官方代码](https://github.com/PeterGriffinJin/Search-R1)

**[文献事实]** R1-Searcher 采用两阶段 outcome-based RL：第二阶段去掉检索奖励，使用答案正确性和格式奖励。其 Qwen2.5-7B-Base 奖励消融中，EM、CEM、F1 的平均 CEM 分别为 39.7、59.5、60.6；F1 相对 EM 提升 52.6%，同时生成更长。这是“F1 能缓解二元答案奖励，但可能增加长度”的直接证据。[论文第 4.4 节](https://arxiv.org/html/2503.05592)、[官方代码](https://github.com/RUCAIBox/R1-Searcher)

**[文献事实]** ReSearch 同样不需要 reasoning-step 监督，在 GRPO 中使用答案 token-F1 与格式奖励，并把检索结果从生成损失中 mask；其训练与本项目在“搜索—推理交错、结果监督、组相对优化”上同类，但模型规模和数据设置不同。[论文](https://arxiv.org/abs/2503.19470)、[官方代码](https://github.com/Agent-RL/ReSearch)

**[文献事实]** 对 reasoning-search interleaved agents 的 Search-R1 后续经验研究报告：格式奖励可帮助稳定交互格式；以“gold answer 是否出现在检索内容”计算的 intermediate retrieval reward 作用有限，增大其系数会持续降低性能，作者解释为它可能过度限制自然搜索轨迹并偏离最终 outcome。[OpenReview 论文](https://openreview.net/forum?id=IQNZIBspz5)、[arXiv](https://arxiv.org/abs/2505.15117)

**[设计推断]** 最后一项与 V1 的 `answer_in_context` 奖励高度同构，是移除该项的最强外部依据；但其模型、数据和超参数不完全相同，所以只能作为同向证据，不能替代本项目消融。

### 3.2 稀疏终局奖励：先恢复“可排序性”，不必立刻奖励每个动作

**[文献事实]** GRPO 用同一问题的一组输出估计相对 advantage；如果一组结果奖励全相同，组内归一化后没有有效排序信号。DAPO 的 dynamic sampling 因而过滤准确率为 0 或 1 的 prompt groups，只保留能产生有效梯度的组。[DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)、[DAPO 论文](https://arxiv.org/abs/2503.14476)、[DAPO 官方项目](https://dapo-sia.github.io/)

**[代码事实 + 推断]** 这条文献结论以“同一 prompt 的 rollouts 共享 group id”为前提；当前课程 pipeline 的独立 UUID 破坏了该前提。V2 用 `data_source:split:index` 恢复共享 `uid` 后，Grouped-EM 才会在混合组中给 winner 正 advantage、loser 负 advantage，并让全零/全一组归零。[当前 UID 生成](../../verl/trainer/ppo/ray_trainer.py)、[GRPO advantage](../../verl/trainer/ppo/core_algos.py)、[V2 manager](../../scripts/improvement_v2/cegr_v2_manager.py)

**[文献事实]** RUDDER 从更一般的 delayed-reward 角度说明，长时延会恶化 credit assignment，并通过 return-equivalent reward redistribution 把信息前移而保持最优策略；但它需要额外的回报分解模型，远超当前课程复现的最小改动范围。[RUDDER, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html)

**[设计推断]** 当前资源约束下，可以不丢弃全零组，也不训练过程奖励模型：两臂先共同修复 grouping；Grouped-EM 提供纯 EM 因果对照；EFF 仅在 Grouped-EM 仍无排序信号的全零组用 F1 恢复差异。这是本文的工程启发式，不是 DAPO 或 RUDDER 的原方法，也没有最终性能保证。

### 3.3 Reward hacking 与代理奖励反转

**[文献事实]** “Concrete Problems in AI Safety”把 reward hacking 定义为 agent 利用目标函数漏洞获得高奖励而没有完成设计者真正意图；Gao 等人的受控实验进一步显示，优化不完美 proxy 超过某一点后，gold reward 可能下降。[Amodei et al., 2016](https://arxiv.org/abs/1606.06565)、[Gao et al., ICML 2023](https://proceedings.mlr.press/v202/gao23h.html)

**[设计推断]** V1 目前最严谨的表述是“出现了代理反转，reward hacking 尚未证实”。若要升级为 hacking 结论，需要轨迹证据表明模型系统性利用 alias substring、少搜或短答获取更高训练奖励，同时规避真实问答目标。

### 3.4 Potential-based reward shaping：有理论保证，但条件不能省略

**[文献事实]** 经典 PBRS 只在 shaping reward 具有 `F(s,a,s') = gamma * Phi(s') - Phi(s)` 的势差形式及相应 MDP 条件下保证最优策略不变；任意加分项一般可能改变最优策略。[Ng, Harada & Russell, ICML 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

**[文献事实]** episodic setting 还必须处理终止状态势函数，否则边界项可能改变策略偏好；Grześ 给出了有限时域下的修正分析。[Grześ, AAMAS 2017](https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf)

**[文献事实]** TIPS 把该思想专门用于 search-augmented LLM：用训练中模型的 checkpoint 作为 self-teacher，以 gold answer 在每个“推理+工具调用”片段后的对数似然增量构造 turn-level potential。ICLR 2026 论文在 7 个 QA benchmark 上报告，Qwen2.5-7B-Instruct 相对 PPO 的平均 EM/F1 提升约 11.8%/13.6%，额外训练 FLOPs 约 11.8%，且不需要单独训练 reward model。[论文](https://openreview.net/pdf?id=eBMOr6a84z)、[官方代码](https://github.com/ucsd-wang-lab-lm/tips)

**[设计推断]** TIPS 是 V2 之后最值得跟进的方案，但不是本轮首选：现有项目是 1.5B+BM25+自定义 GRPO manager，尚无片段级 advantage 和 self-teacher 打分路径；直接迁移会同时改变奖励时序、计算图和训练成本，难以把效果归因给单一变量。

### 3.5 多目标 RL：准确率与效率不应再用无条件加权和混在一起

**[文献事实]** Lexicographic MORL 按优先级先优化第一目标，再在其近最优集合内优化第二目标；论文同时说明，单个线性标量奖励不能普遍表达这种类别式优先关系。[Skalse et al., 2022](https://arxiv.org/abs/2212.13769)

**[文献事实]** Constrained Policy Optimization 把回报最大化与期望成本约束分开，并给出近似约束满足保证；Pareto Q-learning 则学习一组非支配策略，适用于确实需要展示多种权衡的多目标任务。[CPO, ICML 2017](https://proceedings.mlr.press/v70/achiam17a.html)、[Pareto Q-learning, JMLR 2014](https://jmlr.csail.mit.edu/beta/papers/v15/vanmoffaert14a.html)

**[设计推断]** 本项目已有单一首要指标 EM，且 V1 已显示“效率改善不能补偿准确率下降”。因此近期最接近 lexicographic 思路的低成本做法，是把搜索次数、重复率和长度降为**验收约束/监控项**，而不是继续进入每条轨迹的加权训练奖励。这里借用了优先级思想，并未实现正式的 Lexicographic MORL 或 CPO，不能声称享有其理论保证。

### 3.6 自适应检索与 evidence attribution

**[文献事实]** IRCoT 在多跳 QA 中交错推理与检索，并在四个多跳数据集同时提升 retrieval 与 QA；Adaptive-RAG 则按问题复杂度在“不检索、单步检索、迭代检索”之间路由。二者都反对对所有问题施加同一个固定搜索深度假设。[IRCoT, ACL 2023](https://aclanthology.org/2023.acl-long.557/)、[Adaptive-RAG, NAACL 2024](https://aclanthology.org/2024.naacl-long.389/)

**[文献事实]** R1-Searcher 的训练数据消融还观察到，移除困难问题会同时带来更短生成、更少检索和较差准确率，说明在复杂问题上“多一点搜索”可能是完成任务所需行为，而非天然浪费。[R1-Searcher 第 4.4 节](https://arxiv.org/html/2503.05592)

**[设计推断]** 这解释了为什么 V1 的全局搜索/长度处罚风险很高：单跳、已知事实问题与组合式多跳问题的合理检索预算不同。V2 应先让 policy 由答案结果学习是否继续检索；若以后优化效率，应按问题难度或在准确率非劣约束下处理。

**[文献事实]** 真正的 evidence attribution 至少要求“答案主张—支持片段—来源”的对应关系。HotpotQA 提供句级 supporting facts，KILT 把 downstream task score 与 provenance score 分开，ALCE 也把回答正确性与 citation correctness/completeness 分开评价。[HotpotQA](https://aclanthology.org/D18-1259/)、[KILT](https://aclanthology.org/2021.naacl-main.200/)、[ALCE](https://aclanthology.org/2023.emnlp-main.398/)

**[设计推断]** 在七数据集没有统一 gold supporting passage/provenance 的前提下，不应把 `answer_in_context` 升格成 evidence reward。可继续报告它作诊断；只有建立可核验的 gold support 或 citation entailment 标注后，才值得单独研究 attribution reward。

## 4. 五个 V2 候选/对照比较

**共同前置条件不是候选变量：** 所有 V2 arm 都由同一个 manager 恢复 `data_source + split + index` 共享 `uid`，并使用同一个与官方 scorer 行为一致的严格 EM parser。否则比较会把 grouping/parser 差异误算成 reward 差异。[V2 manager](../../scripts/improvement_v2/cegr_v2_manager.py)、[训练入口](../../scripts/improvement_v2/main_ppo_refinement.py)

| 候选 | 训练奖励/优化方式 | 一手依据 | 对现有代码的改动 | 主要风险 | 本轮结论 |
|---|---|---|---|---|---|
| **A. Grouped-EM** | 共享 `uid` 后始终 `R=official_EM` | Search-R1 的最终答案 EM 是 accuracy-aligned 锚。[Search-R1](https://arxiv.org/abs/2503.09516) | grouping 修复 + parser 对齐；step120 后追加 40 步 | 全零组仍无信号；结果可能与历史 singleton baseline 不同 | **必做因果 control** |
| B. F1-only | 所有组都 `R=token-F1`；移除 evidence 与行为项 | R1-Searcher 的 F1 消融最佳；ReSearch 使用 F1。[R1-Searcher](https://arxiv.org/abs/2503.05592)、[ReSearch](https://arxiv.org/abs/2503.19470) | 在共同修复之上只换标量奖励 | F1 可偏好冗长/部分重叠答案，并改变已有 EM winner 组的 advantage | **可选消融** |
| **C. EFF（推荐）** | 有 EM winner 的组用纯 EM；全零组才用 F1 | 结合 Search-R1 的 EM 锚、F1 证据和 DAPO 的全零组诊断。[Search-R1](https://arxiv.org/abs/2503.09516)、[DAPO](https://arxiv.org/abs/2503.14476) | 与 control 同 manager/config，仅 reward gate 不同 | 无最终涨点或 policy-invariance 保证；全零组 F1 仍可能词面投机 | **V2 treatment** |
| D. TIPS/PBRS | 每轮用 self-teacher gold-answer log-likelihood 势差给分 | 同领域、稠密、论文给出 policy-invariance 分析和 QA 增益。[TIPS](https://openreview.net/pdf?id=eBMOr6a84z) | 中到大：分段、teacher forward、turn-level credit | 约 11.8% FLOPs 开销；迁移到 1.5B+当前 GRPO 未验证 | **后续 V3** |
| E. 词典序/约束式双目标 | 先最大化 EM，再在非劣集合内最小化搜索成本；或给成本加显式约束 | [Lexicographic MORL](https://arxiv.org/abs/2212.13769)、[CPO](https://proceedings.mlr.press/v70/achiam17a.html)、[Pareto Q](https://jmlr.csail.mit.edu/beta/papers/v15/vanmoffaert14a.html) | 大：多目标 optimizer/dual、容差和更多 seed | 小样本下约束估计噪声大；复杂度压过课程预算 | **暂只作评测约束** |

### 4.1 为什么不是简单的 `EM + lambda*F1`

**[设计推断]** 在历史 singleton 路径中，固定 `lambda>0` 会让 F1 部分命中的错误轨迹直接得到正 advantage；恢复 grouping 后，它仍会在已有 exact winner 的组内改变均值、方差和相对 advantage。EFF 把 F1 限制在 Grouped-EM reward vector 全零的组，从而让非全零组与因果 control 完全相同。

### 4.2 为什么不是继续调小 evidence/行为惩罚

**[设计推断]** V1 已在主指标上显著负向，且同类 intermediate retrieval reward 的文献消融随权重增加而下降。仅把权重调小仍保留三个不可辨识变量，得到正负结果都难解释；直接归零能恢复单变量实验。[Search-R1 经验研究](https://openreview.net/forum?id=IQNZIBspz5)

## 5. 推荐方案的可执行定义

### 5.1 两臂共同的 grouping 与 parser

**共享 UID 恢复。** V2 manager 在 reward 计算时读取每行 metadata，不依赖 batch 邻接顺序：

```text
group_uid_i = data_source_i + ":" + extra_info_i.split + ":" + extra_info_i.index
data.non_tensor_batch["uid"][i] = group_uid_i
```

同一原题经 `n_agent=5` repeat 后保留相同 `data_source/split/index`，所以 5 条 rollout 得到同一稳定 `uid`；不同数据集用 `data_source` 防碰撞。metadata 缺失、同一 `uid` 数量不等于 5、或同一 `uid` 内 ground truth 不一致时必须 fail closed。该修改位于 `scripts/improvement_v2` manager，**不修改 `verl/`**；即使 trainer 随后 balance/reorder batch，绑定在每行上的 `uid` 仍可正确分组。[repeat 实现](../../verl/protocol.py)、[trainer 的 reorder 警告](../../verl/trainer/ppo/ray_trainer.py)、[V2 manager](../../scripts/improvement_v2/cegr_v2_manager.py)

**统一正式 parser。** 两臂都从有效 response 以大小写敏感规则取最后一个 `<answer>...</answer>`，并在固定 prompt 模板上通过官方 full-sequence scorer parity；EM 与 F1 均使用官方 `qa_em.normalize_answer` 语义。严禁 control 走官方 `RewardManager`、EFF 走 V1 式 case-insensitive parser，否则 treatment 不再只有 reward gate。[官方 parser/normalization](../../verl/utils/reward_score/qa_em.py)、[V2 reward](../../scripts/improvement_v2/cegr_v2_reward.py)、[双臂入口](../../scripts/improvement_v2/main_ppo_refinement.py)、[parity 测试](../../tests/test_cegr_v2_reward.py)

### 5.2 Grouped-EM 与 EFF 奖励

对问题 `q` 的同组 `K=5` 条 rollout，令：

```text
e_i = official_EM(final_answer_i, gold_aliases_q)            in {0, 1}
f_i = max_token_F1(official_normalize(final_answer_i),
                   official_normalize(gold_aliases_q))        in [0, 1]

Grouped-EM control: C_i = e_i

EFF treatment:
if sum_j(e_j) == 0:
    E_i = f_i
else:
    E_i = e_i
```

**[设计推断]** 该定义只带来以下局部、可审计性质：

1. 两臂对每题使用完全相同的 5 条 rollout grouping 和 EM/F1 标签。
2. 组内只要出现 exact answer，`E` 与 `C` reward vector 逐元素相等；在同一 advantage 实现下，该组的 mean/std 和 normalized advantage 也相等。
3. 全零 EM 组的 Grouped-EM advantage 为 0；只有 F1 不全相同时，EFF 才产生均值约为 0、组内有正有负的相对 advantage，而不是 V1 singleton 路径中的一批 raw positive advantages。
4. 全一 EM 组仍不更新；全零且 F1 也全相同的组同样不制造虚假差异。

**这些性质不保证最终 EM 上升。** EFF 在全零组产生的新更新会改变参数、后续 rollout 和哪些问题进入混合组；最终表现还受 warm-start checkpoint、采样随机性和 1.5B 模型能力影响。保证对象仅是当前 batch 的分组、reward vector 与相应 advantage 等价关系。[V2 契约](../../scripts/improvement_v2/contract.py)

### 5.3 明确移除与保留

**从训练 reward 移除：**

- `answer_in_context`/旧 `evidence_answer_coverage`；
- 空查询、重复查询、第 4 次搜索、搜索次数、响应长度等行为成本；
- EM/F1/evidence/penalty 的课程权重；
- 独立格式奖励。大小写错误或缺失 final `<answer>` 标签直接令 EM/F1=0；格式奖励虽有正面文献结果，但本轮加入会破坏单变量归因。[Search-R1 经验研究](https://openreview.net/forum?id=IQNZIBspz5)

**保留为评测与诊断：** EM、F1、`answer_in_context`、平均搜索次数、重复/无效搜索率、响应长度、单跳/多跳分层和逐样本翻转。配对 bootstrap 与 exact McNemar 继续按现有脚本口径报告。[评测协议](../improvement_experiment_zh.md)

### 5.4 理论地位

**[设计推断]** EFF 是“accuracy-first 的组级门控启发式”，**不是** PBRS、正式词典序 MORL 或 constrained RL。它只保证非全零组的有限样本 reward/advantage vector 与 **Grouped-EM control** 相同；它不保证训练轨迹、全局最优策略或最终 EM 不变，更不保证涨点。全零组上的 F1 本来就是有意加入的新偏好，该边界必须在报告中保留。

## 6. 风险与可证伪预测

### 6.1 主要风险

1. **grouping 修复也会改变结果**：共享 `uid` 把历史 raw-reward 更新改成真正的组相对更新。Grouped-EM control 只能隔离 EFF 与纯 EM 的差，不能把 control/历史 baseline 差进一步拆成 grouping 与追加训练各自的贡献；严格 parser 则用于保持与 baseline 官方计分一致。[trainer](../../verl/trainer/ppo/ray_trainer.py)、[官方 parser](../../verl/utils/reward_score/qa_em.py)
2. **parser parity 依赖 prompt 不变量**：response-final 提取与官方 full-sequence parser 等价，依赖每个实际 prompt 恰有一个大小写正确的示例标签。必须在七个数据源的真实 prompt 上审计；模板变化时应直接回退到 full prompt+response scorer。[课程 prompt](../../scripts/data_process/nq_search.py)、[官方 parser](../../verl/utils/reward_score/qa_em.py)
3. **词面投机**：token-F1 可能偏好包含多个候选词的长答案。R1-Searcher 已观察到 F1 训练生成更长，因此长度必须监控，但不应立刻重新放回 reward。[R1-Searcher](https://arxiv.org/html/2503.05592)
4. **信号仍稀疏**：若全零 EM 组的 F1 也几乎总为 0 或常数，EFF 不会解决 credit assignment；此时才有理由升级到 TIPS 或合格的过程监督。[TIPS](https://openreview.net/pdf?id=eBMOr6a84z)
5. **小模型与 warm-start 外推**：R1-Searcher/ReSearch/TIPS 的关键结果主要来自 3B、7B 或更大模型，且不等同于从已训练 step120 checkpoint 再追加 40 步，不能假定 Qwen2.5-1.5B 获得同等收益。[R1-Searcher](https://arxiv.org/abs/2503.05592)、[ReSearch](https://arxiv.org/abs/2503.19470)、[TIPS](https://arxiv.org/abs/2603.22293)
6. **分组碰撞或缺失**：只按 `index` 会跨数据集/分片碰撞，依赖相邻顺序又会被 `_balance_batch` 打乱。必须使用完整 `data_source+split+index`，校验每组数量、问题和 ground truth 一致。[V2 manager](../../scripts/improvement_v2/cegr_v2_manager.py)、[trainer reorder](../../verl/trainer/ppo/ray_trainer.py)
7. **效率反弹**：移除行为惩罚后，搜索次数和长度可能回升。这不是自动失败；只有越过预设资源护栏或准确率仍不升才失败。Adaptive-RAG 与 IRCoT 均表明合理检索深度依问题复杂度而变。[Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/)、[IRCoT](https://aclanthology.org/2023.acl-long.557/)

### 6.2 预先声明的可证伪预测

- **P0，grouping 修复预测**：两个 arm 的每个训练 `uid` 都应恰有 5 条，singleton 比例为 0，且 `uid` 集合/组成员完全一致。任一违反都是实现失败，不进入训练。
- **P1，parser 一致预测**：两臂的 EM 标签必须与官方 full prompt+response `qa_em` scorer 一致；response-only 宽松 parser 会接受、但官方 parser 拒绝的错误大小写/单标签样例，在 V2 中必须得到 EM=F1=0。
- **P2，局部等价预测**：在所有 `sum(EM)>0` 的组上，EFF 与 Grouped-EM 的逐轨迹 reward、组均值、组标准差和 normalized advantage 应逐项相同。任何差异都否证实现，不是实验结果。
- **P3，稀疏性预测**：全零 EM 组中应存在一批 `std(F1)>0` 的组；这些组在 Grouped-EM 下 advantage 全零，在 EFF 下产生组内正负相对 advantage。若低于下文 10% 信号门槛，EFF 的核心机制不值得花训练预算。
- **P4，准确率筛选预测**：同起点、同预算 step120+40 pilot 中，EFF 相对 Grouped-EM 在独立 140 题上至少增加 2 个净 exact correct，且 F1 不下降；否则 F1 fallback 的项目假设未获支持。该门槛只是筛选，不是显著性证明。
- **P5，归因与选择预测**：最终只有 `EFF-40 > Grouped-EM-40` 才支持 F1 fallback 的 reward 贡献；`Grouped-EM-40 > frozen-step120` 只支持“group-corrected 继续训练”这一复合候选，不能拆出 grouping 与追加训练各自的贡献。独立 pilot 先按该边界锁定主候选；若只出现效率改善而 EM/F1 不升，则两个候选都判失败。

## 7. 小规模实验门槛

### 7.1 Phase 0：零训练审计

在任何 GPU 正式训练前，用构造样例及首个真实 rollout batch 同时审计 Grouped-EM 与 EFF：[V2 reward audit](../../scripts/improvement_v2/audit_reward_safety.py)

- **grouping 硬门槛**：每个 `data_source:split:index` 恰有 5 条；不存在 singleton、跨题碰撞或组内 ground truth 不一致；两臂的 `uid` 与成员逐项相同。
- **parser 硬门槛**：V2 EM 与官方 scorer 在 valid、缺标签、错误大小写标签、多标签样例上逐项一致；V1 宽松 parser 与官方不一致的 fixture 必须被显式保留，防止回归。
- **reward 硬门槛**：所有 `sum(EM)>0` 组满足 `E_EFF == C_Grouped-EM` 且 advantage 相同；全零组的 control advantage 全 0；无 NaN/越界奖励。
- **[实验约定] 信号门槛**：在全零 EM 组中，至少 10% 满足 `std(F1)>0`。低于 10% 则不花 40-step 预算，转做 TIPS 可行性或训练数据难度分析。
- 输出分层统计：singleton 比例、全零/混合/全一组占比、全零组 F1 均值与方差、严格 parser 拒绝率、单跳/多跳占比。它们是机制诊断，不是新奖励。

### 7.2 Phase 1：冻结 step120 后的等预算训练

**[实验约定]** 冻结并记录现有 Search-R1 baseline `global_step_120` 的 hash；从这一个 checkpoint 分叉训练 `Grouped-EM-40` 与 `EFF-40`。两臂都追加 **40 步**，共享 manager grouping、严格 parser、训练数据顺序、BM25、batch、group size、学习率、KL、mask、max turns、driver seed 42、vLLM engine seed 42、保存频率和硬件。V2 专用 worker 只覆盖 engine 构造 seed，不设置逐请求 sampling seed，以保留同题 rollout 多样性。唯一 treatment 差异是全零组使用 EM 还是 F1。[V2 契约](../../scripts/improvement_v2/contract.py)、[训练启动器](../../scripts/improvement_v2/train_refinement.sh)、[冻结检查](../../scripts/improvement_v2/freeze_v1.py)

**当前默认不是重新从 Qwen 起点训练 120 步。** 历史 step120 checkpoint 是共同 warm-start；pilot 不追加到 120 步，也不把 `EFF-40` 与“从头 40 步”混比。若未来研究从头训练，必须另立实验，不覆盖本 V2 结论。

### 7.3 Phase 2：不触碰最终 700 的盲 pilot

先用固定脚本生成与最终 700 的 `(data_source, split, index)` **完全不重叠**的 `7 x 20 = 140` 题 pilot，并冻结 manifest/hash。只在该集合上评测 `frozen-step120`、`Grouped-EM-40`、`EFF-40`；三者样本 ID 必须完全一致。[pilot 数据构造](../../scripts/improvement_v2/prepare_pilot_data.py)、[pilot 评测](../../scripts/improvement_v2/evaluate_pilot.sh)、[pilot gate](../../scripts/improvement_v2/pilot_gate.py)

**解锁最终 700 采用预声明的层级候选选择：**

1. `EFF-40` 相对 frozen baseline 和 Grouped-EM 的 `Delta EM` 均至少 `+0.01`，相对两者的 F1 均不下降、单跳均无净 exact loss，同时满足 evidence、有效搜索、重复搜索、总搜索与响应长度护栏时，选择 `eff`；
2. 否则，若 `Grouped-EM-40` 相对 frozen baseline 的 `Delta EM >= +0.01`，F1 不下降、单跳无净 exact loss，并满足同类行为护栏，则选择 `grouped_em`；
3. 若两个候选都不满足，`selected_candidate=null`，停止且不查看最终 700。

140 题上恰好 `+1` 个净 exact correct 为 `1/140=0.0071`，仍低于 `+0.01` 门槛；至少净增加 2 题才可能过准确率门。该阈值仅用于候选筛选。gate 同时绑定三份 pilot 轨迹的 SHA-256；一旦写入，只能验证而不能覆盖。若两个候选都失败，不能根据 pilot 样例调 parser、reward 或护栏再重复撞 gate；若另做第二 seed，必须新立实验协议，不能覆写本次 no-go 结论。[gate 实现](../../scripts/improvement_v2/pilot_gate.py)、[锁验证](../../scripts/improvement_v2/verify_pilot_gate.py)

### 7.4 Phase 3：过门后一次性最终 700

pilot 通过后，冻结 baseline checkpoint 与两个 `step120+40` checkpoint 都通过同一个 V2 严格入口评测最终 `7 x 100 = 700`，共享数据、batch、driver seed 和 vLLM engine seed。V1 已冻结的历史 baseline 轨迹另做严格 parser 离线重评分，只用于报告 mismatch，不进入主比较。不得再训练、换 seed、改 parser、换候选或调 gate；最终入口先验证 final parquet hash 与不可覆盖的 pilot 锁。逐样本配对、bootstrap 和 exact McNemar 继续沿用现有口径。[最终评测入口](../../scripts/improvement_v2/evaluate_final.sh)、[严格重评分](../../scripts/improvement_v2/rescore_frozen_baseline.py)、[三方归因分析](../../scripts/improvement_v2/final_analysis.py)、[原配对统计协议](../improvement_experiment_zh.md)

- **pilot 选择 EFF**：主结论要求 EFF 同时高于 Grouped-EM 和 frozen step120；只有前一个差支持 F1 fallback，后一个差说明最终模型优于未追加训练的 baseline。
- **pilot 选择 Grouped-EM**：主结论只检验 Grouped-EM 对 frozen baseline；可称“group-corrected refinement 复合候选”，不得把差值完全归因于 grouping，也不得宣称 F1 fallback 有效。
- **EFF 仅相对 continuation 有效**：若最终 EFF 高于 Grouped-EM、但不高于 frozen baseline，说明 fallback 至多缓解继续训练退化，仍未得到可部署涨点。
- **预声明最低成功**：pilot 的 `Delta EM >= +0.01` 只负责筛选；固定 700 题上，被锁定候选必须在相关主比较中达到 `Delta EM >= +0.02`，并同时满足 F1、单跳及搜索/重复/长度护栏，才进入 improvement claim。[final analysis](../../scripts/improvement_v2/final_analysis.py)
- **较强证据**：主比较的 paired bootstrap 95% CI 下界大于 0 且 exact McNemar `p<0.05`；EFF 被选中时，baseline 和 Grouped-EM 两个比较都须满足。任何仅搜索/长度改善而 EM 不升的结果均按失败报告。

`+0.02` 沿用 V1 研究笔记的有意义改进阈值；它与 `+0.01` pilot 筛选阈值用途不同。双 comparator 的归因规则由 V2 新增，避免把 grouping 修复或普通继续训练误报成 reward 收益。[原成功标准](search_rl_reward_improvement.md)、[V2 final analysis](../../scripts/improvement_v2/final_analysis.py)

## 8. 决策

**[设计推断]** CEGR V2 的最小可信实验不是单跑 EFF，而是：**在 `scripts/improvement_v2` manager 内恢复同题共享 `uid` 与官方严格 parser；从同一个 frozen step120 分叉 Grouped-EM-40 和 EFF-40；只把 EFF 相对 control 的差归因于全零组 F1 fallback；用独立 pilot 在两个候选中预先锁定最终主方法。** EFF 保留最终答案 EM 作为非全零组的锚，evidence 与效率退出训练 reward、只作诊断/护栏。两个候选都没有最终涨点保证；独立 140 题 gate 与固定 700 题结果决定哪项假设获得支持。
