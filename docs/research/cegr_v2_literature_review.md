# CEGR V2 文献调研与设计依据

## 1. 问题定义

Search-R1 将搜索动作嵌入语言模型的多轮推理轨迹，使用最终答案 Exact Match（EM）作为结果奖励。该奖励与最终问答目标直接一致，但同一问题的候选全部答错时，二元 EM 无法区分“部分接近答案”和“完全无关”的轨迹。

本项目的 CEGR V1 尝试用 token-F1、答案字符串证据覆盖和搜索行为惩罚提供稠密反馈。固定 700 题上，V1 的响应长度、搜索次数和重复搜索下降，但 EM 从 0.2400 降至 0.2071，且下降具有统计支持。这说明效率代理发生了变化，却没有改善最终问答目标。

## 2. 一手文献证据

### 2.1 最终答案奖励是稳定锚点

Search-R1 使用简单的 outcome reward 训练模型在推理中自主生成多轮查询，并对检索返回 token 做策略损失屏蔽。它没有依赖复杂的过程奖励模型，因此本项目继续把最终答案 EM 作为首要指标和奖励锚点。

- Search-R1: <https://arxiv.org/abs/2503.09516>

### 2.2 F1 可以提供部分信用，但不能无条件替代 EM

R1-Searcher 和 ReSearch 均探索了 token-F1 类结果奖励。它们说明词级重叠可以缓解纯二元奖励的稀疏性，但 F1 也可能奖励“包含部分正确词语但最终答案错误”的输出。因此，CEGR V2 不把 EM 与 F1 固定加权，而只在 EM 无法排序的组中使用 F1。

- R1-Searcher: <https://arxiv.org/abs/2503.05592>
- ReSearch: <https://arxiv.org/abs/2503.19470>

### 2.3 GRPO 依赖正确的同题分组

GRPO 使用同一问题的一组输出估计相对 advantage。DeepSeekMath 给出了 GRPO 的基本形式；DAPO 的动态采样进一步强调，全 0 或全 1 的奖励组缺少有效排序信号。

代码审计发现，课程流程在生成同题 5 条 rollout 后给每条轨迹分配了不同 `uid`，导致组相对优势退化为近似单条原始奖励。CEGR V2 因此先用 `data_source + split + index` 恢复共享 `uid`，再讨论 F1 fallback。

- DeepSeekMath: <https://arxiv.org/abs/2402.03300>
- DAPO: <https://arxiv.org/abs/2503.14476>

### 2.4 中间检索奖励可能偏离最终目标

Search-R1 的后续经验研究报告，中间检索奖励影响有限，增大权重可能降低最终性能。CEGR V1 的“答案字符串是否出现在检索文本中”与该类奖励相近，而且字符串出现不能证明检索文档真正支持推理链。因此 V2 删除训练期 evidence reward，只把证据覆盖保留为评测诊断指标。

- Search-R1 empirical study: <https://arxiv.org/abs/2505.15117>

### 2.5 代理目标不能代替真实目标

Reward overoptimization 研究说明，持续优化不完美代理可能提高代理分数却降低真实目标。V1 的“更少搜索、更短回答、准确率下降”与这种代理反转现象一致，但单次实验不足以证明模型有意利用奖励漏洞。

- Gao et al., 2023: <https://proceedings.mlr.press/v202/gao23h.html>

## 3. CEGR V2 方法

对同一问题的第 `i` 条 rollout，定义：

```text
e_i = exact_match(answer_i, gold)  in {0, 1}
f_i = token_f1(answer_i, gold)     in [0, 1]
```

CEGR V2 使用 EM-First, F1-Fallback（EFF）奖励：

```text
if any(e_j == 1 for j in prompt_group):
    reward_i = e_i
else:
    reward_i = f_i
```

随后由 GRPO 计算组相对优势：

```text
advantage_i = (reward_i - group_mean) / (group_std + 1e-6)
```

该设计保证：只要组内存在 exact winner，EFF 与纯 EM 的奖励向量完全相同；F1 只在全零组中恢复候选间的相对排序。V2 不奖励证据字符串出现，不处罚正常搜索次数，也不奖励较短回答。

## 4. 实际实验协议

最终执行的是单新臂等更新实验：保留已有 Search-R1 baseline，不重新训练；V2 从原始 Qwen2.5-1.5B-Instruct 起跑 120 步。两者都接受 120 次更新，但 baseline 来自历史运行，因此比较估计的是“分组修复 + EFF”的组合效果，不能单独识别 F1 fallback 的因果贡献。

固定条件如下：单张 A800 80GB、Qwen2.5-1.5B-Instruct、GRPO、CPU BM25 Top-3、最多 4 轮搜索、batch 32、group size 5、学习率 1e-6、seed 42。最终评测使用七个数据集各 100 题，共 700 题，并报告配对 bootstrap 和精确 McNemar 检验。

## 5. 实验结论对设计假设的反馈

CEGR V2 的平均 EM 从 baseline 的 0.2371 变为 0.2257，差值 -0.0114，95% 置信区间为 [-0.0400, 0.0157]，McNemar `p=0.4926`。主指标没有改善。平均响应长度下降 14.93%，多跳 EM 略升 0.0050，而单跳 EM 下降 0.0333。

因此，实验支持“EFF 能改变组内训练信号和推理效率”，但不支持“统一 F1 fallback 能提高总体准确率”。后续若继续研究，应在独立协议中采用按问题复杂度或检索证据状态门控的 fallback，并增加同起点 Grouped-EM 对照；不能通过提高 F1 权重或只报告效率指标把本次负结果改写为成功。
