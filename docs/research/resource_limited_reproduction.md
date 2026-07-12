# Search-R1 课程级资源受限复现：主来源核查

核查日期：2026-07-12
核查范围：Search-R1 论文 v1、作者官方 GitHub、Qwen 官方模型页，以及本仓库 `codex/course-reproduction` 分支的对应源码。本文只判断方案是否有官方依据及其科学边界，不代表该配置已经完成 GPU 实跑。

## 结论摘要

| 问题 | 核查结论 |
|---|---|
| Search-R1 是否支持 GRPO？ | **明确支持。**论文给出 GRPO 目标、训练设置和 PPO/GRPO 对比；官方仓库也提供 `train_grpo.sh`。 |
| 是否支持 BM25？ | **官方最新版明确支持。**在线检索服务器包含 `BM25Retriever`，索引构建器可生成 Pyserini/Lucene BM25 索引。 |
| 是否支持 Qwen2.5-1.5B-Instruct？ | **架构上兼容，但不是论文或官方脚本验证过的具体实验档位。**官方框架支持 Qwen2/Qwen2.5，1.5B 官方模型是 `Qwen2ForCausalLM`；应写“资源适配”，不能写“作者官方 1.5B 配置”。 |
| 单卡 A800 + 1.5B + GRPO + CPU BM25 是否仍是 Search-R1？ | **是，前提是保留多轮搜索交互、检索 token loss masking 和结果 EM 奖励。**它属于课程级方法复现，不属于论文 v1 主表的严格数值复现。 |

推荐对本实验的准确命名是：

> Search-R1 核心方法的课程级资源受限复现：Qwen2.5-1.5B-Instruct、GRPO 与 CPU BM25。

## 1. Qwen2.5-1.5B-Instruct 的支持边界

作者仓库首页声明支持不同 LLM（包括 Qwen2.5），底层模型注册表把 `qwen2` / `Qwen2Config` 列入 remove-padding 支持范围。Qwen 官方模型页确认 `Qwen/Qwen2.5-1.5B-Instruct` 是 1.54B 参数的因果语言模型，可由 `AutoModelForCausalLM` 和 vLLM 加载；其配置架构属于 `Qwen2ForCausalLM`。因此，把训练脚本的模型路径改为该模型有清晰的架构依据。

但必须保留以下限定：

- Search-R1 v1 的论文实验对象是 Qwen2.5-3B、Qwen2.5-7B 和 LLaMA3.2-3B，没有报告 1.5B。
- 作者的 GRPO 示例明确列出 Qwen2.5-3B/7B，没有列出 1.5B。
- 所以“支持”表示官方框架和模型架构兼容，**不等于作者已经对 1.5B 完成训练稳定性和最终精度验证**。
- Qwen 官方指出 `transformers<4.37.0` 会因不认识 `qwen2` 而报错；课程环境必须满足该下限，并以仓库锁定版本的实际冒烟测试为准。

主来源：[Search-R1 官方 README](https://github.com/PeterGriffinJin/Search-R1)、[官方模型注册表](https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/models/registry.py)、[Qwen2.5-1.5B-Instruct 官方模型页](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)。

## 2. GRPO 是否属于原方法

答案是明确的“是”。论文 v1 第 3.1 节同时定义 PPO 与 GRPO；GRPO 使用同一输入的多条采样回答之间的相对奖励估计优势，不需要 PPO 的独立 value/critic 模型。论文实验设置给出 GRPO policy learning rate 为 `1e-6`、每个 prompt 采样 5 条回答，并在表 3 报告 Qwen2.5-3B-Instruct + GRPO 的七数据集平均 EM 为 0.365。

官方 GRPO 脚本的关键开关包括：

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.n_agent=5
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.state_masking=true
```

本仓库 trainer 也明确在 `adv_estimator == 'grpo'` 时设置 `use_critic = False`。因此，GRPO 确实比 PPO 少一个 critic，但仍保留 actor、rollout、reference policy、优化器状态和多条采样；“单卡更省显存”是合理判断，**不代表未经实测即可保证 batch 32/64 不 OOM**。

论文还报告：GRPO 通常收敛更快，但可能比 PPO 不稳定；曾在 LLaMA3.2-3B-Instruct 上出现 reward collapse。因此课程实验必须监测 reward、KL、响应长度、有效搜索率和 NaN，不能只看最终 EM。

主来源：[Search-R1 论文 v1，第 3.1、4.3、5.1 节及表 3](https://arxiv.org/pdf/2503.09516v1)、[官方 GRPO 脚本](https://github.com/PeterGriffinJin/Search-R1/blob/main/train_grpo.sh)、[官方 trainer](https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/trainer/ppo/ray_trainer.py)。

## 3. BM25 支持及在线服务器启动

官方最新版明确支持本地稀疏 BM25。`retrieval_server.py` 的 `BM25Retriever` 使用 Pyserini `LuceneSearcher`，运行在 CPU；`get_retriever()` 在 `retrieval_method == "bm25"` 时选择它。`index_builder.py` 使用 Pyserini 的 `JsonCollection` 和 `DefaultLuceneDocumentGenerator` 构建 Lucene 索引。

### 3.1 构建索引

语料仍应使用与实验约定一致的 Wikipedia JSONL，每行至少有 `id` 和 `contents`：

```bash
python search_r1/search/index_builder.py \
  --retrieval_method bm25 \
  --corpus_path /path/to/wiki-18.jsonl \
  --save_dir /path/to/index
```

按官方实现，最终索引目录是：

```text
/path/to/index/bm25
```

BM25 依赖 `pyserini` 及其 Java/Lucene 运行环境；正式租机脚本应在下载大文件前检查 `java -version` 和 `python -c "import pyserini"`。

### 3.2 启动在线 API

基于官方最新版 `retrieval_server.py` 的正确形式是：

```bash
python search_r1/search/retrieval_server.py \
  --index_path /path/to/index/bm25 \
  --corpus_path /path/to/wiki-18.jsonl \
  --topk 3 \
  --retriever_name bm25
```

BM25 不应传 `--faiss_gpu`，也不需要 E5 `--retriever_model`。服务器默认监听 `0.0.0.0:8000`，训练端继续调用 `http://127.0.0.1:8000/retrieve`。

### 3.3 当前课程分支的重要兼容性发现

当前 `codex/course-reproduction` 从论文 v1 时代提交派生；其中本地 `search_r1/search/retrieval_server.py` 虽已有 `BM25Retriever` 类，但在线服务器底部仍把 `retrieval_method` 硬编码为 `e5`，CLI 也没有把 `--retriever_name bm25` 接入配置。也就是说，**仅把启动命令改成 BM25 并不能让当前分支真正运行 BM25**。

后续实现必须二选一：

1. 最小移植官方最新版服务器的 `--retriever_name` / `--faiss_gpu` 参数接线；或
2. 在课程分支增加等价的独立 BM25 server wrapper。

无论选择哪种方式，都要先用一个固定 query 验证 `/retrieve` 返回恰好 3 篇有效文档，再启动 RL。这个结论来自官方最新版与当前分支源码的直接对照，不是推测。

主来源：[官方检索服务器](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py)、[官方索引构建器](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/index_builder.py)、[官方检索启动示例](https://github.com/PeterGriffinJin/Search-R1/blob/main/retrieval_launch.sh)。

## 4. 资源版相对论文 v1 的科学差异

| 维度 | 论文 v1 / 原始主路径 | 课程资源版 | 解释限制 |
|---|---|---|---|
| 模型 | Qwen2.5-3B/7B、LLaMA3.2-3B | Qwen2.5-1.5B-Instruct | 模型容量和初始能力改变，不能直接对齐论文 EM。 |
| 默认 RL | 主结果默认 PPO；另有 GRPO 对照 | GRPO | 属于论文支持算法，但优化动态和最终 checkpoint 不同。 |
| 检索器 | Wikipedia 2018 + E5 dense retriever | 同语料 + BM25 sparse retriever | 搜索结果分布改变，训练环境和评测条件均改变。 |
| rollout | GRPO 每 prompt 5 条 | 应保持 5 条，显存不足再作为显式消融修改 | 改 group size 会改变优势估计。 |
| batch | 官方脚本 global train batch 512 | 计划 32，实测后最多 64 | 相同步数不再代表相同 rollout 总量。 |
| 步数 | 官方复现脚本 305 | 计划 100-120 | 只能分析资源预算内学习趋势，不能宣称达到原训练量。 |
| 评测 | 七数据集完整 test/validation | 七数据集固定子集 | 指标方差更大，只能在相同样本清单上比较本项目各方法。 |
| 硬件 | 原始大规模多卡训练 | 1×A800 80GB | 主要改变吞吐和可用 batch，也可能因数值/并行差异影响随机轨迹。 |

最重要的报告原则：不要拿资源版平均 EM 与论文表 2/3 的绝对数值作“复现成功/失败”的直接判据。有效结论应来自同一资源协议下的配对比较，例如：

```text
Pre-RL 1.5B + BM25
vs. Search-R1 GRPO 1.5B + BM25
vs. 改进方法 1.5B + BM25
```

三组必须使用相同模型 revision、语料/索引、训练与评测样本、随机种子和解码设置。

## 5. 为保留 Search-R1 科学内核必须保持的配置

以下项目不是普通工程参数，而是方法身份或公平比较条件：

1. **多轮交互协议**：保留 `<think>`、`<search>`、`<information>`、`<answer>` 标记，以及模型生成查询、环境返回文档、模型继续推理的闭环。
2. **Retrieved-token masking**：保留 `actor_rollout_ref.actor.state_masking=true`。论文表 4 中去掉 masking 后平均 EM 从 0.305 降到 0.147，说明它是核心机制，不是可随意删除的优化。
3. **Outcome reward**：保持最终答案归一化 Exact Match；不能偷偷加入格式奖励或过程奖励后仍称为同一 baseline。
4. **训练数据语义**：保持 NQ + HotpotQA 合并训练；若为了成本抽样，必须固定样本 ID、抽样种子和顺序，并让 baseline 与改进方法共用。
5. **检索语料**：保持论文使用的 Wikipedia 2018 语料版本。BM25 替换的是检索算法，不应同时更换知识库。
6. **Top-k**：保持每次返回 3 篇。论文明确所有 retrieval-based 方法统一为 3 篇。
7. **最大搜索轮数**：保持官方复现脚本的 `max_turns=4`，并在所有对照中一致。该值来自官方脚本，而非论文正文明确陈述。
8. **GRPO group size**：优先保持论文和官方脚本的 `n_agent=5`；若显存迫使修改，必须记录为额外科学差异。
9. **评测协议**：保留七数据集名称和 EM 定义；小规模评测必须生成一次固定 manifest，所有实验逐条复用，不能每次重新随机抽样。
10. **公平预算**：baseline 与改进方法保持相同 batch、optimizer update 数、最大 token 长度、采样温度、seed 和 GPU 型号，并同时报告 wall-clock、GPU-hours 与 rollout 数。

论文 v1 对这些核心事实的主来源是：[方法与 masking](https://arxiv.org/pdf/2503.09516v1)、[训练/评测设置](https://arxiv.org/pdf/2503.09516v1)、[Search-R1 官方代码](https://github.com/PeterGriffinJin/Search-R1)。

## 6. 对课程方案的独立建议

该方案可以作为课程复现主线，但在宣布“可以租机”前，代码至少要通过以下本地静态门禁和云端 2-step 门禁：

- BM25 CLI 确实进入 `BM25Retriever`，而不是因旧代码仍加载 E5。
- retriever 环境包含 Pyserini 和可用 Java，API 固定 query 返回 3 篇有效文档。
- 1.5B 模型能通过当前仓库的 Transformers + vLLM + FSDP 组合加载。
- GRPO 2-step 能完成 5-rollout 分组、EM reward、retrieved-token mask、参数更新和 checkpoint 保存。
- batch 从 8/16 起测；只有 10-step 实测无 OOM/NaN 后，才决定正式 batch 32 或 64。
- Pre-RL、GRPO baseline 和后续改进使用同一固定七数据集子集 manifest。

因此，主来源支持“1×A800 + 1.5B + GRPO + CPU BM25”作为**合理且有研究价值的资源受限适配方向**；主来源不支持在未完成上述兼容改造和 GPU 冒烟测试前声称它一定能直接运行，也不支持把其结果表述为论文 v1 严格数值复现。

## 来源清单

1. Bowen Jin et al., [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning, arXiv v1](https://arxiv.org/pdf/2503.09516v1)。用于方法、PPO/GRPO、masking、训练数据、E5/Wikipedia、Top-3、EM 和七数据集设置。
2. PeterGriffinJin, [Search-R1 官方 GitHub](https://github.com/PeterGriffinJin/Search-R1)。用于框架支持范围和官方操作入口。
3. PeterGriffinJin, [官方 `train_grpo.sh`](https://github.com/PeterGriffinJin/Search-R1/blob/main/train_grpo.sh)。用于 GRPO 参数、Qwen2.5 示例和原始多卡配置。
4. PeterGriffinJin, [官方 `retrieval_server.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py)。用于 BM25 在线 API、CLI 和 CPU LuceneSearcher 实现。
5. PeterGriffinJin, [官方 `index_builder.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/index_builder.py)。用于 BM25 索引目录和 Pyserini 构建流程。
6. PeterGriffinJin, [官方 model registry](https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/models/registry.py)。用于 Qwen2 架构支持判断。
7. Qwen Team, [Qwen2.5-1.5B-Instruct 官方模型页](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)。用于模型规模、架构、加载方式与 Transformers 版本下限。
