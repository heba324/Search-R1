# Search-R1 课程级资源受限复现手册

## 1. 实验定位

本分支完成的是 **Search-R1 核心方法复现**，不是论文表格的严格数值复现。它保留多轮搜索、Top-3检索、结果掩码、规则EM奖励和强化学习训练闭环，但主动改变模型规模、强化学习算法、检索器、批量和训练规模，以适应课程预算。

固定配置：

| 项目 | 配置 |
| --- | --- |
| GPU | 1×A800 80GB |
| 模型 | Qwen2.5-1.5B-Instruct |
| 算法 | GRPO |
| 检索器 | CPU BM25（Pyserini/Lucene） |
| 训练数据 | NQ + HotpotQA官方合并数据 |
| 默认batch | 32 |
| GRPO group size | 5 |
| 正式训练 | 120 steps |
| 最大搜索轮次 | 4 |
| 每轮检索 | Top-3 |
| 奖励 | 最终答案EM |
| 评测 | 七个数据集各固定100题，seed=42 |

论文使用Qwen2.5-3B-Instruct、PPO、E5 Flat和更大批量，因此不能将本实验结果写成复现论文平均EM 0.327。正确表述是“在单卡资源约束下完成Search-R1方法复现并分析其训练行为”。

## 2. 租机要求

- 推荐实例：AutoDL **943机**，1×A800 80GB，数据盘额外扩容455GB并拉满，标称总数据盘约505GB，按量计费约5.98元/小时（以创建页面实时价格为准）
- Ubuntu 20.04或22.04
- 1×NVIDIA A800 80GB，不能是MIG切分卡
- Linux实际可用内存至少110GiB
- 仓库所在数据盘实际空闲空间至少420GiB；943机扩容后通常约460～470GiB可用
- CUDA 12.1开发镜像，包含Conda、Git、GCC/G++和`nvcc`
- `/dev/shm`至少32GB，容器优先使用`--ipc=host`

## 3. 克隆正确分支

```bash
cd /root/autodl-tmp
git clone --branch codex/course-reproduction https://github.com/heba324/Search-R1.git
cd Search-R1
git rev-parse HEAD
python3 scripts/course_reproduction/preflight.py
```

必须在`/root/autodl-tmp`数据盘执行克隆；脚本会把Pip、Hugging Face和Arrow缓存放到仓库的`data/`下。不要在只有几十GB的系统盘克隆，不要克隆默认`main`，也不要使用严格论文分支启动资源版实验。

943机开机后先不要安装，先检查：

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -h
df -h /root/autodl-tmp
df -h /dev/shm
nvcc --version
```

应看到A800 80GB、至少110GiB内存、`/root/autodl-tmp`至少420GiB可用空间、`/dev/shm`至少32GiB且`nvcc`正常。若数据盘仍只有约50GB，立即关机并回控制台完成455GB扩容，不要下载任何资源。

## 4. 安装与下载

```bash
bash scripts/course_reproduction/setup_envs.sh 2>&1 | tee setup-course.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1

bash scripts/course_reproduction/prepare_train_data.sh 2>&1 | tee prepare-train.log
bash scripts/course_reproduction/prepare_model.sh 2>&1 | tee prepare-model.log
bash scripts/course_reproduction/prepare_bm25_index.sh 2>&1 | tee prepare-bm25.log
python scripts/course_reproduction/prepare_eval_data.py 2>&1 | tee prepare-eval.log

python3 scripts/course_reproduction/preflight.py --require-assets
```

BM25索引约2.3GB，官方Wikipedia语料压缩包约5.12GB；首次启动时`datasets`还会生成更大的Arrow缓存，因此磁盘必须留足。检索完全在CPU运行，不占用A800显存。模型、索引与语料均锁定Hugging Face revision。

## 5. 启动检索服务

```bash
tmux new -s retriever
bash scripts/course_reproduction/launch_bm25_retriever.sh 2>&1 | tee bm25-retriever.log
```

看到Uvicorn监听8000端口后，按`Ctrl+B`再按`D`。随后检查：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
python scripts/course_reproduction/check_retriever.py
```

检查器必须确认返回3篇有效文档。

## 6. Pre-RL Baseline

先评测未经强化学习的原始模型：

```bash
tmux new -s pre-eval
bash scripts/course_reproduction/evaluate_pre_rl.sh
```

结果写入：

```text
artifacts/course-reproduction/evaluation/pre-rl/evaluation_completed.json
```

这组结果在报告中命名为 **Pre-RL Baseline**。

## 7. 冒烟测试

```bash
wandb login
tmux new -s smoke
bash scripts/course_reproduction/run_smoke.sh
```

必须确认2步都完成，检索返回Top-3，奖励能计算，没有OOM/NaN，并产生：

```text
artifacts/course-reproduction/course-smoke/training_completed.txt
```

## 8. 十步测速

```bash
tmux new -s timing
bash scripts/course_reproduction/run_timing.sh
```

查看标记中的`elapsed_seconds`：

```bash
cat artifacts/course-reproduction/course-timing/training_completed.txt
```

`pre-rl/evaluation_completed.json`会记录一次完整七数据集评测的`elapsed_seconds`。完整项目的保守估价公式为：

```text
预计GPU秒数 = 十步elapsed_seconds ÷ 10 × 120
              + 6 × 单次七数据集评测秒数
              + 冒烟与十步测速实际秒数
预计总费用 = 预计GPU秒数 ÷ 3600 × 实例每小时价格
```

这里的6次完整评测包括Pre-RL一次、正式训练内部四次和Post-RL一次；环境安装与下载时间另加。不要只用十步耗时线性外推。

若十步出现OOM，先将正式实验的`TRAIN_BATCH_SIZE`和`PPO_MINI_BATCH_SIZE`同时改为16，不要自行修改模型、搜索轮次或group size。

## 9. 正式训练

```bash
tmux new -s train
bash scripts/course_reproduction/train_grpo.sh
```

正式实验保存40、80、120步checkpoint，在0、50、100和完成120次参数更新后进行验证。上游训练器先将控制器计数加一再记录最终验证，因此最后一个验证点显示为 **W&B step 121**；这不代表训练了121次，最终模型仍是`global_step_120`。

```text
artifacts/course-reproduction/search-r1-course-qwen2.5-1.5b-grpo-bm25/training_completed.txt
```

W&B项目名为`Search-R1-course`。重点保存奖励、验证EM、响应长度和`state_tokens/coverage`曲线。

## 10. Post-RL评测与证据

```bash
bash scripts/course_reproduction/evaluate.sh
bash scripts/course_reproduction/collect_evidence.sh
```

Post-RL结果位于：

```text
artifacts/course-reproduction/evaluation/post-rl/evaluation_completed.json
```

报告至少比较Pre-RL与Post-RL七项EM及平均EM，并报告GPU小时、训练用时和资源限制。证据包生成后下载到本地，再关机释放实例。

## 11. 成功标准

本轮复现成功必须同时满足：

1. BM25服务稳定返回Top-3。
2. 2步冒烟测试、10步测速、120步正式训练全部结束。
3. W&B奖励和验证曲线完整，没有NaN。
4. Pre-RL和Post-RL使用同一份固定七数据集子集并都生成七项EM。
5. Post-RL相比Pre-RL出现可解释变化；即使没有提升，也必须保留日志并分析小模型、BM25和训练规模限制。
