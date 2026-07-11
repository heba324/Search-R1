# Search-R1 论文 arXiv v1 严格复现手册

## 1. 本次复现对象

本分支只复现 `Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning` 的 arXiv v1。作者训练核心冻结在 `118c6e7`，这是论文脚本发布当天用于修复七数据集 EM 评分的提交。

主实验固定为：Qwen2.5-3B-Instruct、PPO、NQ+HotpotQA 合并训练集、纯最终答案 EM 奖励、305 steps、最多 4 轮搜索、每轮返回 top-3 文档。不能把 1005 steps、GRPO 或格式奖励结果写成该主实验。

论文 v1 表 2 的目标是：NQ 0.323、TriviaQA 0.537、PopQA 0.364、HotpotQA 0.308、2WikiMultiHopQA 0.336、Musique 0.105、Bamboogle 0.315，平均 EM 为 0.327。

## 2. 租用配置

为最大程度接近作者环境，正式运行选择单机 8×H100 80GB、至少 128 GiB 内存、至少 500 GiB 可用磁盘，Ubuntu 20.04/22.04，带 Conda、CUDA 12.1 开发工具的按小时镜像。不要选择分散在不同机器上的 8 张卡。

正式租机前只需保证本分支已推送，不需要在本地下载模型和数据。云实例创建后先执行预检；预检失败就保存输出并关闭实例。

## 3. 下载冻结分支

```bash
git clone --branch codex/paper-v1-reproduction https://github.com/heba324/Search-R1.git
cd Search-R1
python3 scripts/paper_v1/preflight.py
```

预检应显示作者核心为 `118c6e7361bb68e33c525b50d62f83b63462799e`，且核心目录没有改动。

## 4. 安装两个 Conda 环境

```bash
bash scripts/paper_v1/setup_envs.sh 2>&1 | tee setup-paper-v1.log
```

训练环境固定名为 `Search-R1`，检索环境固定名为 `Search-R1-retriever`。安装结束必须看到 PyTorch CUDA 可用和 FAISS GPU 可用。

## 5. 准备数据与完整检索资源

```bash
bash scripts/paper_v1/prepare_train_data.sh 2>&1 | tee prepare-train-data.log
bash scripts/paper_v1/prepare_models.sh 2>&1 | tee prepare-models.log
bash scripts/paper_v1/prepare_retrieval_assets.sh 2>&1 | tee prepare-retrieval.log
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
python scripts/paper_v1/prepare_eval_data.py 2>&1 | tee prepare-eval-data.log
python3 scripts/paper_v1/preflight.py --require-assets
```

训练 parquet 固定到作者 2025-03-13 的 Hugging Face 修订并进行 SHA-256 校验。Qwen2.5-3B-Instruct 和 E5 模型也固定到论文发布前已有的修订。评测数据固定到论文发布前的 FlashRAG 修订，包含 NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、Musique 和 Bamboogle。

## 6. 启动检索服务

```bash
tmux new -s retriever
cd ~/Search-R1
bash scripts/paper_v1/launch_retriever.sh 2>&1 | tee retriever-paper-v1.log
```

看到 Uvicorn 正常监听 8000 端口后，按 `Ctrl+B`，松开，再按 `D` 返回普通终端。保持这个 tmux 会话运行。

## 7. 运行论文 PPO 主实验

先登录 WandB 以保存作者同类训练曲线：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
wandb login
tmux new -s train
cd ~/Search-R1
bash scripts/paper_v1/train_qwen25_3b_instruct_ppo.sh
```

训练必须完整运行到 305 steps，并生成 `artifacts/paper-v1/training_completed.txt`。只看到进程启动、两步 smoke 或 checkpoint 都不能称为论文复现成功。

## 8. 七数据集评测

作者脚本每 100 步保存一次，因此 305-step 训练最后可用的 checkpoint 是 `actor/global_step_300`。评测脚本默认使用该路径：

```bash
bash scripts/paper_v1/evaluate_qwen25_3b_instruct_ppo.sh
```

只有在手动评测其他 checkpoint 时才设置 `MODEL_PATH=/绝对路径/到/actor/global_step_xxx`。

只有日志同时出现七个 `val/test_score/<dataset>`，并生成 `evaluation_completed.txt`，才完成论文结果评测。逐项记录复现 EM、论文 EM、绝对差值和相对差值，随机强化学习不应承诺每个小数位逐位相同。

## 9. 保存证据并停止计费

无论成功或失败都执行：

```bash
bash scripts/paper_v1/collect_evidence.sh
```

把生成的 `artifacts/paper-v1/evidence-*.tar.gz` 下载到本地，确认文件完整后，回到云平台控制台停止或销毁实例以停止计费。仅关闭 SSH 或 tmux 不会停止计费。
