# Search-R1 新手云端复现操作手册

这份手册用于先把代码、环境检查和省钱门禁准备好，再租 GPU。请严格区分下面三种状态：

1. 本地测试通过：只说明辅助脚本的逻辑和语法通过检查。
2. 单卡 smoke 通过：说明 Search-R1 的环境、模型、检索调用和 2-step GRPO 训练链路可以启动。
3. 八卡 full 完成：有完整训练日志、checkpoint、评测指标和环境证据后，才能讨论论文复现结果。

**smoke test 不等于论文复现成功。当前阶段也不能声称已经复现成功。**

## 1. 已准备好的代码

本地仓库：

```text
D:\python_code\Search-R1
```

GitHub 仓库：

```text
https://github.com/heba324/Search-R1
```

官方上游代码来自：

```text
https://github.com/PeterGriffinJin/Search-R1
```

准备工作以官方提交 `598e61b` 为基线。官方训练、奖励、检索和数据处理核心代码保持不变，租机辅助逻辑放在 `scripts/cloud_*` 文件中。

主要脚本：

```text
scripts/cloud_preflight.py                 租机配置检查
scripts/cloud_setup_searchr1.sh            安装训练环境
scripts/cloud_setup_retriever.sh           安装检索环境
scripts/cloud_prepare_smoke_assets.sh      构建单卡微型检索索引
scripts/cloud_prepare_data_and_index.sh    下载完整 Wikipedia 语料和索引
scripts/cloud_launch_retriever.sh          启动 smoke 或 full 检索服务
scripts/cloud_check_retriever.py           检查检索 API 和响应结构
scripts/cloud_train_grpo_smoke.sh          运行 2-step GRPO
scripts/cloud_train_grpo_full.sh           运行 8 卡完整训练
scripts/cloud_collect_evidence.sh          收集报告证据
```

## 2. 为什么采用两阶段租用

第一次租机只使用仓库自带的 `example/corpus.jsonl` 构建很小的 E5 索引，不下载完整 Wikipedia。这样可以先验证：

```text
Conda 和 CUDA 环境
PyTorch、vLLM、Ray、FSDP 和 flash-attn
Qwen2.5-3B 模型加载
E5 + FAISS 检索服务
<search> -> <information> -> <answer> 调用链
GRPO reward 和 2-step 训练入口
```

微型索引不用于评测准确率。只有单卡 smoke 通过后，才租八卡机器下载完整索引并跑正式训练。

## 3. 租什么配置

### 阶段 A：省钱 smoke

推荐配置：

```text
系统：Ubuntu 22.04（Ubuntu 20.04 也可以）
GPU：1 x A100 80GB
内存：64GB 以上，推荐 96GB 或 128GB
硬盘：至少 150GB，推荐 200GB
镜像：带 Conda 的 PyTorch/CUDA 镜像
计费：按小时
```

`cloud_preflight.py --profile smoke` 的硬门槛是 1 张至少 75GiB 显存的 GPU、64GiB 内存和 100GiB 可用磁盘。页面上选择磁盘时应留安装和模型缓存余量，因此建议 200GB。

### 阶段 B：完整训练

推荐配置：

```text
系统：Ubuntu 20.04 或 Ubuntu 22.04
GPU：8 x A100 40GB，优先 8 x A100 80GB
内存：至少 128GB，推荐 256GB
硬盘：至少 500GB，推荐 1TB
镜像：带 Conda 的 PyTorch/CUDA 镜像
计费：按小时
```

不要选择 Windows 云主机、消费级小显存单卡、50GB 系统盘或按月计费实例。

## 4. 租机后连接服务器

平台会提供 IP、SSH 端口、用户名和密码或私钥。在本地 PowerShell 执行：

```powershell
ssh root@服务器IP
```

SSH 端口不是 22 时，例如 12345：

```powershell
ssh -p 12345 root@服务器IP
```

连接后先确认基础命令：

```bash
nvidia-smi
df -h
free -h
conda --version
git --version
```

如果缺少 `git` 或 `tmux`：

```bash
apt-get update
apt-get install -y git tmux
```

如果缺少 Conda，最省事的做法是关闭实例，重新选择带 Anaconda/Miniconda 的镜像。

## 5. 下载我们准备好的仓库

```bash
git clone https://github.com/heba324/Search-R1.git
cd Search-R1
git rev-parse HEAD
```

确认辅助脚本存在：

```bash
ls scripts/cloud_*
```

如果 clone 后看不到这些脚本，先不要安装环境，检查是否拉取了正确仓库和分支。

## 6. 阶段 A：运行单卡 smoke

### 6.1 租机预检

```bash
python3 scripts/cloud_preflight.py --profile smoke
```

最后必须看到：

```text
Preflight passed for profile: smoke
```

如果出现 `Preflight failed`，不要继续安装。保存完整输出并关闭实例。

### 6.2 安装两个 Conda 环境

训练环境名称严格为：

```text
Search-R1
```

检索环境名称严格为：

```text
Search-R1-retriever
```

安装命令：

```bash
bash scripts/cloud_setup_searchr1.sh 2>&1 | tee setup-Search-R1.log
bash scripts/cloud_setup_retriever.sh 2>&1 | tee setup-retriever.log
```

这一步可能需要 20 到 60 分钟。`flash-attn` 编译期间一段时间没有新输出是正常现象。如果安装报错，不要继续训练。

手动查看环境时使用：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

conda activate Search-R1-retriever
python -c "import faiss; print(hasattr(faiss, 'StandardGpuResources'))"
```

### 6.3 准备微型检索索引和 NQ 数据

```bash
bash scripts/cloud_prepare_smoke_assets.sh 2>&1 | tee prepare-smoke.log
```

它会：

```text
下载并处理 NQ 数据集
下载 intfloat/e5-base-v2
读取 example/corpus.jsonl
生成 data/smoke_retriever/e5_Flat.index
```

它不会下载完整 Wikipedia 索引。

### 6.4 启动 smoke 检索服务

```bash
tmux new -s retriever
cd ~/Search-R1
bash scripts/cloud_launch_retriever.sh 2>&1 | tee retriever-smoke.log
```

默认 `ASSET_PROFILE=smoke`。服务固定监听：

```text
http://127.0.0.1:8000/retrieve
```

看到 Uvicorn 启动信息后，按 `Ctrl+B`，松开，再按 `D`，退出 tmux 但保持服务运行。

查看服务日志：

```bash
tmux attach -t retriever
```

### 6.5 检查检索 API

在普通 SSH 终端执行：

```bash
cd ~/Search-R1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
python scripts/cloud_check_retriever.py
```

应看到 `retriever ok` 和 3 条文档。如果失败，不要运行训练。

### 6.6 运行 2-step GRPO

```bash
tmux new -s train
cd ~/Search-R1
bash scripts/cloud_train_grpo_smoke.sh
```

脚本会再次检查单卡配置和检索 API，然后运行：

```text
模型：Qwen/Qwen2.5-3B
算法：GRPO
训练样本：8 条
训练 batch：4
rollout 数量：2
训练步数：2
日志：console
checkpoint：不保存
```

训练日志保存在：

```text
nq-search-r1-grpo-qwen2.5-3b-smoke.log
```

只有看到训练正常结束到第 2 step，才能说“smoke 训练链路通过”。这仍然不是论文指标复现。

### 6.7 收集 smoke 证据

不论成功或失败，都执行：

```bash
cd ~/Search-R1
bash scripts/cloud_collect_evidence.sh smoke
```

输出目录类似：

```text
artifacts/reproduction-smoke-20260710T120000Z/
```

里面包含 Git 提交、GPU、内存、磁盘、Conda 依赖、数据哈希、日志和检查点清单，不包含 API 密钥。

将证据包下载到本地后，立刻去云平台控制台停止或销毁单卡实例。只关闭 SSH 不会停止计费。

## 7. 阶段 B：运行八卡完整复现

阶段 A 失败时不要进入阶段 B。阶段 A 通过后，再创建一台满足 full 配置的实例，从第 4 节重新连接并 clone 仓库。

### 7.1 八卡预检

```bash
cd ~/Search-R1
python3 scripts/cloud_preflight.py --profile full
```

最后必须看到：

```text
Preflight passed for profile: full
```

### 7.2 安装环境

新实例需要重新安装：

```bash
bash scripts/cloud_setup_searchr1.sh 2>&1 | tee setup-Search-R1.log
bash scripts/cloud_setup_retriever.sh 2>&1 | tee setup-retriever.log
```

### 7.3 下载完整数据和索引

```bash
bash scripts/cloud_prepare_data_and_index.sh 2>&1 | tee prepare-full.log
```

脚本会下载 `part_aa`、`part_ab` 和压缩 Wikipedia 语料，验证两个索引分片大小，原子拼接 `e5_Flat.index`，再原子解压语料。成功输出包括：

```text
data/wiki18/e5_Flat.index
data/wiki18/wiki-18.jsonl
data/nq_search/train.parquet
data/nq_search/test.parquet
```

下载中断后可以重新执行同一命令，Hugging Face 缓存会继续使用已下载内容。

### 7.4 启动 full 检索服务

如果已有 smoke 检索 tmux，先停止：

```bash
tmux kill-session -t retriever
```

然后启动完整 Wikipedia 检索：

```bash
tmux new -s retriever
cd ~/Search-R1
ASSET_PROFILE=full bash scripts/cloud_launch_retriever.sh 2>&1 | tee retriever-full.log
```

按 `Ctrl+B`，再按 `D` 离开 tmux。检查：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
python scripts/cloud_check_retriever.py
head -n 3 artifacts/retriever_profile.txt
```

`retriever_profile.txt` 第一行必须是 `full`。

### 7.5 运行完整 GRPO

默认使用控制台日志，不要求 WandB：

```bash
tmux new -s train
cd ~/Search-R1
CONFIRM_FULL_RUN=YES bash scripts/cloud_train_grpo_full.sh
```

完整脚本会再次检查八卡配置、full 检索标记、NQ parquet 和检索 API。默认参数为：

```text
Qwen/Qwen2.5-3B base
GRPO
8 GPU
NQ train/test
Wikipedia 2018 + E5 dense retriever
1005 training steps
每 100 step 保存 checkpoint
每 50 step 验证
```

如需 WandB，先登录，再显式启用：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
wandb login
CONFIRM_FULL_RUN=YES TRAINER_LOGGER=wandb bash scripts/cloud_train_grpo_full.sh
```

没有写 `CONFIRM_FULL_RUN=YES` 时，脚本会拒绝启动昂贵训练，这是正常的费用保护。

### 7.6 收集 full 证据

```bash
cd ~/Search-R1
bash scripts/cloud_collect_evidence.sh full
```

至少保存：

```text
完整训练日志
nvidia-smi.txt
两个 Conda explicit 依赖文件
data-sha256.txt
retriever_profile.txt
checkpoint-files.txt
verl_checkpoints/ 中的 checkpoint
评测指标和 WandB 曲线（如果启用）
```

保存完毕后在云平台控制台停止或销毁实例。

## 8. 从服务器下载证据

先在服务器打包：

```bash
cd ~/Search-R1
tar -czf search-r1-evidence.tar.gz artifacts *.log
```

在本地 PowerShell 下载，端口为 22 时：

```powershell
scp root@服务器IP:~/Search-R1/search-r1-evidence.tar.gz .
```

如果需要保存 checkpoint，单独打包并下载；checkpoint 可能很大，不要在确认下载完成前销毁磁盘。

## 9. 失败时怎么做

### 环境安装失败

保存：

```text
setup-Search-R1.log
setup-retriever.log
nvidia-smi 输出
```

不要继续准备数据或训练。

### 数据下载失败

先看磁盘：

```bash
df -h
du -sh data/*
```

磁盘足够时可以重新运行准备命令。不要手工删除 Hugging Face 缓存，除非已经确认缓存损坏。

### 检索服务失败

```bash
tmux attach -t retriever
cat artifacts/retriever_profile.txt
nvidia-smi
```

把完整错误和 profile 文件一起保存。

### 训练 OOM

立即保存日志并停止训练。不要直接修改 full 参数后仍称为官方配置复现。任何 batch、长度、GPU 数量或显存利用率变化都必须记录在实验报告中。

## 10. 报告中可以怎样表述

只完成本地检查时：

```text
已完成 Search-R1 官方代码与云端复现脚本的静态检查和自动化测试，
尚未在 Linux GPU 环境执行训练，因此不构成实验复现结果。
```

完成单卡 smoke 时：

```text
在单张 A100 80GB 上，使用微型 E5 检索索引完成了 Search-R1 2-step GRPO
训练链路验证。该实验验证了系统可运行性，不用于复现论文准确率。
```

完成八卡 full 后，报告必须给出硬件、Git 提交、依赖、训练步数、日志、checkpoint、评测方法、指标和论文结果对比，不能只写“运行成功”。

## 11. 最短命令清单

单卡 smoke：

```bash
git clone https://github.com/heba324/Search-R1.git
cd Search-R1
python3 scripts/cloud_preflight.py --profile smoke
bash scripts/cloud_setup_searchr1.sh
bash scripts/cloud_setup_retriever.sh
bash scripts/cloud_prepare_smoke_assets.sh

tmux new -s retriever
bash scripts/cloud_launch_retriever.sh
# Ctrl+B，然后 D

bash scripts/cloud_train_grpo_smoke.sh
bash scripts/cloud_collect_evidence.sh smoke
```

八卡 full：

```bash
git clone https://github.com/heba324/Search-R1.git
cd Search-R1
python3 scripts/cloud_preflight.py --profile full
bash scripts/cloud_setup_searchr1.sh
bash scripts/cloud_setup_retriever.sh
bash scripts/cloud_prepare_data_and_index.sh

tmux new -s retriever
ASSET_PROFILE=full bash scripts/cloud_launch_retriever.sh
# Ctrl+B，然后 D

CONFIRM_FULL_RUN=YES bash scripts/cloud_train_grpo_full.sh
bash scripts/cloud_collect_evidence.sh full
```

每次失败或结束后，都要回到云平台网页确认实例已经停止计费。
