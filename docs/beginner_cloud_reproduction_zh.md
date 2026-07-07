# Search-R1 新手云端复现操作手册

这份手册的目标是：尽量少花钱，把 Search-R1 官方训练复现流程在云 GPU 上先跑通一个 2-step smoke test，再决定是否继续完整训练。

如果你是第一次租 GPU 服务器，照着下面一步一步做即可。不要一开始就跑完整训练。

## 0. 你现在本地已经准备好了什么

本地路径：

```text
D:\python_code\Search-R1
```

里面已经新增了一套云端辅助脚本：

```text
scripts/cloud_setup_searchr1.sh          训练环境安装
scripts/cloud_setup_retriever.sh         检索环境安装
scripts/cloud_prepare_data_and_index.sh  下载数据和索引
scripts/cloud_launch_retriever.sh        启动检索服务
scripts/cloud_check_retriever.py         检查检索 API
scripts/cloud_train_grpo_smoke.sh        2-step 省钱验证训练
scripts/cloud_train_grpo_full.sh         完整训练
docs/rental_reproduction_runbook.md      英文简版手册
docs/beginner_cloud_reproduction_zh.md   本文档
```

## 1. 先理解整体流程

Search-R1 官方完整训练不是一个普通 Python 脚本，它分成两条线：

```text
检索服务线：
下载 Wikipedia 语料和 e5 检索索引
启动本地检索 API：http://127.0.0.1:8000/retrieve

训练线：
加载 Qwen2.5-3B
用 Ray + FSDP + vLLM 跑 GRPO 强化学习
模型生成 <search> query </search>
程序调用检索 API
把结果塞回 <information>...</information>
模型继续生成 <answer>...</answer>
```

所以云服务器上至少要开两个终端：

```text
终端 A：一直运行检索服务
终端 B：运行 smoke test 或完整训练
```

## 2. 租服务器前先准备

### 2.1 推荐配置

优先租：

```text
系统：Ubuntu 20.04 或 Ubuntu 22.04
GPU：8 × A100 40GB 或 8 × A100 80GB
内存：128GB 以上
硬盘：500GB 以上，最好 1TB
镜像：PyTorch / CUDA / Ubuntu 镜像
计费方式：按小时
```

如果平台有现成镜像，优先选类似：

```text
PyTorch 2.x + CUDA 12.x + Ubuntu 22.04
```

不要租：

```text
Windows 云服务器
单卡 4GB / 8GB / 12GB GPU
只有 50GB 硬盘的机器
按月包租
```

### 2.2 最省钱原则

先只做 smoke test：

```text
只训练 2 step
只验证环境、数据、检索、vLLM、Ray、FSDP 能跑通
```

smoke test 成功以后，才考虑完整训练。

smoke test 失败以后：

```text
保存日志
立刻关机
把日志发给我排查
不要继续烧钱
```

## 3. 把本地代码传到云服务器

你有两种方式。新手推荐方式 A。

## 方式 A：上传到你自己的 GitHub，再在服务器 git clone

### 3.1 本地创建自己的 GitHub 仓库

在 GitHub 上新建一个仓库，例如：

```text
Search-R1-reproduce
```

可以设为 private。

### 3.2 本地提交这些文件

在 PowerShell 进入本地仓库：

```powershell
cd D:\python_code\Search-R1
git status
```

你应该能看到新增文件。

如果你要把当前仓库推到自己的 GitHub，先设置你自己的远程地址。把下面的地址换成你的：

```powershell
git remote remove origin
git remote add origin https://github.com/你的用户名/Search-R1-reproduce.git
```

提交：

```powershell
git add docs scripts
git commit -m "add cloud reproduction scripts"
git push -u origin main
```

如果 `git push` 报错，不要慌，把完整报错发给我。

## 方式 B：直接压缩上传

如果你不会 GitHub，可以把整个文件夹压缩成 zip，再用平台网页上传，或者用 `scp`。

例如本地 PowerShell：

```powershell
cd D:\python_code
Compress-Archive -Path .\Search-R1 -DestinationPath .\Search-R1.zip -Force
```

然后上传到服务器再解压：

```bash
unzip Search-R1.zip
cd Search-R1
```

但方式 B 在大文件多的时候不如 GitHub 方便。

## 4. 租到服务器以后怎么连接

平台通常会给你：

```text
IP 地址
SSH 端口
用户名，常见是 root
密码或私钥
```

在你本地 PowerShell 连接：

```powershell
ssh root@服务器IP
```

如果端口不是 22，比如是 12345：

```powershell
ssh -p 12345 root@服务器IP
```

连上以后你会看到类似：

```text
root@xxx:~#
```

这说明你已经进入云服务器。

## 5. 服务器上第一件事：检查 GPU 和硬盘

在服务器里执行：

```bash
nvidia-smi
```

你应该看到 8 张 A100。如果不是 8 张，先不要继续。

再看硬盘：

```bash
df -h
```

确保可用空间至少 500GB。低于 300GB 不建议继续完整复现。

检查 conda：

```bash
conda --version
```

如果提示 `conda: command not found`，说明镜像没有 conda。先停下，把情况发给我。

## 6. 下载代码到服务器

如果你用了 GitHub 方式：

```bash
git clone https://github.com/你的用户名/Search-R1-reproduce.git
cd Search-R1-reproduce
```

如果你直接克隆官方仓库，则没有我们新增的云端脚本，不推荐。

确认脚本存在：

```bash
ls scripts/cloud_*
ls docs/*reproduction*
```

## 7. 安装训练环境

在仓库根目录执行：

```bash
bash scripts/cloud_setup_searchr1.sh
```

它会创建 conda 环境：

```text
searchr1
```

这个过程可能需要 20-60 分钟，取决于网络。

成功时你会看到类似：

```text
torch: 2.4.0+cu121
cuda available: True
gpu count: 8
Search-R1 training environment is ready: searchr1
```

如果 `flash-attn` 安装失败，保存报错，不要继续完整训练。

## 8. 安装检索环境

继续执行：

```bash
bash scripts/cloud_setup_retriever.sh
```

它会创建 conda 环境：

```text
retriever
```

成功时你会看到类似：

```text
cuda available: True
gpu count: 8
faiss gpu resources: True
Retriever environment is ready: retriever
```

## 9. 下载数据和索引

执行：

```bash
bash scripts/cloud_prepare_data_and_index.sh
```

它会下载：

```text
wiki-18 corpus
e5 dense index
NQ 数据集
```

成功时应该看到：

```text
Index: /.../data/wiki18/e5_Flat.index
Corpus: /.../data/wiki18/wiki-18.jsonl
Train data: /.../data/nq_search/train.parquet
Test data: /.../data/nq_search/test.parquet
```

这个步骤可能很慢，因为文件较大。下载失败通常是网络问题。

## 10. 启动检索服务

推荐用 `tmux`，这样你关掉 SSH 窗口服务也不会立刻死。

先开一个 tmux：

```bash
tmux new -s retriever
```

进入 tmux 后执行：

```bash
bash scripts/cloud_launch_retriever.sh
```

看到类似下面内容说明服务在启动：

```text
Uvicorn running on http://0.0.0.0:8000
```

这个终端不要关闭。

从 tmux 临时退出但保持服务运行：

```text
按 Ctrl+B
松开
再按 D
```

回到普通终端后，可以重新进入：

```bash
tmux attach -t retriever
```

## 11. 检查检索 API 是否可用

新开一个 tmux 或普通 SSH 终端：

```bash
tmux new -s train
```

进入仓库目录：

```bash
cd Search-R1-reproduce
```

如果你的目录名不同，用你实际的目录名。

激活训练环境：

```bash
conda activate searchr1
```

检查检索服务：

```bash
python scripts/cloud_check_retriever.py
```

成功时会看到：

```text
retriever ok
1. score=...
2. score=...
3. score=...
```

如果这里失败，不要跑训练。先检查检索服务 tmux 有没有启动。

## 12. 跑最省钱 smoke test

确认检索服务可用后，执行：

```bash
bash scripts/cloud_train_grpo_smoke.sh
```

这个脚本只跑：

```text
2 training steps
16 条训练样本
8 条验证样本
console logger
不保存大 checkpoint
```

它的作用不是复现论文指标，而是验证：

```text
模型能下载
Qwen2.5-3B 能加载
Ray 能启动
FSDP 能启动
vLLM 能 rollout
检索 API 能被调用
reward 逻辑能执行
训练循环能跑起来
```

成功后会生成日志：

```text
nq-search-r1-grpo-qwen2.5-3b-smoke.log
```

如果它跑完 2 step，说明环境基本成功。

## 13. smoke test 失败怎么办

立刻做三件事：

```bash
nvidia-smi > nvidia-smi.txt
conda list -n searchr1 > conda-searchr1.txt
conda list -n retriever > conda-retriever.txt
```

把日志文件也留下：

```bash
ls *.log
```

然后打包：

```bash
tar -czf debug_logs.tar.gz *.log nvidia-smi.txt conda-searchr1.txt conda-retriever.txt
```

下载到本地，或者把报错粘给我。

最重要：失败后不要继续完整训练。

## 14. smoke test 成功后，是否跑完整训练

如果你只是课程作业，不一定需要完整训练。

你可以在报告里写：

```text
已在 8×A100 云服务器上完成 Search-R1 官方训练链路 smoke test，
验证了数据处理、检索服务、Qwen2.5-3B 加载、Ray/FSDP/vLLM、
GRPO 训练入口、搜索调用和 reward 计算均可运行。
由于完整 1005-step 训练成本较高，本文仅做短步数复现验证。
```

如果老师要求必须尽量完整复现，再跑完整训练。

## 15. 跑完整训练

先登录 WandB：

```bash
conda activate searchr1
wandb login
```

然后：

```bash
bash scripts/cloud_train_grpo_full.sh
```

完整训练默认是：

```text
Qwen/Qwen2.5-3B
GRPO
NQ search
8 GPU
1005 training steps
```

完整训练会更贵，可能跑很久。中途不要随便关机，否则钱花了但结果不完整。

## 16. 需要保存哪些证据

为了写实验报告，至少保存：

```text
1. nvidia-smi 输出
2. conda 环境版本
3. 数据下载成功截图或日志
4. 检索 API 成功输出
5. smoke test 日志
6. 如果跑完整训练，保存完整训练日志
7. 如果用了 WandB，保存 WandB 链接和曲线截图
8. checkpoint 目录截图
```

常用命令：

```bash
nvidia-smi
conda list -n searchr1
conda list -n retriever
ls data/wiki18
ls data/nq_search
ls verl_checkpoints
```

## 17. 最后一定要关机

训练结束或失败后，先保存日志，然后关机：

```bash
sudo shutdown now
```

还要去云平台网页控制台确认：

```text
实例已停止
或实例已销毁
没有继续计费
```

只断开 SSH 不等于停止计费。

## 18. 新手常见问题

### Q1：我看到命令卡住了，是不是坏了？

不一定。安装 `flash-attn`、下载 Hugging Face 文件、加载检索索引都可能很慢。

但如果 20 分钟没有任何输出，可以把当前输出发给我判断。

### Q2：为什么要两个 conda 环境？

因为训练环境和检索环境依赖不完全一样。

官方 README 也建议分开：

```text
searchr1：训练 Qwen/vLLM/Ray/FSDP
retriever：FAISS/pyserini/检索服务
```

### Q3：我能不能用 1 张 4090？

不建议做官方完整训练。可能可以做更小的调参实验，但不算严格复现官方脚本。

### Q4：为什么先跑 smoke test？

因为完整训练很贵。smoke test 可以在很少步数内证明环境是否能跑通。

### Q5：smoke test 成功是否等于论文复现成功？

不等于。smoke test 只是证明训练链路可运行。

论文级复现需要完整训练步数、评测指标和结果对比。

## 19. 你可以直接复制的最短命令清单

假设你已经 SSH 到服务器，并且代码仓库已经 clone 好：

```bash
cd Search-R1-reproduce

bash scripts/cloud_setup_searchr1.sh
bash scripts/cloud_setup_retriever.sh
bash scripts/cloud_prepare_data_and_index.sh

tmux new -s retriever
bash scripts/cloud_launch_retriever.sh
```

按 `Ctrl+B`，再按 `D` 退出 tmux。

然后：

```bash
tmux new -s train
cd Search-R1-reproduce
conda activate searchr1
python scripts/cloud_check_retriever.py
bash scripts/cloud_train_grpo_smoke.sh
```

smoke 成功后，如果决定完整训练：

```bash
wandb login
bash scripts/cloud_train_grpo_full.sh
```

结束后：

```bash
sudo shutdown now
```

