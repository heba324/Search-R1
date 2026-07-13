# Search-R1 CEGR 改进实验手册

## 1. 为什么新建改进版

原课程复现分支 `codex/course-reproduction` 保留为不可变 baseline；改进代码位于 `codex/search-r1-improvement`。这样可以明确区分：

```text
baseline：Search-R1 原始 EM 奖励
improvement：仅把训练奖励改为 CEGR
```

两者共用模型、训练数据、BM25、超参数、120 步 checkpoint 和固定七数据集评测协议。不要在原 baseline 分支直接覆盖训练脚本或 checkpoint。

## 2. 改进内容

CEGR 奖励包含：

```text
严格答案 EM
+ token-F1 部分正确反馈
+ 检索证据覆盖 gold alias
- 空查询、重复查询和过量查询惩罚
```

权重随 120 次更新线性变化：早期偏探索和部分反馈，后期偏严格 EM 与效率。训练日志每步输出一行 `CEGR_METRICS`，正式训练结束后保存为：

```text
artifacts/improvement/search-r1-cegr-qwen2.5-1.5b-grpo-bm25/cegr_metrics.json
```

## 3. 云端恢复实例后检查

启动原 943 实例，不要释放原数据盘。登录后执行：

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -h
df -h /root/autodl-tmp
```

确认原仓库、模型和 baseline checkpoint 仍存在：

```bash
test -s /root/autodl-tmp/Search-R1/data/models/Qwen2.5-1.5B-Instruct/config.json
test -s /root/autodl-tmp/Search-R1/verl_checkpoints/search-r1-course-qwen2.5-1.5b-grpo-bm25/actor/global_step_120/config.json
echo $?
```

最后输出 `0` 才继续。

## 4. 创建独立改进 worktree

先保留云端临时修复：

```bash
cd /root/autodl-tmp/Search-R1
git status --short
git diff > /root/autodl-tmp/baseline-cloud-runtime.patch
git remote -v
git fetch origin codex/search-r1-improvement
git worktree add /root/autodl-tmp/Search-R1-improvement \
  -b cloud/search-r1-improvement origin/codex/search-r1-improvement
```

把大体积资产链接到改进 worktree，不重复下载：

```bash
cd /root/autodl-tmp/Search-R1-improvement
ln -s /root/autodl-tmp/Search-R1/data data
ln -s /root/autodl-tmp/Search-R1/verl_checkpoints verl_checkpoints
ln -s /root/autodl-tmp/Search-R1/artifacts artifacts
git rev-parse HEAD
```

若 `data`、`verl_checkpoints` 或 `artifacts` 已存在，先用 `ls -ld` 检查；不要用递归删除命令。目标是三个符号链接都指向原仓库。

## 5. 启动 BM25

沿用复现阶段已经验证过的 Java 17 启动方式。先检查端口：

```bash
curl -s http://127.0.0.1:8000/health || true
```

若服务未运行，在 tmux 中从原仓库启动原来成功的 Java 17 BM25 命令；然后在改进 worktree 验证：

```bash
cd /root/autodl-tmp/Search-R1-improvement
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
PYTHONPATH=$PWD python scripts/course_reproduction/check_retriever.py
```

## 6. 本地逻辑测试

```bash
cd /root/autodl-tmp/Search-R1-improvement
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
python -m unittest discover -s tests -v
```

必须全部通过后再占用训练时间。

## 7. 两步冒烟测试

```bash
tmux new -s cegr-smoke
cd /root/autodl-tmp/Search-R1-improvement
bash scripts/improvement/run_smoke.sh 2>&1 | tee cegr-smoke-console.log
```

检查：

```bash
grep 'CEGR_METRICS' cegr-smoke-console.log
cat artifacts/improvement/cegr-smoke/training_completed.txt
find verl_checkpoints/cegr-smoke -type d -name 'global_step_2' -print
```

合格条件：有两行左右 `CEGR_METRICS`、marker 为 `status=completed`、存在 `global_step_2`、无 OOM/NaN/Traceback。

## 8. 十步测速

```bash
tmux new -s cegr-timing
cd /root/autodl-tmp/Search-R1-improvement
bash scripts/improvement/run_timing.sh 2>&1 | tee cegr-timing-console.log
```

比较原 baseline 的十步耗时。CEGR 只增加字符串解析，理论上每步开销应远小于 rollout；若总耗时明显增加 10% 以上，应先检查日志或 I/O，而不是直接正式训练。

## 9. 120 步正式训练

确认正式 run 名下没有旧 checkpoint：

```bash
test ! -e verl_checkpoints/search-r1-cegr-qwen2.5-1.5b-grpo-bm25
```

然后：

```bash
tmux new -s cegr-formal
cd /root/autodl-tmp/Search-R1-improvement
bash scripts/improvement/train_cegr.sh 2>&1 | tee cegr-formal-console.log
```

根据原实验 `18334` 秒估算，纯训练约 5.1 小时。按 5.98 元/小时计算约 30.5 元，另加冒烟、测速和评测时间；实际费用必须用十步测速和平台当前单价重算。

完成后检查：

```bash
cat artifacts/improvement/search-r1-cegr-qwen2.5-1.5b-grpo-bm25/training_completed.txt
find verl_checkpoints/search-r1-cegr-qwen2.5-1.5b-grpo-bm25/actor \
  -type d -name 'global_step_120' -print
test -s artifacts/improvement/search-r1-cegr-qwen2.5-1.5b-grpo-bm25/cegr_metrics.json
```

## 10. 公平配对评测

下面的脚本会在同一固定 700 条样本上重新评测 baseline 与 CEGR，保存逐样本预测，并自动计算配对统计：

```bash
tmux new -s cegr-eval
cd /root/autodl-tmp/Search-R1-improvement
bash scripts/improvement/evaluate_cegr.sh 2>&1 | tee cegr-eval-console.log
```

关键产物：

```text
artifacts/improvement/baseline-vs-cegr.json
artifacts/improvement/paired-evaluation/baseline.jsonl
artifacts/improvement/paired-evaluation/cegr.jsonl
artifacts/improvement/paired-statistical-analysis.json
```

统计文件包含每数据集和总体的 EM/F1、差值、配对 bootstrap 95% CI、两个方向的翻转样本数和 exact McNemar p 值。

## 11. 证据归档

```bash
cd /root/autodl-tmp/Search-R1-improvement
bash scripts/improvement/collect_evidence.sh
sha256sum artifacts/improvement/evidence/search-r1-cegr-evidence.tar.gz
```

下载压缩包和哈希后再关机。不要在确认本地文件可打开之前释放实例。

## 12. 报告结论规则

主指标始终是七数据集固定子集宏平均 EM，不使用 CEGR 训练奖励替代效果指标。推荐表格：Pre-RL、Search-R1-EM、Search-R1-CEGR 三行；列出七数据集 EM、宏平均 EM、F1、平均搜索次数、重复/无效搜索率和响应长度。

结果未显著提升时也保留实验：分析是 F1 上升但 EM 不升、证据覆盖未变、还是搜索惩罚抑制了必要多跳探索。报告中明确区分“方法假设得到支持”“仅机制指标改善”和“改进失败”。
