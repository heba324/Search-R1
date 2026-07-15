# CEGR V2 实验操作手册

> 最终采用的实验路径：CEGR V1 和课程 baseline 已完成；V2 先运行 2 步冒烟，再从原始 Qwen2.5-1.5B-Instruct 训练 120 步并完成固定 700 题评测。
>
> 分支：`codex/search-r1-cegr-v2`
>
> V1 冻结提交：`8672aad0f4089f0fca388601cd9ce20fc9b8b776`

## 1. 先读结论

可以在 2 步冒烟通过后直接做 120 步，但要遵守以下边界：

1. **baseline 不重新训练**。已有 baseline `global_step_120` 是完整训练产物，重训既花钱，也会因随机性产生另一条 baseline。
2. **baseline 必须重新评测**。旧 checkpoint 要与新 EFF120 checkpoint 使用同一份 700 题、同一检索器、同一 parser、同一 seed 和同一评测脚本重新生成轨迹。
3. 2 步只检查能否训练，不接着训练。正式 120 步会重新从原始 `Qwen2.5-1.5B-Instruct` 起跑，保证 baseline 和 EFF120 都各自接受 120 次参数更新。
4. 单臂不是算法要求。本实验复用已有 baseline，从而避免重复训练一个 120 步模型。
5. 这条路线比较的是“旧 Search-R1 baseline”与“分组修复 + EFF”的整体差异，**不能单独归因**为 F1 fallback 的效果。

若必须严格回答“变化来自分组修复还是 F1 fallback”，仍需另行预注册并训练一个从原始 Qwen 起跑的 `Grouped-EM-120` 对照。本次已完成结果只估计“分组修复 + EFF”的组合效应。

## 2. 实验契约

| 项目 | 已有 baseline | 新 EFF120 |
|---|---:|---:|
| 起始模型 | Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct |
| 训练更新数 | 120 | 120 |
| 训练 batch | 32 | 32 |
| group size | 5 | 5 |
| 学习率 | `1e-6` | `1e-6` |
| warmup ratio | `0.95` | `0.95` |
| 检索 | CPU BM25, Top-3 | CPU BM25, Top-3 |
| 最大搜索轮次 | 4 | 4 |
| 奖励/分组 | 原课程 baseline | 修复分组后的 EFF |
| 正式评测 | 重新评测固定 700 题 | 重新评测同一固定 700 题 |

本路线省掉了第二个新训练臂和训练中的周期性 700 题验证，因此是“关键训练预算匹配”，不是全流程计算量完全相同的随机对照试验。分析文件会自动记录这个限制。

## 3. 租机与资源

推荐沿用已经选定的实例：

```text
1 x A800 80GB
RAM 实际可用至少 110 GiB
/root/autodl-tmp 实际可用至少 420 GiB
数据盘标称约 505 GB
Ubuntu 22.04 + CUDA 12.1 开发镜像
```

如果页面仍显示 5.98 元/小时，以创建实例时的实时价格为准。单卡训练时间不能在实测前保证；总费用按下面计算：

```text
总费用 = 实例实际运行秒数 / 3600 x 实时单价
```

正式费用包含 2 步冒烟、120 步训练、baseline 700 题复评、EFF120 700 题评测和证据整理。单臂只省掉一个额外 120 步训练，不能省掉公平评测。

## 4. 开机后先检查硬件

登录服务器后先不要下载或训练，执行：

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -h
df -h /root/autodl-tmp
df -h /dev/shm
nvcc --version
```

继续实验前应看到：

```text
GPU 为 A800 80GB
内存建议不低于 110 GiB
/root/autodl-tmp 可用空间不低于 420 GiB
/dev/shm 建议至少 32 GiB
nvcc 可以正常输出版本
```

若数据盘只显示约 50 GB，立即关机并在平台控制台处理扩容，不要开始下载。

## 5. 确认 V1 资产仍在

以下命令假定完成 V1 的目录是 `/root/autodl-tmp/Search-R1-improvement`。若实际目录不同，只修改第一行：

```bash
export V1_ROOT=/root/autodl-tmp/Search-R1-improvement
test -d "$V1_ROOT" || { echo "找不到 V1_ROOT"; exit 1; }
git -C "$V1_ROOT" rev-parse HEAD
git -C "$V1_ROOT" status --short
```

第一条 Git 命令必须输出：

```text
8672aad0f4089f0fca388601cd9ce20fc9b8b776
```

`status --short` 应无输出。再检查关键资产：

```bash
test -s "$V1_ROOT/data/models/Qwen2.5-1.5B-Instruct/config.json"
test -s "$V1_ROOT/verl_checkpoints/search-r1-course-qwen2.5-1.5b-grpo-bm25/actor/global_step_120/config.json"
test -s "$V1_ROOT/artifacts/improvement/paired-evaluation/baseline.jsonl"
test -d "$V1_ROOT/data/wiki18_bm25/bm25"
echo "V1 关键资产存在"
```

任一 `test` 失败都先停止。应从已保存的数据盘或证据备份恢复 V1，不要临时重训 baseline。

## 6. 获取 V2 代码

推荐使用独立目录，避免触碰冻结的 V1：

```bash
export V2_ROOT=/root/autodl-tmp/Search-R1-cegr-v2
cd /root/autodl-tmp
git clone --branch codex/search-r1-cegr-v2 \
  https://github.com/heba324/Search-R1.git "$V2_ROOT"
cd "$V2_ROOT"
git branch --show-current
git rev-parse HEAD
git status --short --branch
```

若该 V2 目录已经存在，不要再次 clone，改用：

```bash
export V2_ROOT=/root/autodl-tmp/Search-R1-cegr-v2
cd "$V2_ROOT"
git fetch origin
git switch codex/search-r1-cegr-v2
git pull --ff-only origin codex/search-r1-cegr-v2
git rev-parse HEAD
git status --short --branch
```

最终提交哈希以交付消息和仓库分支最新提交为准。把 `git rev-parse HEAD` 输出记进实验记录。

## 7. 链接 V1 资产并做离线预检

V2 只链接模型、数据、baseline checkpoint 和冻结证据；V2 输出保留在新目录：

```bash
cd "$V2_ROOT"
V1_ROOT="$V1_ROOT" bash scripts/improvement_v2/link_assets.sh
bash scripts/improvement_v2/prepare.sh
```

`prepare.sh` 会检查 V1 哈希、生成不重叠 pilot、运行全部单元测试、编译 Python、检查所有 shell 语法，并确认 V2 没有修改 `search_r1/` 或 `verl/`。

只有最后看到以下文字才继续：

```text
CEGR V2 offline preparation passed.
```

检查 Conda 环境仍存在：

```bash
conda env list
```

应至少有：

```text
Search-R1
Search-R1-retriever
```

## 8. 启动 BM25 检索服务

新建第一个 tmux 会话：

```bash
tmux new -s bm25
cd "$V2_ROOT"
bash scripts/course_reproduction/launch_bm25_retriever.sh
```

看到服务启动后，按 `Ctrl+B`，松开，再按 `D`，让它在后台继续运行。

回到普通终端后验证：

```bash
cd "$V2_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
python -m scripts.course_reproduction.check_retriever
```

检索检查失败时不要训练。可用 `tmux attach -t bm25` 查看服务报错。

## 9. 运行 2 步冒烟

新建第二个 tmux 会话：

```bash
tmux new -s direct120-smoke
cd "$V2_ROOT"
bash scripts/improvement_v2/run_smoke.sh
```

它会检查：

```text
原始 Qwen 模型能加载
BM25 能返回结果
分组固定为同题 5 条 rollout
EFF fallback 在 2 步内产生至少 10% 的有效稠密信号
没有 NaN/Inf
global_step_2 checkpoint、日志、奖励指标和完成标记齐全
```

成功后执行：

```bash
cd "$V2_ROOT"
cat artifacts/improvement-v2/search-r1-cegr-v2-eff-direct120-smoke/training_completed.txt
python -m scripts.improvement_v2.verify_training \
  --repo-root "$V2_ROOT" \
  --run-name search-r1-cegr-v2-eff-direct120-smoke \
  --method eff --steps 2 --group-size 5 --minimum-signal 0.10 \
  --seed 42 \
  --initial-model "$V2_ROOT/data/models/Qwen2.5-1.5B-Instruct" \
  --train-batch-size 8 --learning-rate 1e-6 --lr-warmup-ratio 0.95
```

必须看到 `Verified completed V2 run`。2 步 checkpoint 仅用于验收，不作为 120 步的起点。

若冒烟失败，不要删除目录后假装成功，也不要调低门槛。保留 `train.log`、`reward_metrics.json` 和报错，先定位 OOM、检索中断、NaN 或 fallback 无信号。

## 10. 直接运行正式 120 步

冒烟通过后可立即开始，无需再做 10 步测速：

```bash
tmux new -s direct120-train
cd "$V2_ROOT"
bash scripts/improvement_v2/run_train.sh
```

按 `Ctrl+B`、再按 `D` 退出 tmux。监控方法：

```bash
watch -n 2 nvidia-smi
```

另一个终端查看日志：

```bash
tail -f "$V2_ROOT/artifacts/improvement-v2/search-r1-cegr-v2-eff-direct120-qwen2.5-1.5b-grpo-bm25/train.log"
```

恢复训练窗口：

```bash
tmux attach -t direct120-train
```

脚本会保存：

```text
verl_checkpoints/search-r1-cegr-v2-eff-direct120-qwen2.5-1.5b-grpo-bm25/actor/global_step_40
verl_checkpoints/search-r1-cegr-v2-eff-direct120-qwen2.5-1.5b-grpo-bm25/actor/global_step_80
verl_checkpoints/search-r1-cegr-v2-eff-direct120-qwen2.5-1.5b-grpo-bm25/actor/global_step_120
```

正式完成后校验：

```bash
cd "$V2_ROOT"
bash scripts/improvement_v2/run_train.sh
cat artifacts/improvement-v2/search-r1-cegr-v2-eff-direct120-qwen2.5-1.5b-grpo-bm25/training_completed.txt
```

第二次调用不会重训；它只验证已有完整结果。若第一次中断，脚本会拒绝覆盖残留现场。此时不要直接重跑，应先保留日志并分析故障。

## 11. 公平复评 baseline 和 EFF120

确认 BM25 tmux 仍在运行：

```bash
tmux ls
cd "$V2_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
python -m scripts.course_reproduction.check_retriever
```

开始两次固定 700 题评测：

```bash
tmux new -s direct120-eval
cd "$V2_ROOT"
bash scripts/improvement_v2/run_evaluation.sh
```

该脚本依次完成：

```text
校验 V1 冻结证据
严格重算历史 baseline 轨迹，仅作审计
用当前统一评测器重新生成 baseline 的 700 条轨迹
用同一评测器生成 EFF120 的 700 条轨迹
做逐题配对、Bootstrap 95% CI 和 exact McNemar 检验
```

这里是复评旧 checkpoint，不是重新训练 baseline。

## 12. 查看最终结果

```bash
cd "$V2_ROOT"
export RESULT=artifacts/improvement-v2/direct120-final-evaluation/step-120/direct120-analysis.json
test -s "$RESULT"
python -m json.tool "$RESULT" | less
```

重点查看：

```text
comparison.overall.em_delta
comparison.overall.f1_delta
comparison.overall.em_delta_bootstrap_95_ci
comparison.overall.mcnemar_exact_p
effectiveness.predeclared_success
effectiveness.statistically_supported
effectiveness.primary_metric_pass
effectiveness.guardrails_pass
effectiveness.claim_level
causal_limit
```

预注册成功门槛是：整体 EM 至少提高 `0.02`，F1 不下降，single-hop EM 不下降，同时证据覆盖、有效搜索、重复搜索和响应长度不越过安全边界。统计支持还要求 EM 的 Bootstrap 95% CI 下界大于 0，且 McNemar `p < 0.05`。

结果解释：

```text
statistically_supported_improvement
  可以写“在本实验设置下获得统计支持的整体改进”。

directional_improvement
  可以写“达到预注册效果门槛，但统计证据不足”。

not_effective_on_primary_metric
  必须写“未达到预注册主要指标门槛”，继续分析失败原因，不能宣称有效。

primary_gain_with_guardrail_failure
  EM 达到主要门槛，但至少一项次要安全护栏失败；必须准确报告是哪项护栏失败，不能写成完整改进成功。
```

无论结果正负，都只能归因于“分组修复 + EFF 整体方案”。没有 Grouped-EM-120 新对照时，不能写“实验证明 F1 fallback 单独带来提升”。

## 13. 收集与备份证据

```bash
cd "$V2_ROOT"
REQUIRE_FINAL_CHECKPOINT=true bash scripts/improvement_v2/collect_evidence.sh
ls -lh artifacts/improvement-v2/evidence/
cd artifacts/improvement-v2/evidence
sha256sum -c search-r1-cegr-v2-evidence.tar.gz.sha256
tar -tzf search-r1-cegr-v2-evidence.tar.gz | \
  grep 'search-r1-cegr-v2-eff-direct120-qwen2.5-1.5b-grpo-bm25/actor/global_step_120/config.json'
cd "$V2_ROOT"
```

至少保存：

```text
search-r1-cegr-v2-evidence.tar.gz
search-r1-cegr-v2-evidence.tar.gz.sha256
direct120-analysis.json
EFF120 的 global_step_120 checkpoint
W&B 曲线或训练日志
```

启用最终 checkpoint 门禁后，checkpoint 缺失会直接失败；成功时完整 `global_step_120` 会进入证据压缩包。确认压缩包已下载、SHA-256 校验成功且上述 `tar -tzf` 能找到 checkpoint 后，才关闭并释放云实例。

在你自己的 Windows PowerShell 中下载时，把尖括号内容替换为租机页面的 SSH 信息：

```powershell
scp -P <SSH端口> root@<服务器地址>:/root/autodl-tmp/Search-R1-cegr-v2/artifacts/improvement-v2/evidence/search-r1-cegr-v2-evidence.tar.gz .
scp -P <SSH端口> root@<服务器地址>:/root/autodl-tmp/Search-R1-cegr-v2/artifacts/improvement-v2/evidence/search-r1-cegr-v2-evidence.tar.gz.sha256 .
```

下载后在 Windows PowerShell 校验：

```powershell
$Expected = (Get-Content .\search-r1-cegr-v2-evidence.tar.gz.sha256).Split()[0].ToUpper()
$Actual = (Get-FileHash .\search-r1-cegr-v2-evidence.tar.gz -Algorithm SHA256).Hash
if ($Actual -ne $Expected) { throw "SHA-256 校验失败" }
Write-Host "SHA-256 校验通过: $Actual"
```

若平台不开放 `scp`，使用平台文件管理器下载同名两个文件。checkpoint 已装入压缩包，不必再单独逐文件下载模型目录。

## 14. 报告中的准确表述

推荐写法：

> 在资源和时间受限条件下，本实验复用已完成的 120 步 Search-R1 baseline checkpoint，并从同一 Qwen2.5-1.5B-Instruct 初始模型训练 120 步 CEGR V2 EFF 模型。两者使用统一固定 700 题协议重新评测。该对比估计分组修复与 EM-First F1-Fallback 的组合效果；由于未新增 Grouped-EM-120 对照，不对 F1 fallback 的独立因果贡献作结论。

不能写：

```text
只看旧日志就说新模型优于 baseline
把 2 步 checkpoint 接着训练并声称双方都是独立 120 步
没有 Grouped-EM-120 却声称 F1 fallback 单独有效
未达到门槛仍写“改进成功”
```

## 15. 最短命令清单

前面的检查全部通过后，真正执行阶段只有三条：

```bash
bash scripts/improvement_v2/run_smoke.sh
bash scripts/improvement_v2/run_train.sh
bash scripts/improvement_v2/run_evaluation.sh
```

每一条必须成功结束后才能执行下一条。不要并行训练和评测，不要同时启动第二个 GPU 任务。
