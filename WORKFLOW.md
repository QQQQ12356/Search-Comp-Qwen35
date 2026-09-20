# Search-Comp Qwen3.5 训练与评测流程

本文说明从数据准备、原生 SFT、Beacon SFT、交互式搜索评测到结果统计的完整流程。
所有 shell 入口都会在终端实时显示进度，并把相同输出保存到 `outputs/logs/`。
算法、状态边界和逐层张量流详见 `SEARCH_AGENT_BEACON_METHOD.md`。

## 1. 代码结构

```text
search_comp/
├── data/                  # 轨迹解析、数据集（含 Search-R1）、collator、BM25 检索
├── models/                # Qwen3.5 Beacon、纯文本 SFT 模型（plain_qwen3）与加载
├── trainer/               # 标准 Trainer 训练入口（plain_sft_trainer / beacon）
├── evaluation/            # 交互评测（plain_interactive_eval / beacon）、EM/F1 统计
├── milestones/            # 数据探针与 Qwen3.5 文本模型加载
└── utils/                 # 配置覆盖、运行元数据、JSONL Trainer 日志
configs/train/             # 可复用 YAML 训练配置
scripts/                   # 数据、训练、评测、测试、统计入口
tests/                     # 不下载大模型的单元与小模型回归测试
```

无 Beacon 的「纯文本 SFT」线各有一套最小实现，互不影响 Beacon 文件：

- 建模：`search_comp/models/plain_qwen3.py`
- 训练：`search_comp/trainer/plain_sft_trainer.py` + `configs/train/qwen35_plain_sft.yaml`
- 评测：`search_comp/evaluation/plain_interactive_eval.py`

核心数据协议为：模型生成 `<search>query</search>`，检索器返回
`<information>...</information>`，模型继续生成并最终输出
`<answer>...</answer>`。只有 `<information>` 文档 token 被 Beacon 压缩。

## 2. 环境安装

```bash
conda activate search-comp-qwen3.5
python -m pip install -r requirements.txt
```

可通过 `PYTHON=/path/to/python` 指定解释器；否则脚本优先使用
`$CONDA_ENV`，其默认值为 `search-comp-qwen3.5`。

### 2.1 仓库不含哪些数据（首次使用需补齐）

本仓库只跟踪代码与配置，以下内容都在 `.gitignore` 内，需按需获取或由脚本重建：

| 内容 | 位置 | 获取方式 |
| --- | --- | --- |
| 基础模型权重 | HF 缓存 | 首次运行时自动从 `Qwen/Qwen3.5-2B` 下载 |
| HotpotQA 语料 / 交互轨迹 | `outputs/data/` | `scripts/21`、`22`、`23` 首次运行自动构建 |
| Search-R1 SFT 轨迹 | `outputs/data/searchr1/` | 手动下载，见下（`20`/`24` 的输入） |
| checkpoint / 日志 / 评测结果 | `outputs/models/`、`outputs/logs/`、`outputs/results/` | 训练与评测时生成 |

Search-R1 SFT 轨迹（`20_native_train.sh` 与 `24_beacon_train_searchr1.sh` 的共同输入，
约 60MB）下载：

```bash
mkdir -p outputs/data/searchr1
huggingface-cli download --repo-type dataset PeterJinGo/nq_hotpotqa_train \
    --include '*instruct-sft.jsonl' --local-dir outputs/data/searchr1
```

已有该文件时可建软链复用，避免重复下载：

```bash
ln -s /path/to/qwen3-4b-instruct-sft.jsonl \
      outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl
```

## 3. 先运行测试

```bash
bash scripts/10_test.sh
```

该命令运行不下载大模型的完整单元测试。若本地已有 Qwen3.5 权重并希望验证真实模型：

```bash
FULL_MODEL_TEST=1 CUDA_VISIBLE_DEVICES=0 bash scripts/10_test.sh
```

真实模型验证检查：all-keep 数值一致性、full-attention K/V 压缩、线性层不存在
`DynamicCache` 旁路、压缩边界卷积状态清空以及 Beacon writer recurrent state 存在。

## 4. 原生 / 纯文本 SFT 训练

`scripts/20_native_train.sh` 走**纯文本 SFT**（无 Beacon、不压缩）：实现时参考
Beacon 各为一套建模 / 训练 / 评测文件，但做了最小化脱钩，互不影响 Beacon 文件。
训练数据是 Search-R1 官方 `messages` 轨迹
`outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl`（需先按 §2.1 准备），与 Beacon
训练（`beacon_qwen35_searchr1.yaml`）使用**同一份文件**，可直接对比「压缩 vs 不压缩」。

直接调用标准 Trainer：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/20_native_train.sh \
  configs/train/qwen35_plain_sft.yaml
```

无需编辑 YAML 即可覆盖参数：

```bash
bash scripts/20_native_train.sh configs/train/qwen35_plain_sft.yaml \
  --set learning_rate=1e-5 \
  --set max_train_steps=100 \
  --set exp_name=plain_smoke
```

### train / eval 严格对齐

评测（`build_search_chat_prompt`）的初始上下文是
`system = 角色说明 + 完整搜索协议`、`user = 裸问题`；而 Search-R1 官方数据原本把
搜索协议塞在首个 user。plain 与 Beacon 的 Search-R1 训练共用
`SearchR1SFTDataset`，装载时会**把协议移进 system、首个 user 只留裸问题**，使训练
看到的提示与评测逐字一致（见 `_align_to_eval_prompt`）。对齐只改输入上下文，
loss 掩码不变——system/user 仍为 `-100`，只有 assistant 输出（think / search /
answer）计损失。

Search-R1 轨迹**不截断**，collator 固定 `batch_size=1`，靠 `grad_accum_steps`
扩大有效 batch。

原生 Trainer checkpoint 可使用 Python 入口恢复（注意 `CUDA_VISIBLE_DEVICES`：
不指定时 Trainer 会把可见 GPU 全部用上并把 batch 放大，searchr1 的 collator 只接受 1）：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m search_comp.trainer.plain_sft_trainer \
  --config configs/train/qwen35_plain_sft.yaml \
  --resume_from_checkpoint outputs/models/qwen35_plain_sft_v1/checkpoint-1000
```

### LoRA 与合并

`use_lora: true`（默认）时冻结基础权重，只训练低秩适配器，`final/` 保存的是
adapter。评测入口加载的是完整模型，因此**评测前必须合并**：

```bash
python -u -m search_comp.trainer.merge_lora \
  --base_model_path Qwen/Qwen3.5-2B \
  --adapter_path outputs/models/qwen35_plain_sft_v1/final \
  --output_path outputs/models/qwen35_plain_sft_v1/final_merged
```

合并默认在 CPU 上进行，不需要额外预留 GPU 显存。中间的
`checkpoint-*` 目录同样是 adapter，可按需合并任意一个。

## 5. Beacon 模型训练

使用本项目构建的交互轨迹：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/22_beacon_train.sh \
  configs/train/beacon_qwen35.yaml
```

使用已有 Search-R1 messages 轨迹（输入文件需先按 §2.1 准备）：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/24_beacon_train_searchr1.sh \
  configs/train/beacon_qwen35_searchr1.yaml
```

常用覆盖示例：

```bash
bash scripts/24_beacon_train_searchr1.sh \
  configs/train/beacon_qwen35_searchr1.yaml \
  --set max_train_steps=200 \
  --set beacon.beacon_ratio=32 \
  --set beacon.beacon_linear_writer_rank=64 \
  --set exp_name=beacon_ratio32_smoke
```

Beacon 训练同样使用标准 `transformers.Trainer` 负责 dataloader、梯度累积、进度条和
日志。模型保存使用额外 callback，以确保 LoRA adapter 与所有 Beacon 参数完整保存。
旧版 checkpoint 没有 linear-state writer，不能用于当前安全压缩路径，必须重新训练。

## 6. 训练参数

| 参数 | 含义 | 建议 |
| --- | --- | --- |
| `model_name_or_path` | 基础模型或 checkpoint | `Qwen/Qwen3.5-2B` |
| `train_data_path` | JSONL 训练集 | 必填 |
| `val_data_path` | 可选验证集 | 设置后按 `eval_steps` 验证 |
| `data_mode` | `interactive` 或 `searchr1` | 与 JSONL 格式一致 |
| `max_length` | 单条轨迹最大长度 | 显存不足时减小 |
| `per_device_batch_size` | 单卡 batch | Beacon 当前使用 1 |
| `grad_accum_steps` | 梯度累积步数 | 用于扩大有效 batch |
| `learning_rate` | 学习率 | LoRA/Beacon 通常 `1e-5` 至 `5e-5` |
| `optim` | Trainer 优化器 | `adamw_torch` 或 `adamw_bnb_8bit` |
| `lr_scheduler_type` | Trainer 学习率调度器 | Beacon 默认 `constant` |
| `warmup_steps` | warmup 步数 | constant 可设 0 |
| `max_grad_norm` | 梯度裁剪阈值 | 默认 1.0 |
| `use_bf16` | BF16 混合精度 | 支持 BF16 的 GPU 建议开启 |
| `num_epochs` | 数据轮数 | `max_train_steps=null` 时生效 |
| `max_train_steps` | 优化器总步数 | 非 null 时覆盖 epoch |
| `logging_steps` | Trainer 指标显示/写盘间隔 | 调试 1，正式训练 5–20 |
| `save_freq_steps` | 模型保存间隔 | 根据训练时长设置 |
| `use_lora` | 是否使用 LoRA | 单卡训练建议开启 |
| `beacon_window` | 每个 information 压缩窗口长度 | 需能被 ratio 整除 |
| `beacon_ratio` | 原始文档 token / Beacon token | 越大压缩越强 |
| `beacon_linear_writer_rank` | 线性层状态 writer 秩 | 默认 128；小模型可降低 |
| `beacon_loss_chunk_size` | 有效监督 token 的 LM-head 分块 | 越小峰值显存越低 |
| `beacon_checkpoint_loss` | 反向时重算分块 logits | 24GB 单卡建议开启 |
| `beacon_cpu_offload_activations` | 将保存激活卸载到 CPU | 仅显存不足时开启 |
| `beacon_cpu_offload_threshold` | 仅对达到该长度的样本 offload | 24GB + 2B Search-R1 建议 2048 |

训练结束时脚本会在独立进程中把 LoRA 与 Beacon 参数合并为 `final` checkpoint。
合并默认使用 CPU，不需要为合并过程预留一份完整的 GPU 模型显存；若只需要
adapter，可直接使用 `final_adapter`，跳过合并阶段。

### 常见日志提示

- `HF_TOKEN` 未设置只会降低 Hugging Face Hub 请求速率；模型已在本地缓存时不影响训练。
- `flash-linear-attention` 或 `causal-conv1d` 缺失时会回退到 PyTorch 实现，主要影响速度，不是 OOM 的直接原因。
- 若出现 `CUDA out of memory`，先确认 `beacon_loss_chunk_size: 32`、
  `beacon_checkpoint_loss: true` 和 `beacon_cpu_offload_threshold: 2048`，再逐步将
  `beacon_loss_chunk_size` 降到 `16` 或提高 offload 覆盖范围。

## 7. 交互式评测

原生 / 纯文本模型（`scripts/21_native_eval.sh` → `search_comp.evaluation.plain_interactive_eval`，
LoRA 训练时传**合并后**的目录，见 §4）：

```bash
MAX_QUESTIONS=200 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/21_native_eval.sh \
  outputs/models/qwen35_plain_sft_v1/final_merged \
  outputs/results/qwen35_plain_sft_v1/predictions.jsonl
```

评测使用与训练严格对齐的同一提示词（`system` 含完整搜索协议、`user` 为裸问题），
真实 BM25 检索 + 多轮 `think → <search> → <information> → <answer>`，生成时不做任何压缩。

Beacon 模型：

```bash
MAX_QUESTIONS=200 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/23_beacon_eval.sh \
  outputs/models/beacon_qwen3_searchr1_v3/final \
  outputs/results/beacon_qwen3_searchr1_v3/predictions.jsonl
```

评测会逐题写入 JSONL，进程中断时可继续：

```bash
RESUME=1 bash scripts/23_beacon_eval.sh MODEL_PATH RESULT_PATH
```

可通过环境变量修改 `MAX_QUESTIONS`、`MAX_TURNS`、`TOP_K`、
`MAX_DOCS_TOKENS`；额外 Python 参数可直接追加在两个位置参数之后。

## 8. 统计与实验比较

```bash
bash scripts/25_summarize_results.sh outputs/results/comparison.json \
  outputs/results/qwen35_plain_sft_v1/predictions.jsonl \
  outputs/results/beacon_qwen3_searchr1_v3/predictions.jsonl
```

统计包括 EM、F1、作答率、搜索触发率、多轮搜索数、平均轮次、总/平均/中位/P95
耗时、information token 数、Beacon token 数和有效压缩比。

## 9. 输出目录

```text
outputs/
├── data/                         # 语料和训练轨迹
├── logs/*.log                    # 终端完整日志
├── models/<exp_name>/
│   ├── resolved_config.json      # 覆盖后的实际配置
│   ├── run_metadata.json         # 命令、Python、Torch、GPU 等环境
│   ├── dataset_summary.json      # 数据量和参数量
│   ├── trainer_metrics.jsonl     # 实时 Trainer 指标
│   ├── trainer_state_summary.json
│   ├── train_summary.json
│   ├── checkpoint-*/             # 周期 checkpoint/adapter
│   └── final/                    # 最终可评测模型
└── results/
    ├── predictions.jsonl         # 每题轨迹，逐题落盘
    └── predictions_metrics.json  # 评测参数与汇总指标
```

复现实验时至少保留：YAML、`resolved_config.json`、`run_metadata.json`、完整日志、
`trainer_metrics.jsonl`、最终模型、预测 JSONL 和 metrics JSON。
