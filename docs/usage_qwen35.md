# Search-Comp · Qwen3.5 使用说明

本文档说明 Qwen3.5 版 SearchAgent 的**训练 / 测试脚本**、**数据构建**、以及
**全部超参数含义**。conda 环境：`search-comp-qwen3.5`。

---

## 0. 快速总览

| 目标 | 脚本 | 说明 |
|------|------|------|
| 单元测试（不下载大模型） | `scripts/10_test.sh` | `FULL_MODEL_TEST=1` 时追加真实权重验证 |
| 检索能力探针（先验证基础模型） | `scripts/00_searchagent_probe.sh` | 小语料 + 原生生成，看模型会不会搜索 |
| 原生 SFT 训练 | `scripts/20_native_train.sh` | 标准 Trainer，交互式轨迹数据 |
| 原生评估 | `scripts/21_native_eval.sh` | EM/F1 |
| Beacon SFT 训练（交互式轨迹） | `scripts/22_beacon_train.sh` | 自动建语料+轨迹，Beacon 压缩 |
| Beacon 评估 | `scripts/23_beacon_eval.sh` | 真实语料+BM25 索引+交互式 Agent |
| Beacon SFT 训练（Search-R1 数据） | `scripts/24_beacon_train_searchr1.sh` | **推荐**，官方 SFT 轨迹 |
| Beacon 正确性验证 | `scripts/verify_beacon.sh` | 等价性 + 压缩生效校验 |
| 多实验汇总统计 | `scripts/25_summarize_results.sh` | EM/F1/耗时/压缩比对比表 |

> **推荐主流程**（Beacon + Search-R1 数据）：
> 1. `bash scripts/10_test.sh`（先确认环境与代码自洽）
> 2. `bash scripts/verify_beacon.sh`（验证移植正确性，需已缓存 Qwen3.5-2B）
> 3. `bash scripts/24_beacon_train_searchr1.sh`（训练，需先下载 SFT 轨迹）
> 4. `bash scripts/23_beacon_eval.sh outputs/models/<exp>/final`（评估 EM/F1）
>
> 更完整的流程、产物清单与排障见仓库根目录 [`WORKFLOW.md`](../WORKFLOW.md)；
> 算法与张量流见 [`SEARCH_AGENT_BEACON_METHOD.md`](../SEARCH_AGENT_BEACON_METHOD.md)。

---

## 1. 训练脚本

### 1.1 `24_beacon_train_searchr1.sh`（推荐）

用 **Search-R1 官方 SFT 轨迹**训练 Beacon 模型：

```bash
bash scripts/24_beacon_train_searchr1.sh [config]
# 默认 config = configs/train/beacon_qwen35_searchr1.yaml
```

- 训练数据：`outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl`
  （10000 条 messages 格式轨迹，**无需构建数据**）。
  该文件约 60MB，**不在本仓库内**，首次使用需自行下载：

  ```bash
  mkdir -p outputs/data/searchr1
  huggingface-cli download --repo-type dataset PeterJinGo/nq_hotpotqa_train \
      --include '*instruct-sft.jsonl' --local-dir outputs/data/searchr1
  ```

  已有该文件（或位于别处）时，建软链或直接覆盖配置：

  ```bash
  ln -s /path/to/qwen3-4b-instruct-sft.jsonl outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl
  # 或
  bash scripts/24_beacon_train_searchr1.sh configs/train/beacon_qwen35_searchr1.yaml \
      --set train_data_path=/path/to/qwen3-4b-instruct-sft.jsonl
  ```
- 数据格式：`{messages: [{role: system}, {role: user, ...Question}, {role: assistant, <think>...<search>...}, {role: user, <information>...}, {role: assistant, <think>...<answer>...}]}`。
- 处理逻辑（`searchr1_dataset.py`）：
  - 手动 ChatML 渲染（不触发 Qwen3.5 模板的空 `<think>` 块与思考剥离）。
  - 训练数据 `<thinking>` → 原生 `<think>` 特殊 token。
  - `<information>` 文档块 → **Beacon 压缩区**；损失只算 assistant 生成片段。
- 文档格式：原始轨迹里的文档是 `[Document N] (ID: ..., Score: ...)` + 带引号标题行的旧
  格式，而评测检索结果是另一种渲染，两者不一致会让模型看到分布外的文档格式。训练前先
  用 `bash scripts/29_normalize_docs.sh` 改写成统一格式（`Doc N` + 标题行 + 原样正文，
  开头连续重复的标题折叠成一个；无 ID/Score；见 `WORKFLOW.md` §2.1），再把
  `train_data_path` 指向新文件。

### 1.2 `22_beacon_train.sh`（交互式轨迹数据）

```bash
bash scripts/22_beacon_train.sh [config]
# 默认 config = configs/train/beacon_qwen35.yaml
```

自动执行：① 构建语料（`build_corpus`，train/val 各 5000 样本）→ ② 构建交互式
轨迹（`build_interactive_data`，train 2000 条）→ ③ 训练。

### 1.3 `20_native_train.sh`（无 Beacon 的原生 SFT）

```bash
bash scripts/20_native_train.sh [config]
# 默认 config = configs/train/native_qwen35.yaml
```

用标准 `transformers.Trainer`（tqdm 进度条 + 实时 loss）做原生搜索轨迹 SFT，
作为基线 / `enable_beacon=False` 的对照。

---

## 2. 测试 / 评估脚本

### 2.1 `23_beacon_eval.sh`（Beacon 交互式评估）

```bash
bash scripts/23_beacon_eval.sh [model_path] [result_path]
# 默认 model = outputs/models/beacon_qwen3_searchr1_v1/final
```

真实构建：语料库（不存在则 `build_corpus`）→ `BM25Retriever` 内建索引 →
交互式 SearchAgent（`beacon_generate` 分段压缩解码）→ EM/F1。

Agent 流程（`beacon_interactive_eval.py`）：
1. 模型生成 `<think>...</think><search>query</search>`（`</search>` 停止）。
2. 提取 query → BM25 在线检索 top-k → 追加 `<information>` 文档块（token 区间记为压缩区）。
3. 继续生成，最多 `max_turns` 轮，直到 `</answer>`。

### 2.2 `21_native_eval.sh`（原生评估）

```bash
bash scripts/21_native_eval.sh [model_path] [result_path]
```

无 Beacon 的原生交互式评估，EM/F1。

### 2.3 `00_searchagent_probe.sh`（能力探针）

```bash
bash scripts/00_searchagent_probe.sh
```

小语料 + 原生生成若干条轨迹，验证基础模型是否支持检索（think→search→observe→answer）。

### 2.4 `verify_beacon.sh`（正确性验证）

```bash
bash scripts/verify_beacon.sh
```

1. 空压缩区 beacon 前向 ≈ 原生前向（损失差 < 0.05）。
2. 压缩生效：检索文档 K/V 被丢弃、只保留 beacon，损失不含文档。

---

## 3. 超参数说明

### 3.1 Beacon 压缩超参数（`configs/*.yaml` 的 `beacon:` 块）

对应 `search_comp/models/beacon_config.py` 的 `BeaconConfig`。

| 参数 | 默认 | 含义 |
|------|------|------|
| `enable_beacon` | `true` | 是否启用 Beacon 压缩；`false` 退化为原生因果 LM 前向 |
| `beacon_window` | 256/512 | 滑动窗口大小（token）。检索文档按此切窗 |
| `beacon_stride` | =window | 窗口步长；本实现用 append 模式且 `stride == window`（无重叠） |
| `beacon_ratio` | 32 | **压缩比**：每 `ratio` 个原始 token → 1 个 beacon。如 32 表示 32:1 压缩 |
| `beacon_param` | `"q k v"` | 为 beacon 引入哪些独立投影矩阵（`q`/`k`/`v`/`o` 空格分隔子集） |
| `beacon_attn` | `"full-coverage"` | beacon 注意力模式（仅支持 full-coverage：beacon 关注窗口内全部 token） |
| `beacon_pos` | `"append"` | beacon 放置方式（仅支持 append：追加在窗口末尾） |
| `beacon_embed_init` | `"eos"` | beacon 嵌入初始化来源（`eos`/`bos`） |
| `beacon_doc_region_split` | `false` | 把每个 `<information>` 块按子文档切成多个压缩段（每篇文档一段）。续写监督的正确性依赖它 |
| `beacon_continuation_loss_weight` | `0.0` | 续写损失权重；`0` = 关闭（逐位等价于历史行为）。见 §3.1.1 |
| `beacon_continuation_tokens` | `4` | 每个读出点预测的目标 token 数 k（两种模式都生效，见 §3.1.1） |
| `beacon_continuation_per_beacon` | `false` | 续写监督粒度：`false` = window 模式，`true` = beacon 模式（见 §3.1.1） |
| `beacon_keep_window` | `null` | keep 段切窗上限（`null` 跟随 `beacon_window`，`0` 不切）；只影响前向次数与峰值显存，不改变结果 |

> **压缩效果**：一个满窗口 `beacon_window` 个 token 末尾追加
> `beacon_window // beacon_ratio` 个 beacon；不满窗口按 `ceil(剩余/ratio)` 生成。
> 只压缩 `<information>` 检索文档块，其余（question/think/search/answer）为 keep。

#### 3.1.1 续写监督：window 模式 vs beacon 模式

续写损失只监督「左侧没有可见原文、只能靠已提交的 beacon 记忆预测下一个 token」的位置。
粒度由 `beacon_continuation_per_beacon` 二选一：

| | window 模式（默认） | beacon 模式 |
|---|---|---|
| 开关 | `beacon_continuation_per_beacon: false` | `beacon_continuation_per_beacon: true` |
| 监督位置 | 每个压缩段**非首窗**窗头的前 k 个 token | 每个 chunk 边界：该 chunk 的 beacon → 其后 k 个 token |
| 读出头 | 窗头第 `0..k-1` 行隐状态（k 个目标各读一行） | 该 chunk 的 beacon 隐状态（k 个目标**共用一行**） |
| 参数要求 | 需 `beacon_pos: "append"`（intersect 布局不产生窗头监督） | **强制 `beacon_window == beacon_ratio`** |
| 段尾 | 段内最后一个窗口可能监督到段外 token | 目标钳制在 `seg_end` 内，永不跨段 |

> `beacon_continuation_tokens`（默认 4）在两种模式下都生效，但含义不同：window 模式的 k
> 个目标来自 k 个**不同**位置的隐状态；beacon 模式下 chunk 隔离后唯一「没有可见原文」的
> 读出点只有 beacon 自己，所以 k 个目标必须由**同一个**向量预测（类似多 token 预测的辅助
> 头）。`k <= beacon_ratio` 时目标正好是下一个 chunk 的 k 个 token；k 更大则伸进更后面的
> chunk（仍在同一压缩段内）。注意默认值 4 意味着 beacon 模式下每个 beacon 要预测 4 个
> token，只想要「每个 chunk 边界 1 个 token」就显式设 `beacon_continuation_tokens: 1`。

> **为什么 beacon 模式强制 `beacon_window == beacon_ratio`**：chunk 隔离只能来自
> **窗口边界**——窗末只提交 beacon、丢弃原始 K/V，于是下一个 chunk 的 token 左侧只有已
> 提交的 beacon。改注意力掩码做不到：`attn_mask` 只作用于 full-attention 层，18 个
> linear-attention 层是**无掩码**的循环扫描，窗内跨 chunk 的原始 token 仍会经残差流泄漏。
> 窗口大于 ratio 时，窗内 beacon 仍能直接看见本窗更早的原文，「仅凭压缩记忆预测」不成立。
> 代价：文档区前向次数 = 段长 / ratio（96→16 即 6 倍）；keep 段仍由 `beacon_keep_window`
> 独立控制粒度。
>
> **跨子文档伪续写**：`beacon_doc_region_split: false` 时一个压缩段横跨多篇子文档，
> 段内的窗口边界会监督「预测下一篇文档的首 token」——而下一篇是检索拼进来的，与上文无
> 因果关系。实测（`window=4`，文本 `"Doc 1 A\n\nDoc 2 B"`，第二篇首 token 在位置 9）：

| 配置 | 目标是新文档首 token 的监督点 |
|------|------|
| window 模式 + `doc_region_split: false` | `(窗起始 8 → 目标 9)` |
| window 模式 + `doc_region_split: true` | `(窗起始 8 → 目标 9)` ← **仍然存在** |
| beacon 模式 + `doc_region_split: true` | 无 |

> 要完全避免：`beacon_continuation_per_beacon: true` + `beacon_window == beacon_ratio` +
> `beacon_doc_region_split: true`。根因是 `split_document_regions` 把分隔空行 `\n\n` 划给
> **前一篇**，于是前一篇压缩段的最后一个窗口压在换行上、目标正好是下一篇的首 token；
> window 模式只能靠 `doc_region_split: true` 把暴露面收敛到「每个子文档段最多 1 个点」，
> 当子文档最后一窗长度 ≤ `beacon_continuation_tokens`（约 `k/window` 比例的子文档）时仍会命中。
> k 只改变监督点数量，不改变这个性质：beacon 模式的目标恒被钳制在 `seg_end` 内，k 再大也不会跨段。
>
> 训练/评测启动时 `describe_layout()` 会打印**实际生效**的粒度，例如
> `…；续写监督=每 beacon 1 个`。intersect 布局配 window 模式会明确打印
> 「关闭（intersect 不产生窗头监督）」，避免出现「权重设了但其实不生效」的静默漂移。

### 3.2 训练超参数（`configs/*.yaml` 顶层）

| 参数 | 默认 | 含义 |
|------|------|------|
| `model_name_or_path` | `Qwen/Qwen3.5-2B` | 基础模型（本地已缓存） |
| `train_data_path` | 见各 config | 训练数据 JSONL 路径 |
| `data_mode` | 见各 config | 数据模式：`searchr1`（Search-R1 messages）/ `interactive`（交互式轨迹）。native 配置默认 `searchr1`，`beacon_qwen35.yaml` 为 `interactive` |
| `max_length` | 8192 | 最大 token 数（超出的压缩区/损失区被丢弃）；`searchr1` 轨迹不截断，此参数只对 `interactive` 生效 |
| `learning_rate` | `5.0e-5` | AdamW 学习率 |
| `weight_decay` | `0.01` | 权重衰减 |
| `num_epochs` | `1` | 训练轮数 |
| `grad_accum_steps` | `8` | 梯度累积步数（bs=1 时扩大有效 batch = 1×8） |
| `save_freq_steps` | `200` | 每多少步保存 checkpoint |
| `seed` | `42` | 随机种子 |
| `exp_name` | — | 实验名（模型保存到 `outputs/models/<exp_name>/`） |
| `output_dir` | `outputs` | 输出根目录 |

Beacon 训练（`beacon_trainer.py`）额外说明：

- 优化器：优先 `bitsandbytes` 8-bit AdamW（24GB 单卡控制显存），回退普通 AdamW。
- 交互式/searchr1 数据均 `batch_size=1`（样本结构多变，用梯度累积扩大 batch）。
- **显存优化**（24GB 单卡必须开启）：
  - `use_lora: true` → LoRA 冻结基础模型，只训练低秩适配器 + beacon 参数，
    可训练参数从 1.88B 降到 ~74M（-10GB 优化器/梯度显存）。
  - `gradient_checkpointing: true` → 逐层梯度检查点，linear_attention 层的
    O(seq) 中间量在反向时重算而非保留（-激活显存）。

LoRA 相关超参数：

| 参数 | 默认 | 含义 |
|------|------|------|
| `use_lora` | `false` | 是否用 LoRA（推荐 `true` 以降低显存） |
| `lora_r` | `16` | LoRA 秩 |
| `lora_alpha` | `32` | LoRA 缩放系数 |
| `lora_dropout` | `0.05` | LoRA dropout |
| `gradient_checkpointing` | `true` | 逐层梯度检查点（省激活显存） |

> LoRA 目标模块：`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`；
> beacon 投影（`beacon_q/k/v/o_proj`）与 `beacon_embed_tokens` 保持**全量可训练**。
> 训练结束后自动 `merge_and_unload`，保存为标准 `BeaconQwen3_5ForCausalLM` checkpoint。

原生训练（`native_trainer.py`，标准 Trainer）额外参数：

| 参数 | 默认 | 含义 |
|------|------|------|
| `per_device_batch_size` | `1` | 每卡 batch；`interactive` 可 >1（统一 padding），`searchr1` collator 强制为 1，用 `grad_accum_steps` 扩大有效 batch。注意多卡可见时 DataLoader batch 会 ×GPU 数，单卡请设 `CUDA_VISIBLE_DEVICES=0` |
| `warmup_ratio` | `0.05` | 学习率预热比例 |
| `use_bf16` | `true` | 是否 bf16 混合精度 |
| `log_freq_steps` | `5` | 每多少步打印 loss |
| `optim` | `adamw_bnb_8bit` | 优化器（8-bit Adam 控显存） |
| `gradient_checkpointing` | `true` | 梯度检查点（省显存） |
| `use_lora` | 见各 config | LoRA（冻结基础权重只训适配器）；native 默认 `true`。保存的是 adapter，评测前需 `python -m search_comp.trainer.merge_lora` 合并 |
| `lora_r` / `lora_alpha` / `lora_dropout` | `16` / `32` / `0.05` | LoRA 秩、缩放与 dropout |

### 3.3 数据构建超参数（脚本环境变量）

| 变量 | 默认 | 含义 |
|------|------|------|
| `CORPUS` | `outputs/data/hotpotqa_corpus.jsonl` | 语料路径 |
| `CORPUS_PER_SPLIT` | `5000` | 每个 split 取多少样本构建语料 |
| `MAX_TRAIN` | `2000` | 训练轨迹条数 |
| `MAX_DOCS_TOKENS` | `1024` | 每轮检索文档最大 token（截断） |
| `TOP_K` | `3` | 检索 top-k |
| `MAX_QUESTIONS` | 100/200 | 评估题目数 |
| `MAX_TURNS` | `4` | 交互式搜索最大轮数（训练轨迹最多 3 次 search + 1 次 answer） |
| `MAX_NEW_TOKENS_PER_TURN` | `768` | 每轮生成上限（覆盖训练 assistant 段 p99≈580；过小会在 `</search>`/`</answer>` 前截断） |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU 选择 |

> 生成预算的取值依据见 `search_comp/evaluation/budget.py` 与 `WORKFLOW.md` §7；
> 三条评测路径（beacon / plain / native）共用这两个常量。

---

## 4. 输出产物

```
outputs/
├── data/                          # 语料、轨迹数据
│   ├── hotpotqa_corpus.jsonl
│   └── hotpotqa_train_interactive.jsonl
├── models/<exp_name>/             # 训练产物
│   ├── config.json                # 超参数快照（含 beacon 字段）
│   ├── train_summary.json         # 训练统计
│   ├── checkpoint-<step>/         # 中间检查点
│   └── final/                     # 最终模型（HF 格式，含 beacon 参数）
└── results/<exp>/                 # 评估结果
    ├── predictions.jsonl          # {id, prediction, ground_truth, turns, queries}
    └── predictions_metrics.json   # {em, f1, valid_count}
```

---

## 5. 环境

```bash
conda create -n search-comp-qwen3.5 python=3.11 -y
conda activate search-comp-qwen3.5
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==5.9.0" accelerate datasets rank-bm25 sentencepiece \
            numpy tqdm pyyaml safetensors protobuf bitsandbytes pytest
```

> `transformers>=5` 才识别 `model_type=qwen3_5`（4.57 发行版不含）。
