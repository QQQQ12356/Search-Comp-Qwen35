# Search-Comp · Qwen3.5 迁移

把 [search-comp](../search-comp)（Search-R1 SFT + Activate Beacon 长上下文压缩）的
基础大模型从 **Qwen2.5 迁移到 Qwen3.5 系列**。conda 环境：`search-comp-qwen3.5`。

## 文档索引

| 想了解什么 | 看哪里 |
|---|---|
| 从零跑通训练 / 评测的完整流程、产物清单、排障 | [`WORKFLOW.md`](WORKFLOW.md) |
| 每个脚本的用法、数据构建、超参含义 | [`docs/usage_qwen3.5.md`](docs/usage_qwen3.5.md) |
| SearchAgent、检索信息与混合 Beacon 的逐层算法与张量流 | [`SEARCH_AGENT_BEACON_METHOD.md`](SEARCH_AGENT_BEACON_METHOD.md) |
| 迁移结论、里程碑结果、Beacon 移植状态 | 本文件下文 |

## 快速开始（新克隆）

```bash
# 1) 建环境（详见下文「环境」一节）
conda create -n search-comp-qwen3.5 python=3.11 -y
conda activate search-comp-qwen3.5
python -m pip install -r requirements.txt

# 2) 不加载模型权重的单元测试（应全绿：24 passed）
bash scripts/10_test.sh

# 3) 端到端冒烟：探针 -> 训练 -> 评测
bash scripts/00_searchagent_probe.sh                       # 自动建小语料并探检索能力
bash scripts/20_native_train.sh configs/train/native_qwen3.5.yaml
bash scripts/21_native_eval.sh outputs/models/native_qwen3_sft_v1/final
```

> **仓库不含大文件。** `outputs/data/`（语料，约 70MB）、`outputs/models/`
> （checkpoint）与 `outputs/logs/` 都不入库，分别由训练/评测脚本首次运行自动重建，
> 或从 Hugging Face 拉取。基础模型 `Qwen/Qwen3.5-2B` 权重首次运行时会从
> Hugging Face 下载并缓存到 `~/.cache/huggingface/hub/`。
> Search-R1 SFT 轨迹（约 60MB）需自行下载，命令见
> [`docs/usage_qwen3.5.md`](docs/usage_qwen3.5.md) 第 1.1 节。

> ⚠️ **重要架构事实**：`Qwen/Qwen3.5-2B` 不是 Qwen2 式纯因果 LM，而是**多模态
> 混合架构**（`Qwen3_5ForConditionalGeneration`）——24 层中仅 6 层为标准
> `full_attention`（带 QKV 缓存），其余 18 层为 `linear_attention`
> （GatedDeltaNet，固定尺寸循环状态）；并含 mRoPE、注意力输出门控、
> partial_rotary。因此 Beacon 压缩只能落在 6 个 full_attention 层上，移植工作量
> 远超「只换基础模型」，详见下文「里程碑 2：Beacon 移植到 Qwen3.5」。

---

## 环境

```bash
conda create -n search-comp-qwen3.5 python=3.11 -y
conda activate search-comp-qwen3.5
# 先装与本机 CUDA 匹配的 torch，再装其余依赖（实测环境为 CUDA 12.6 / torch 2.7.0）
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

> `transformers>=5` 才识别 `model_type=qwen3_5`（4.57 发行版不含），
> `requirements.txt` 已锁定该下界。Qwen3.5-2B 权重首次运行时从 Hugging Face
> 下载并缓存到 `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B`；离线环境请
> 预先 `huggingface-cli download Qwen/Qwen3.5-2B` 并设置 `HF_HOME`。
>
> 脚本默认优先使用名为 `search-comp-qwen3.5` 的 conda 环境；可用 `PYTHON=` 指定
> 其他解释器，用 `CONDA_ENV=` 换环境名。无需 conda 时确保 `python -c "import torch"`
> 可用即可。

---

## 目录

```
search_comp/
├── models/
│   ├── beacon_config.py     # Beacon 超参（复用，未改动）
│   ├── beacon_qwen3.py      # ★ Qwen3.5 版 Beacon（full_attention 层，研究脚手架）
│   └── modeling_utils.py    # 损失/张量工具（复用）
├── data/                    # 数据管线（复用，tokenizer 改为 Qwen3.5）
├── trainer/
│   └── native_trainer.py    # ★ 原生搜索轨迹 SFT（标准 Trainer + 可视化）
├── evaluation/
│   ├── native_interactive_eval.py  # ★ 原生交互式 SearchAgent 评估（EM/F1）
│   └── em_f1.py / evaluate.py      # 指标（复用）
└── milestones/              # ★ 里程碑脚本
    ├── build_small_corpus.py      # 小规模语料 + 简单 BM25 索引
    ├── qwen35_native.py           # 原生多模态模型加载 + 文本生成
    ├── qwen35_text.py             # 纯文本 Qwen3_5ForCausalLM 加载（训练用）
    └── searchagent_probe.py       # Milestone-1 检索能力探针
```

---

## 里程碑 1：Qwen3.5-2B 检索能力探针 ✅

在**不加 Beacon** 的前提下，用原生 Qwen3.5-2B + 小语料 + BM25 跑 SearchAgent，
验证基础模型是否支持检索（`think → <search> → observe → <answer>`）。

```bash
bash scripts/00_searchagent_probe.sh    # 构建小语料 + 生成轨迹
```

**结果**（HotpotQA validation）：
- 贪心解码（6 题）：**0% 触发搜索**，模型凭参数知识自信作答且常答错。
- 采样解码（temp 0.8，12 题）：**25% 触发搜索**，含一条真实 3 轮多跳轨迹
  （Annie Morton → Terry Richardson birth date → … celebrity），`<information>`
  注入 + 再推理正常。

**结论**：Qwen3.5-2B 原生具备调用 `<search>` 检索的能力，但**不可靠、格式脏**
（query 常带 `Query:`/`query:` 前缀），与参考项目结论一致——**必须 SFT 搜索
轨迹**才能可靠 think→search→observe→answer。

---

## 里程碑 3：原生搜索轨迹 SFT + 评估（完整可用）✅

标准 Trainer（tqdm 进度条 + 实时 loss 可视化）在搜索轨迹上做 SFT：

```bash
# 构建交互式轨迹 + 训练（可视化到终端）
bash scripts/20_native_train.sh configs/train/native_qwen3.5.yaml
# 交互式 SearchAgent 评估 + EM/F1
bash scripts/21_native_eval.sh outputs/models/native_qwen3_sft_v1/final
```

- 8-bit Adam（bitsandbytes）+ 梯度检查点，24GB 单卡可微调 1.88B。
- 已验证端到端：加载文本权重 → 标准 Trainer 训练（train_loss 1.525）→ 保存。

---

## 里程碑 2：Beacon 移植到 Qwen3.5（已实现并验证正确性）✅

`search_comp/models/beacon_qwen3.py` 把 Beacon 机制同时适配到 full attention 与
linear attention 两类层：

- **只压缩检索内容**：由 ``regions``（各 ``<information>`` 块）指定，文档按
  ``beacon_window`` 切窗，每窗末尾追加 ``window // beacon_ratio`` 个 beacon，
  beacon K/V 进入持久缓存，**原始文档 K/V 丢弃**。
- **损失不含检索内容**：labels 全局预移位后文档/beacon 位置为 -100，只在
  think/search/answer 生成片段上计算损失。
- `BeaconQwen3Attention`：独立 `beacon_q/k/v/o_proj`，`torch.where` 切换（含门控
  query），K/V 缓存保存 RoPE 前 key 并每窗口重施加 mRoPE。
- linear_attention 层不再跨压缩窗保留原生 `DynamicCache`。压缩窗内的
  GatedDeltaNet recurrent/conv state 仅作临时 reader 状态；窗结束后原生状态被
  丢弃，仅由该层 beacon 激活经 `BeaconLinearStateWriter` 重建持久 recurrent state，
  conv 尾状态清空。因此跨压缩边界不存在原始 `<information>` token 的隐藏状态旁路。

> 旧版 Beacon checkpoint 没有训练 linear-state writer，加载时会明确拒绝，而不会用
> 随机 writer 静默推理；需要用当前实现重新训练 Beacon 参数。

**正确性验证**（`bash scripts/verify_beacon.sh`）：
- 空压缩区（all-keep）beacon 前向 vs 原生前向损失差 < 0.05（实测 ~0.008）。
- 压缩：37 原始 token → 全注意力层缓存 12（4 问题 + 2 beacon + 6 答案），文档 K/V 被丢弃。
- 损失仅监督答案段，检索内容被掩码排除。

**训练**：`search_comp/trainer/beacon_trainer.py`（tqdm 可视化，8-bit AdamW）。
- 交互式轨迹数据（自动构建语料+轨迹）：`bash scripts/22_beacon_train.sh`
- Search-R1 官方 SFT 轨迹（messages 格式，`data_mode: searchr1`，读
  `outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl`，该文件**不在仓库内**，
  下载方式见 [`docs/usage_qwen3.5.md`](docs/usage_qwen3.5.md) 第 1.1 节）：
  `bash scripts/24_beacon_train_searchr1.sh`

> 显存注意：Beacon 前向保留跨窗口状态，不能直接对整个模型启用 Trainer 层级
> checkpoint。当前实现已对有效监督位置分块计算 LM head，并可对长样本启用
> loss checkpoint 与 CPU activation offload；24GB 单卡建议保持
> `beacon_loss_chunk_size=32`、`beacon_checkpoint_loss=true`、
> `beacon_cpu_offload_threshold=2048`。若仍 OOM，先降低该分块大小或限制样本长度。
> 训练结束后的 LoRA 合并默认在 CPU 执行，避免第二次占满 GPU 显存。

**推理**：`prefill_and_get_cache` + `decode_step` + `beacon_generate` 已实现，
把文档压缩为 beacon K/V 后自回归解码（`</search>`/`</answer>` 分段停止）。
`search_comp/evaluation/beacon_interactive_eval.py` 跑交互式 SearchAgent 并算 EM/F1。

`enable_beacon=False` 退化为原生 `Qwen3_5ForCausalLM` 前向。

### Beacon 交错布局

训练 YAML 的 `beacon.beacon_pos` 支持 `append`（默认，窗口末尾追加）和
`intersect`（检索内容内部交错）。例如：

```yaml
beacon:
  beacon_pos: "intersect"
  beacon_window: 256
  beacon_stride: 256
  beacon_ratio: 32
  beacon_attn: "full-coverage"
```

在现有配置中修改这些字段即可，其他字段保持不变。文档先按 `beacon_window`
个原始 token 分 chunk，每个 chunk 内每 `beacon_ratio` 个原始 token 插入一个
Beacon，末尾不足 ratio 个 token 也生成一个。上述配置每个完整 chunk 有 256 个
原始 token 和 8 个 Beacon，布局为 `token1…32 → B1 → token33…64 → B2 → …`。

chunk 内使用完整因果注意力：普通 token 和 Beacon 均可读取当前 chunk 内此前的
所有普通 token 和 Beacon，因此 B2 可以直接读取 token1…64。自位置保留标准因果
注意力语义，未来位置不可见。后续 chunk 只能读取前面 chunk 的 Beacon、保留的
非文档上下文，以及当前 chunk 内本位置之前的 token 和 Beacon，不能直接读取
前面 chunk 的原始文档 token。

实现逐 chunk 前向，chunk 结束才丢弃原始文档 K/V，仅提交 Beacon K/V。
Qwen3.5 线性层在 chunk 内保留 reader 状态，chunk 结束仅提交 Beacon writer
更新的循环状态并清空临时卷积状态。训练与推理共享该布局配置；不再按 ratio
拆成多次前向，chunk 划分与 `append` 一致。

线性状态隔离：压缩 chunk 的 reader 使用历史 recurrent/conv 状态的独立副本，
不允许通过原地写入污染持久状态。跨 chunk 的状态矩阵由
`writer(当前 chunk 的 Beacon 激活, chunk 开始前的持久状态)` 更新，
绝不提交当前 chunk 的原始 reader 最终状态或卷积尾状态。副本保留训练梯度。
这里隔离的是原始文档状态的直接传递；Beacon 压缩后携带的文档信息仍会按设计保留。

---

## 测试

```bash
bash scripts/10_test.sh            # 等价于 pytest tests -q，日志同时落到 outputs/logs/
# 24 passed（不需要 GPU；不加载模型权重）
```

> 默认测试会从 Hugging Face 拉取 Qwen3.5 的 **tokenizer 文件**（数 MB，不是模型
> 权重）到 `~/.cache/huggingface/`。完全离线时请先缓存，或设置
> `HF_HUB_OFFLINE=1` 复用已有缓存。

覆盖：EM/F1 指标、BM25 检索、Search-R1 轨迹压缩区与损失区定位、
Qwen3.5 tokenizer 推理特殊 token、轨迹
think→search→<information>→answer 的压缩区/损失区定位、统计汇总、配置覆盖与
运行元数据。

本地已缓存 Qwen3.5-2B 权重时，可追加真实权重回归（需要 GPU，约 1 分钟）：

```bash
FULL_MODEL_TEST=1 CUDA_VISIBLE_DEVICES=0 bash scripts/10_test.sh
```
