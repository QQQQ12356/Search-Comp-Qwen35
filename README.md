# Search-Comp · Qwen3.5 迁移

把 [search-comp](../search-comp)（Search-R1 SFT + Activate Beacon 长上下文压缩）的
基础大模型从 **Qwen2.5 迁移到 Qwen3.5 系列**。conda 环境：`search-comp-qwen3.5`。

> ⚠️ **重要架构事实**：`Qwen/Qwen3.5-2B` 不是 Qwen2 式纯因果 LM，而是**多模态
> 混合架构**（`Qwen3_5ForConditionalGeneration`）——24 层中仅 6 层为标准
> `full_attention`（带 QKV 缓存），其余 18 层为 `linear_attention`
> （GatedDeltaNet，固定尺寸循环状态）；并含 mRoPE、注意力输出门控、
> partial_rotary。因此 Beacon 压缩只能落在 6 个 full_attention 层上，移植工作量
> 远超「只换基础模型」，详见下文「Beacon 移植状态」。

> 📖 **训练/测试脚本与超参数说明**：见 [`docs/usage_qwen3.5.md`](docs/usage_qwen3.5.md)。

---

## 环境

```bash
conda create -n search-comp-qwen3.5 python=3.11 -y
conda activate search-comp-qwen3.5
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==5.9.0" accelerate datasets rank-bm25 sentencepiece \
            numpy tqdm pyyaml safetensors protobuf bitsandbytes pytest
```

> `transformers>=5` 才识别 `model_type=qwen3_5`（4.57 发行版不含）。Qwen3.5-2B
> 权重已缓存在本地 `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B`。

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

`search_comp/models/beacon_qwen3.py` 把 Beacon 机制落到 **6 个 full_attention 层**，
语义与参考实现完全一致：

- **只压缩检索内容**：由 ``regions``（各 ``<information>`` 块）指定，文档按
  ``beacon_window`` 切窗，每窗末尾追加 ``window // beacon_ratio`` 个 beacon，
  beacon K/V 进入持久缓存，**原始文档 K/V 丢弃**。
- **损失不含检索内容**：labels 全局预移位后文档/beacon 位置为 -100，只在
  think/search/answer 生成片段上计算损失。
- `BeaconQwen3Attention`：独立 `beacon_q/k/v/o_proj`，`torch.where` 切换（含门控
  query），K/V 缓存保存 RoPE 前 key 并每窗口重施加 mRoPE。
- linear_attention 层用 transformers `DynamicCache` 跨窗口承接 GatedDeltaNet 循环状态。

**正确性验证**（`bash scripts/verify_beacon.sh`）：
- 空压缩区（all-keep）beacon 前向 vs 原生前向损失差 < 0.05（实测 ~0.008）。
- 压缩：37 原始 token → 全注意力层缓存 12（4 问题 + 2 beacon + 6 答案），文档 K/V 被丢弃。
- 损失仅监督答案段，检索内容被掩码排除。

**训练**：`search_comp/trainer/beacon_trainer.py`（tqdm 可视化，8-bit AdamW）。
- 交互式轨迹数据（自动构建语料+轨迹）：`bash scripts/22_beacon_train.sh`
- Search-R1 官方 SFT 轨迹（messages 格式，`data_mode: searchr1`，直接用
  `Search-R1-SFT-jsonl/qwen3-4b-instruct-sft.jsonl`）：`bash scripts/24_beacon_train_searchr1.sh`

> 显存注意：beacon 手写窗口前向未做逐层梯度检查点（linear_attention 层有
> DeltaNet 循环状态，checkpoint 需谨慎处理），因此 24GB 单卡训练长序列
> （max_length≈8192）可能 OOM。可调小 `max_length`/`beacon_window`，或后续接入
> flash-linear-attention + causal-conv1d 快速内核（见 Qwen3.5 加载时的提示）。

**推理**：`prefill_and_get_cache` + `decode_step` + `beacon_generate` 已实现，
把文档压缩为 beacon K/V 后自回归解码（`</search>`/`</answer>` 分段停止）。
`search_comp/evaluation/beacon_interactive_eval.py` 跑交互式 SearchAgent 并算 EM/F1。

`enable_beacon=False` 退化为原生 `Qwen3_5ForCausalLM` 前向。

---

## 测试

```bash
PYTHONPATH=. pytest tests/ -q     # 12 passed（em_f1 / retrieval / 里程碑烟雾）
```

覆盖：EM/F1 指标、BM25 检索、Qwen3.5 tokenizer 推理特殊 token、轨迹
think→search→<information>→answer 的压缩区/损失区定位。
