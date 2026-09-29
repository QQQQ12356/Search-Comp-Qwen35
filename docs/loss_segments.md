# 分段损失（loss_segments）

按片段类别控制「哪些 token 计损失、各占多少权重」。由
`beacon.loss_segments` 配置，实现见 `search_comp/loss_segments.py`。

## 1. 为什么需要

Beacon SFT 的损失是**监督 token 的全局平均**。而一条 Search-R1 轨迹里各类片段的
token 数差异极大——用本仓库的分类器在 4000 条真实轨迹（185 万监督 token）上实测：

| 类别 | 占监督 token |
| --- | --- |
| `think_content` | **89.18%** |
| `think_tag` | 3.21% |
| `search_content` | 3.15% |
| `search_tag` | 1.90% |
| `answer_tag` | 1.29% |
| `answer_content` | **0.746%** |
| `other` | 0.53% |

`<answer>` 内容平均只占 **0.95%**（p50 0.74%）。也就是说 thinking 的篇幅直接稀释了
answer 的梯度份额。本模块把这件事变成可配置、可消融的。

## 2. 类别定义

| 类别 | 覆盖的 token |
| --- | --- |
| `think_tag` | `<thinking>` / `<think>` / `</thinking>` / `</think>` |
| `think_content` | 上述标签之间的思考正文 |
| `search_tag` | `<search>` / `</search>` |
| `search_content` | 标签之间的检索 query |
| `answer_tag` | `<answer>` / `</answer>` |
| `answer_content` | 标签之间的最终答案 |
| `other` | 其余被监督的 assistant token（标签间的换行等） |

标签本身与正文是**分开**的类别——`<answer>` 这个标签是「决定停止思考并作答」的
那个决策 token，它和答案内容的作用完全不同，可以单独加权。

## 3. 权重语义

每个监督 token 的交叉熵乘以它所属类别的权重，分母同步换成权重和：

```
L = Σ_i w_i · CE_i / Σ_i w_i
```

三个直接推论：

1. **权重是相对量**。把 `answer_content` 调大不会整体放大损失或有效学习率，只改变
   类别之间的份额（测试 `test_weights_are_relative_and_renormalised` 钉住这一点）。
2. **`enabled: false` 等价于 `weight: 0`**。该类既不进分子也不进分母，完全不产生
   梯度。如果所有类别都关掉，损失恒为 0。
3. **全默认（都 enabled、weight=1.0）与不加权逐位一致**，并且此时连分类都不做
   （不额外分词），历史行为零改动。

## 4. 怎么配

YAML（`configs/train/beacon_qwen35_searchr1.yaml` 已给出全默认值）：

```yaml
beacon:
  loss_segments:
    think_content:  {enabled: true, weight: 0.2}
    answer_tag:     {enabled: true, weight: 5.0}
    answer_content: {enabled: true, weight: 15.0}
```

命令行覆盖（支持点号路径，YAML 简写也认）：

```bash
python -m search_comp.trainer.beacon_trainer \
  --config configs/train/beacon_qwen35_searchr1.yaml \
  --set beacon.loss_segments.answer_content.weight=15.0 \
  --set beacon.loss_segments.think_content.enabled=false
```

允许的简写：`{enabled: true, weight: 2.0}`、`true` / `false`、或直接一个数字
（`answer_content: 15.0`）。

### 想要 answer 占到损失的 10%

`answer_content` 基线占比 `p = 0.00746`，其余类别保持 1.0，则加权后份额为
`w·p / (w·p + (1-p))`。令其等于 0.10 解得 **w ≈ 15**：

```
w = 15  →  15×0.00746 / (15×0.00746 + 0.99254) = 0.112 / 1.104 = 10.1%
```

同类推：想把整个 `<answer>…</answer>` 块（标签+内容，基线 2.04%）提到 10%，两个
类别的 weight 都设到 **≈ 5.3**。

## 5. 作用范围

- **训练**：`data_mode: searchr1` 与 `interactive` 两条路径都支持，分别在
  `SearchR1SFTDataset._segment_ids_for` 与 `build_sequence_ids` 里分类。
- **推理**：完全不影响。`beacon_generate` 不传 `loss_segment_ids`，权重路径不激活。
- **辅助损失**：question memory v1 的 readout 蒸馏项是在加权平均**之后**相加的，
  不受 `loss_segments` 影响。
- **checkpoint**：`loss_segments` 会随 `config.json` 落盘。它不是结构参数，因此
  `set_beacon_config` 允许运行中改；旧 checkpoint 的 config 里没有该字段时回落到
  全默认。

## 6. 已知限制

- 分类依赖 tokenizer 的 `offset_mapping`（fast tokenizer）。不支持时整段退化为
  `other`（即全部 weight=1.0），不会错位但也不分类。
- token 归属按其**起始字符**所在区间判定；跨标签边界的 token 归左区间。标签由
  标点分隔，实际不构成问题。
- 本机制只调整**损失份额**。已有的 ckpt1000 评测显示失败模式主要是「停不下来」
  （50% 的题没吐出 `<answer>`，其中 45/50 卡在 thinking 里，格式化后 EM 22% vs
  整体 11%）。要打这个靶子，加权对象应该是 `*_tag`（停止决策 token）而不是
  `answer_content`。

## 7. 测试

```bash
HF_HUB_OFFLINE=1 python -m pytest tests/test_loss_segments.py -q
```

覆盖：标签/正文区分、无 offset mapping 的降级、配置往返与非法值、加权 CE 与手算
一致、权重为 0 处无梯度、权重是相对量、全默认时与历史路径逐位一致、数据集产出的
类别 id 与 labels 对齐。
