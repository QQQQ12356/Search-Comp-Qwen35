# Question memory v1

此实现通过 `beacon.beacon_question_memory_v1` 隔离，默认 `false`。
关闭时仍使用原 Writer、参数结构和更新公式。已有 checkpoint 不允许运行中切换
这个结构开关；请从基础模型构造并训练 v1，或加载已保存的 v1 checkpoint。

## 配置与单卡训练

在已有训练 YAML 的 `beacon` 下增加：

```yaml
beacon_question_memory_v1: true
beacon_question_max_tokens: 128
beacon_readout_distill_weight: 0.1
```

也可以不修改原 YAML，直接使用已有覆盖入口：

```bash
CUDA_VISIBLE_DEVICES=0 python -m search_comp.trainer.beacon_trainer \
  --config configs/train/beacon_qwen35_searchr1.yaml \
  --set beacon.beacon_question_memory_v1=true \
  --set beacon.beacon_question_max_tokens=128 \
  --set beacon.beacon_readout_distill_weight=0.1
```

CUDA 的有效变量名称是 `CUDA_VISIBLE_DEVICES`，不是 `CUDA_VISIBLE_DEVICE`。

## 行为与边界

- Interactive 数据使用原始 question；Search-R1 数据只使用首个 assistant 之前的
  user 内容（可能包括任务模板），不包括文档、搜索输出、答案。
- 模型首次调用需要 `question_input_ids`，保留前 `beacon_question_max_tokens` 个
  token 的 embedding；空问题报错，不从整条训练轨迹猜测问题。
- question 经低秩投影条件化 Writer，输出投影零初始化；相关性门与归一化残差
  新颖性联合控制写入强度，并以 Beacon 数量归一化窗口内衰减。
- 读出蒸馏只在训练且有 labels 时计算，教师是当前窗口的临时 Reader state，
  不是额外完整原始模型。使用 question 经 Writer key 投影得到的归一化探针；
  探针和教师都停止梯度。它是读出代理目标，不是原生卷积 query 的精确重放。
- 蒸馏为各压缩窗口、各 linear 层损失的平均值，权重可设为零；推理不计算。
- 持久文档信息仍只能经过 Beacon；不保留临时 Reader state 或原文卷积尾状态。
- 增量搜索复用首次 question，开始新样本时重置；不能在同一缓存中切换问题。
- 本版只条件化 Writer，不修改 Beacon 编码器、KV 淘汰或 keep 段原生更新。
  不提供当前子问题单独编码、双状态或子空间保护。

新增条件向量是有界的额外持久存储；比较方法时应计入显存和计算预算。
功能测试通过不代表问答准确率或长期记忆性能已经提升，仍需固定预算对照评测。

## 回归测试

```bash
CUDA_VISIBLE_DEVICES=0 BEACON_TEST_DEVICE=cuda PYTHONPATH=. python -m pytest \
  tests/test_beacon_question_memory.py \
  tests/test_beacon_qwen35_linear_memory.py \
  tests/test_beacon_interactive_cache.py tests/test_searchr1_dataset.py -q
```

v1 测试通过 `BEACON_TEST_DEVICE=cuda` 强制在唯一可见 GPU 上运行，默认 CPU。

使用本地已缓存的真实模型执行一条短样本的优化器更新和两 token 生成：

```bash
CUDA_VISIBLE_DEVICES=0 BEACON_TEST_DEVICE=cuda \
BEACON_FULL_MODEL=Qwen/Qwen3.5-2B HF_HUB_OFFLINE=1 \
HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=. \
python -m pytest tests/test_beacon_question_memory.py -q -s
```

2026-09-23 验证：RTX 4090，唯一可见 GPU 为物理卡 0，BF16；上述测试 17 项通过。
真实模型短样本 smoke 使用窗口 32、压缩率 8、Writer 中间维度 16；loss 为
5.935566，完成反向、优化器更新和两 token 生成，PyTorch 峰值 allocated 显存
4.272 GiB。这不是正式训练配置的显存估算或准确率评测；环境使用 PyTorch
线性注意力回退实现，未验证额外安装的融合算子。
