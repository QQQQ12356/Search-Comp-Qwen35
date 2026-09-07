"""EM / F1 评估指标（HotpotQA 标准）。

- ``extract_answer``：从模型输出中提取最后一个 ``<answer>...</answer>`` 的内容。
- ``normalize_answer``：标准化（去冠词/标点/小写/压缩空格）。
- ``compute_em`` / ``compute_f1``：分别计算精确匹配与 token 级 F1。
- ``compute_metrics``：对一批预测计算平均 EM/F1。

F1 使用 SQuAD 风格的 token 级 F1（预测与金标准的最优对齐）。
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple

#: 提取 <answer> 标签内的内容
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(solution_str: str) -> Optional[str]:
    """从模型输出中提取最后一个 ``<answer>...</answer>`` 的内容。

    Args:
        solution_str: 模型生成的完整输出文本。

    Returns:
        提取的答案字符串；未找到时返回 None。
    """
    matches = list(_ANSWER_PATTERN.finditer(solution_str))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def normalize_answer(s: str) -> str:
    """标准化答案：去冠词、去标点、转小写、压缩空白。"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_em(prediction: str, ground_truth: str) -> float:
    """精确匹配（EM）：标准化后是否完全一致。

    Args:
        prediction: 模型预测答案。
        ground_truth: 金标准答案。

    Returns:
        0.0 或 1.0。
    """
    return (
        1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0
    )


def _f1_score(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    """计算两个 token 序列的 F1。"""
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1(prediction: str, ground_truth: str) -> float:
    """token 级 F1（SQuAD 风格，对答案取最优 F1）。

    Args:
        prediction: 模型预测答案。
        ground_truth: 金标准答案。

    Returns:
        F1 分数（0~1）。
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    return _f1_score(pred_tokens, gold_tokens)


def compute_metrics(
    predictions: List[Tuple[str, str, str]],
) -> Dict[str, float]:
    """对一批 ``(prediction, ground_truth, answer_str)`` 计算平均 EM/F1。

    Args:
        predictions: ``(id, prediction, ground_truth)`` 列表。

    Returns:
        ``{"em", "f1", "valid_count"}`` 的 dict。
    """
    ems, f1s, valid = [], [], 0
    for _qid, pred, gold in predictions:
        ems.append(compute_em(pred, gold))
        f1s.append(compute_f1(pred, gold))
        valid += 1
    if valid == 0:
        return {"em": 0.0, "f1": 0.0, "valid_count": 0}
    return {
        "em": sum(ems) / valid,
        "f1": sum(f1s) / valid,
        "valid_count": valid,
    }
