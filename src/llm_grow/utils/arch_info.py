"""Architecture info parser and parameter counter utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass
class ArchInfo:
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    vocab_size: int = 0
    max_position_embeddings: int = 0
    model_type: str = ""
    extra: dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


def parse_arch_info(model: nn.Module) -> ArchInfo:
    """从 model.config 解析架构参数，返回 ArchInfo 数据类。"""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return ArchInfo()

    return ArchInfo(
        hidden_size=getattr(cfg, "hidden_size", 0),
        intermediate_size=getattr(cfg, "intermediate_size", 0),
        num_hidden_layers=getattr(cfg, "num_hidden_layers", 0),
        num_attention_heads=getattr(cfg, "num_attention_heads", 0),
        num_key_value_heads=getattr(cfg, "num_key_value_heads", 0),
        vocab_size=getattr(cfg, "vocab_size", 0),
        max_position_embeddings=getattr(cfg, "max_position_embeddings", 0),
        model_type=getattr(cfg, "model_type", ""),
    )


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数量（默认统计全部参数）。"""
    return sum(
        p.numel()
        for p in model.parameters()
        if not trainable_only or p.requires_grad
    )


def param_diff_report(
    original: nn.Module,
    expanded: nn.Module,
) -> None:
    """打印扩增前后的参数量对比报告。"""
    orig_total = count_params(original)
    exp_total = count_params(expanded)
    exp_trainable = count_params(expanded, trainable_only=True)

    orig_info = parse_arch_info(original)
    exp_info = parse_arch_info(expanded)

    print("\n" + "=" * 55)
    print("  Parameter Expansion Report")
    print("=" * 55)
    print(f"  Original  total  : {orig_total:>15,}  ({orig_total/1e9:.2f}B)")
    print(f"  Expanded  total  : {exp_total:>15,}  ({exp_total/1e9:.2f}B)")
    print(f"  Expanded trainable: {exp_trainable:>14,}  ({exp_trainable/1e9:.2f}B)")
    print(f"  Expansion ratio  : {exp_total / orig_total:.3f}x")
    if orig_info.num_hidden_layers and exp_info.num_hidden_layers:
        print(f"  Layers: {orig_info.num_hidden_layers} → {exp_info.num_hidden_layers}")
    if orig_info.hidden_size and exp_info.hidden_size:
        print(f"  Hidden: {orig_info.hidden_size} → {exp_info.hidden_size}")
    if orig_info.intermediate_size and exp_info.intermediate_size:
        print(f"  FFN:    {orig_info.intermediate_size} → {exp_info.intermediate_size}")
    print("=" * 55 + "\n")
