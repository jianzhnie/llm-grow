"""Layer freezing / unfreezing utilities for staged training."""
from __future__ import annotations

import torch.nn as nn


def freeze_original_layers(model: nn.Module) -> int:
    """冻结所有非新增层（没有 _is_new_growth 标记的参数）。

    Returns:
        被冻结的参数数量。
    """
    frozen = 0
    for param in model.parameters():
        if not getattr(param, "_is_new_growth", False):
            param.requires_grad_(False)
            frozen += param.numel()
    return frozen


def unfreeze_all(model: nn.Module) -> int:
    """解冻全部参数，用于 Phase-2 全量微调。

    Returns:
        解冻的参数数量。
    """
    unfrozen = 0
    for param in model.parameters():
        param.requires_grad_(True)
        unfrozen += param.numel()
    return unfrozen


def freeze_layers_by_index(
    model: nn.Module,
    layer_indices: list[int],
    layer_attr: str = "model.layers",
) -> None:
    """按层序号冻结指定层。"""
    layers = _get_attr(model, layer_attr)
    if layers is None:
        raise AttributeError(f"Cannot find {layer_attr!r} in model.")
    for idx in layer_indices:
        for param in layers[idx].parameters():
            param.requires_grad_(False)


def report_trainable(model: nn.Module) -> dict[str, int]:
    """统计可训练 vs 冻结参数量，返回字典。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(
        f"[Params] trainable={trainable:,}  frozen={frozen:,}  total={total:,}  "
        f"({100 * trainable / total:.1f}% trainable)"
    )
    return {"trainable": trainable, "frozen": frozen, "total": total}


def _get_attr(obj: object, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj
