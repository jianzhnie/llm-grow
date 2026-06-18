"""Layer freezing / unfreezing utilities for staged training."""

from __future__ import annotations

import torch.nn as nn

from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


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


def mark_new_params(model: nn.Module, original_param_ids: set[int]) -> int:
    """根据扩增前的参数快照，标记所有新增参数。

    Args:
        model: 扩增后的模型。
        original_param_ids: 扩增前通过 ``snapshot_param_ids(model)`` 获取的
            参数 id 集合。

    Returns:
        被标记的新增参数元素数量。
    """
    count = 0
    for param in model.parameters():
        if id(param) not in original_param_ids:
            param._is_new_growth = True
            count += param.numel()
    return count


def snapshot_param_ids(model: nn.Module) -> set[int]:
    """获取模型当前所有参数的 id 集合。扩增前调用以建立快照。

    用法::

        from llm_grow.training.freeze import snapshot_param_ids, mark_new_params

        original_ids = snapshot_param_ids(model)
        expander.expand(model, config)
        mark_new_params(model, original_ids)
        freeze_original_layers(model)
    """
    return {id(p) for p in model.parameters()}


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
    logger.info(
        "trainable=%s  frozen=%s  total=%s  (%.1f%% trainable)",
        f"{trainable:,}",
        f"{frozen:,}",
        f"{total:,}",
        100 * trainable / total,
    )
    return {"trainable": trainable, "frozen": frozen, "total": total}


def _get_attr(obj: object, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj
