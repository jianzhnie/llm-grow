"""Growth scheduler for MSG-style progressive mask unlocking."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch.nn as nn

from llm_grow.training.freeze import mark_new_params, snapshot_param_ids
from llm_grow.utils.insertion import NEW_GROWTH_ATTR


@dataclass
class GrowthScheduleConfig:
    total_steps: int = 100_000
    """总训练步数。"""

    warmup_ratio: float = 0.3
    """前 warmup_ratio 比例的步数内保持原始参数空间（新维度 mask=0）。"""

    strategy: str = "linear"
    """解锁策略：'linear' | 'cosine' | 'step'。"""


class GrowthScheduler:
    """MSG 渐进式掩码生长调度器。

    用法::

        scheduler = GrowthScheduler(config)
        for step, batch in enumerate(dataloader):
            ratio = scheduler.get_unlock_ratio(step)
            scheduler.apply_masks(model, ratio)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
    """

    def __init__(self, config: GrowthScheduleConfig):
        self.config = config

    def get_unlock_ratio(self, step: int) -> float:
        """返回当前步数对应的新参数解锁比例 [0, 1]。"""
        cfg = self.config
        warmup_steps = int(cfg.total_steps * cfg.warmup_ratio)
        if step < warmup_steps:
            return 0.0
        progress = (step - warmup_steps) / max(cfg.total_steps - warmup_steps, 1)
        if cfg.strategy == "linear":
            return min(progress, 1.0)
        if cfg.strategy == "cosine":
            return 0.5 * (1 - math.cos(math.pi * min(progress, 1.0)))
        if cfg.strategy == "step":
            return min(int(progress * 4), 4) * 0.25
        raise ValueError(f"Unknown growth strategy: {cfg.strategy!r}")

    def apply_masks(self, model: nn.Module, unlock_ratio: float) -> None:
        """按 unlock_ratio 调整新增参数的 requires_grad 和梯度缩放。

        简化实现：unlock_ratio < 1 时对新增参数施加梯度缩放，
        完整实现应维护二进制掩码矩阵。
        """
        for _name, param in model.named_parameters():
            if getattr(param, NEW_GROWTH_ATTR, False):
                param.requires_grad_(unlock_ratio > 0)
                if hasattr(param, "_growth_scale"):
                    param._growth_scale = unlock_ratio  # type: ignore[attr-defined]

    def register_new_params(
        self, model: nn.Module, original_param_ids: set[int] | None = None
    ) -> int:
        """标记模型中的新增参数为"新增"（设置 ``_is_new_growth = True``）。

        Args:
            model: 扩增后的模型。
            original_param_ids: 扩增前通过 ``snapshot_param_ids(model)`` 获取的
                参数 id 集合。如果为 None，则退化为计算所有已设置
                ``_is_new_growth`` 的参数数量。

        Returns:
            被标记的新增参数元素数量。
        """
        if original_param_ids is not None:
            return mark_new_params(model, original_param_ids)
        count = 0
        for param in model.parameters():
            if getattr(param, NEW_GROWTH_ATTR, False):
                count += param.numel()
        return count

    @staticmethod
    def snapshot_param_ids(model: nn.Module) -> set[int]:
        """在扩增前调用，获取所有参数的 id 集合作为快照。

        用法::

            original_ids = GrowthScheduler.snapshot_param_ids(model)
            expander.expand(model, config)
            scheduler.register_new_params(model, original_ids)
        """
        return snapshot_param_ids(model)
