"""Growth scheduler for MSG-style progressive mask unlocking."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


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
            import math

            return 0.5 * (1 - math.cos(math.pi * min(progress, 1.0)))
        if cfg.strategy == "step":
            thresholds = [0.25, 0.5, 0.75, 1.0]
            values = [0.25, 0.5, 0.75, 1.0]
            for t, v in zip(thresholds, values, strict=False):
                if progress < t:
                    return v - 0.25
            return 1.0
        raise ValueError(f"Unknown growth strategy: {cfg.strategy!r}")

    def apply_masks(self, model: nn.Module, unlock_ratio: float) -> None:
        """按 unlock_ratio 调整新增参数的 requires_grad 和梯度缩放。

        简化实现：unlock_ratio < 1 时对新增参数施加梯度缩放，
        完整实现应维护二进制掩码矩阵。
        """
        for _name, param in model.named_parameters():
            if getattr(param, "_is_new_growth", False):
                param.requires_grad_(unlock_ratio > 0)
                if hasattr(param, "_growth_scale"):
                    param._growth_scale = unlock_ratio

    def register_new_params(
        self, model: nn.Module, original_param_ids: set[int] | None = None
    ) -> int:
        """标记模型中的新增参数为"新增"（设置 ``_is_new_growth = True``）。

        使用参数 ``id()`` 快照来精确区分新旧参数：仅标记扩增后新增的参数，
        而非所有 ``requires_grad=True`` 的参数。

        Args:
            model: 扩增后的模型。
            original_param_ids: 扩增前通过 ``snapshot_param_ids(model)`` 获取的
                参数 id 集合。如果为 None，则退化为标记所有已设置
                ``_is_new_growth`` 的参数（即依赖扩增步骤中的标记）。

        Returns:
            被标记的新增参数元素数量。
        """
        count = 0
        for param in model.parameters():
            if original_param_ids is not None:
                if id(param) not in original_param_ids:
                    param._is_new_growth = True
                    count += param.numel()
            else:
                if getattr(param, "_is_new_growth", False):
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
        return {id(p) for p in model.parameters()}
