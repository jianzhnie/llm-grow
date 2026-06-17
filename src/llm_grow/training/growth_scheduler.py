"""Growth scheduler for MSG-style progressive mask unlocking."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class GrowthScheduleConfig:
    total_steps: int = 100_000
    """总训练步数。"""

    warmup_ratio: float = 0.3
    """前 warmup_ratio 比例的步数内保持原始参数空间（新维度 mask=0）。"""

    strategy: str = "linear"
    """解锁策略：'linear' | 'cosine' | 'step'。"""

    growth_dims: list[str] = field(
        default_factory=lambda: ["depth", "width", "ffn", "heads"]
    )
    """需要渐进解锁的维度名称列表。"""


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
        self._masks: dict[str, torch.Tensor] = {}

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
            for t, v in zip(thresholds, values):
                if progress < t:
                    return v - 0.25
            return 1.0
        raise ValueError(f"Unknown growth strategy: {cfg.strategy!r}")

    def apply_masks(self, model: nn.Module, unlock_ratio: float) -> None:
        """按 unlock_ratio 调整新增参数的 requires_grad 和梯度缩放。

        简化实现：unlock_ratio < 1 时对新增参数施加梯度缩放，
        完整实现应维护二进制掩码矩阵。
        """
        for name, param in model.named_parameters():
            if getattr(param, "_is_new_growth", False):
                param.requires_grad_(unlock_ratio > 0)
                if hasattr(param, "_growth_scale"):
                    param._growth_scale = unlock_ratio

    def register_new_params(self, model: nn.Module) -> int:
        """标记模型中所有 requires_grad=True 且未被冻结的参数为"新增"参数。

        应在扩增后、冻结原始参数前调用。
        """
        count = 0
        for param in model.parameters():
            if param.requires_grad:
                param._is_new_growth = True
                count += param.numel()
        return count
