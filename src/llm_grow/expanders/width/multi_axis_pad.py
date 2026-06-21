"""MultiAxisPad: 多维掩码生长扩增 (arXiv:2305.02869, MSG).

支持四维度任意组合扩增：
  - 深度（+层数）：恒等块插入（复用 ZeroBlockInsert）
  - 宽度（+hidden_size）：零填充 + 掩码
  - 注意力头数（+num_heads）：o_proj=0
  - FFN 宽度（+intermediate_size）：零填充 + 掩码

Function-preserving：✓（所有新维度初始为零，不影响前向输出）。

原始论文: Du et al., "Stacking More Layers Differently: High-Rank Training
    Through Low-Rank Updates", arXiv:2305.02869, 2023.

Related:
    - ``ZeroBlockInsert`` (zero_block_insert.py): 仅深度的 FP 扩增
    - ``DenseToMoE`` (dense_to_moe.py): Dense 转稀疏 MoE
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

from llm_grow.configs.base import BaseDepthConfig, BaseWidthConfig
from llm_grow.expanders.base import AbstractExpander
from llm_grow.utils.expansion_rules import compute_pad_deltas
from llm_grow.utils.insertion import NEW_GROWTH_ATTR

_GrowthSchedule = Literal["instant", "linear", "cosine", "step"]


@dataclass
class MultiAxisPadConfig(BaseDepthConfig, BaseWidthConfig):
    """MultiAxisPad 多维度扩增配置。"""

    intermediate_size_expansion: int | None = None
    """Deprecated alias for ``ffn_size_expansion``."""

    freeze_original: bool = True
    """是否冻结原始参数。"""

    growth_schedule: _GrowthSchedule = "instant"
    """掩码生长调度策略：
    - 'instant'  : 立即解锁全部新参数（默认，用于扩增后直接 CPT）
    - 'linear'   : 按训练步数线性解锁（需配合 GrowthScheduler）
    - 'cosine'   : 余弦解锁
    - 'step'     : 分阶段解锁
    """

    attn_output_proj_names: list[str] = field(
        default_factory=lambda: ["o_proj", "out_proj"]
    )
    mlp_output_proj_names: list[str] = field(
        default_factory=lambda: ["down_proj", "fc2"]
    )

    def __post_init__(self) -> None:
        if self.intermediate_size_expansion is not None:
            warnings.warn(
                "intermediate_size_expansion is deprecated; use ffn_size_expansion",
                DeprecationWarning,
                stacklevel=2,
            )
            self.ffn_size_expansion = self.intermediate_size_expansion
        self.intermediate_size_expansion = self.ffn_size_expansion
        super().__post_init__()

        if not self.attn_output_proj_names or not all(
            isinstance(n, str) and n for n in self.attn_output_proj_names
        ):
            raise ValueError(
                "attn_output_proj_names must be a non-empty list of non-empty strings"
            )
        if not self.mlp_output_proj_names or not all(
            isinstance(n, str) and n for n in self.mlp_output_proj_names
        ):
            raise ValueError(
                "mlp_output_proj_names must be a non-empty list of non-empty strings"
            )

        if self.growth_schedule not in ("instant", "linear", "cosine", "step"):
            raise ValueError(
                f"growth_schedule must be one of instant/linear/cosine/step, "
                f"got {self.growth_schedule!r}"
            )


class MultiAxisPadExpander(AbstractExpander[MultiAxisPadConfig]):
    """MultiAxisPad 多维度扩增器。

    建议搭配 GrowthScheduler 使用（training/growth_scheduler.py），
    实现渐进式掩码解锁以避免训练震荡。
    """

    def expand(self, model: nn.Module, config: MultiAxisPadConfig) -> nn.Module:
        if config.hidden_size_expansion > 0 or config.ffn_size_expansion > 0:
            model = _expand_width(model, config)

        if config.num_new_layers > 0:
            model = _expand_depth(model, config)

        if config.freeze_original:
            _freeze_original_params(model)

        return model


# ---------------------------------------------------------------------------
# width expansion
# ---------------------------------------------------------------------------


def _expand_width(model: nn.Module, config: MultiAxisPadConfig) -> nn.Module:
    """对所有线性层做零填充宽度扩展。"""
    dh = config.hidden_size_expansion
    di = config.ffn_size_expansion

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Derive the layer-relative suffix (e.g. "self_attn.q_proj.weight")
        # so it can be classified by the same rules used by safetensor expanders.
        suffix = _layer_relative_suffix(name)
        if suffix is None:
            continue
        pad_r, pad_c = compute_pad_deltas(
            suffix,
            ffn_size_expansion=di,
            hidden_size_expansion=dh,
        )
        if pad_r == 0 and pad_c == 0:
            continue
        _pad_linear(module, in_delta=pad_c, out_delta=pad_r)

    cfg = getattr(model, "config", None)
    if cfg is not None:
        if dh > 0 and hasattr(cfg, "hidden_size"):
            cfg.hidden_size += dh
        if di > 0 and hasattr(cfg, "intermediate_size"):
            cfg.intermediate_size += di
    return model


def _layer_relative_suffix(full_name: str) -> str | None:
    """Return the layer-relative suffix for a module name, or None."""
    # Module names look like "layers.0.self_attn.q_proj" or "lm_head".
    # We want "self_attn.q_proj" (without the leading layer path).
    parts = full_name.split(".")
    if len(parts) < 2:
        return None
    # Heuristic: drop the "layers.N" or "transformer.h.N" prefix.
    if parts[0] in ("layers", "transformer", "decoder") and parts[1].isdigit():
        return ".".join(parts[2:])
    return None


def _pad_linear(module: nn.Linear, in_delta: int, out_delta: int) -> None:
    """在行（out）和列（in）方向零填充权重矩阵，保持 function-preserving。

    新创建的参数会被标记 ``_is_new_growth = True``，以便后续 freeze 机制识别。
    """
    old_w = module.weight.data
    old_out, old_in = old_w.shape

    new_out = old_out + out_delta
    new_in = old_in + in_delta
    new_w = torch.zeros(new_out, new_in, dtype=old_w.dtype, device=old_w.device)
    new_w[:old_out, :old_in] = old_w

    new_param = nn.Parameter(new_w)
    setattr(new_param, NEW_GROWTH_ATTR, True)
    module.weight = new_param
    module.out_features = new_out
    module.in_features = new_in

    if module.bias is not None:
        old_b = module.bias.data
        new_b = torch.zeros(new_out, dtype=old_b.dtype, device=old_b.device)
        new_b[:old_out] = old_b
        new_bias_param = nn.Parameter(new_b)
        setattr(new_bias_param, NEW_GROWTH_ATTR, True)
        module.bias = new_bias_param


# ---------------------------------------------------------------------------
# depth expansion
# ---------------------------------------------------------------------------


def _expand_depth(model: nn.Module, config: MultiAxisPadConfig) -> nn.Module:
    from llm_grow.expanders.depth.zero_block_insert import (
        ZeroBlockInsertConfig,
        ZeroBlockInsertExpander,
    )

    sub_config = ZeroBlockInsertConfig(
        num_new_layers=config.num_new_layers,
        insert_strategy="uniform",
        freeze_original=False,
        attn_output_proj_names=config.attn_output_proj_names,
        mlp_output_proj_names=config.mlp_output_proj_names,
    )
    return ZeroBlockInsertExpander().expand(model, sub_config)


# ---------------------------------------------------------------------------
# freeze helpers
# ---------------------------------------------------------------------------


def _freeze_original_params(model: nn.Module) -> None:
    """冻结所有被标记为原始参数的层（未被 zero_output_projections 触碰的层）。"""
    for param in model.parameters():
        if not getattr(param, NEW_GROWTH_ATTR, False):
            param.requires_grad_(False)
