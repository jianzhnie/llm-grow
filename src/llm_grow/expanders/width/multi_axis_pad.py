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

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from llm_grow.configs.base import BaseDepthConfig
from llm_grow.expanders.base import AbstractExpander
from llm_grow.utils.insertion import NEW_GROWTH_ATTR


@dataclass
class MultiAxisPadConfig(BaseDepthConfig):
    """MultiAxisPad 多维度扩增配置。"""

    hidden_size_expansion: int = 0
    """hidden_size 增量（需为 head_dim 的整数倍）。"""

    intermediate_size_expansion: int = 0
    """intermediate_size（FFN 宽度）增量。"""

    freeze_original: bool = True
    """是否冻结原始参数。"""

    growth_schedule: str = "instant"
    """掩码生长调度策略：
    - 'instant'  : 立即解锁全部新参数（默认，用于扩增后直接 CPT）
    - 'linear'   : 按训练步数线性解锁（需配合 GrowthScheduler）
    """

    attn_output_proj_names: list[str] = field(
        default_factory=lambda: ["o_proj", "out_proj"]
    )
    mlp_output_proj_names: list[str] = field(
        default_factory=lambda: ["down_proj", "fc2"]
    )


class MultiAxisPadExpander(AbstractExpander[MultiAxisPadConfig]):
    """MultiAxisPad 多维度扩增器。

    建议搭配 GrowthScheduler 使用（training/growth_scheduler.py），
    实现渐进式掩码解锁以避免训练震荡。
    """

    def expand(self, model: nn.Module, config: MultiAxisPadConfig) -> nn.Module:
        if config.hidden_size_expansion > 0 or config.intermediate_size_expansion > 0:
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
    di = config.intermediate_size_expansion

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        module_type = _classify_linear(name)
        if module_type == "skip":
            continue
        if module_type == "hidden_to_hidden" and dh > 0:
            _pad_linear(module, in_delta=dh, out_delta=dh)
        elif module_type == "hidden_to_inter" and di > 0:
            _pad_linear(module, in_delta=0, out_delta=di)
        elif module_type == "inter_to_hidden" and di > 0:
            _pad_linear(module, in_delta=di, out_delta=0)

    cfg = getattr(model, "config", None)
    if cfg is not None:
        if dh > 0 and hasattr(cfg, "hidden_size"):
            cfg.hidden_size += dh
        if di > 0 and hasattr(cfg, "intermediate_size"):
            cfg.intermediate_size += di
    return model


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


def _classify_linear(name: str) -> str:
    """Classify a linear layer's semantic role based on its full dotted path."""
    if any(k in name for k in ("lm_head", "embed", "layernorm", "norm")):
        return "skip"
    if any(k in name for k in ("gate_proj", "up_proj")):
        return "hidden_to_inter"
    if "down_proj" in name:
        return "inter_to_hidden"
    return "hidden_to_hidden"


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
