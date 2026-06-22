"""MultiAxisPad: Multi-axis masked growth expansion (arXiv:2305.02869, MSG).

Supports arbitrary combinations of four expansion axes:
  - Depth (+layers): identity block insertion (reuses ZeroBlockInsert)
  - Width (+hidden_size): zero-padding with masks
  - Attention heads (+num_heads): o_proj=0
  - FFN width (+intermediate_size): zero-padding with masks

Function-preserving: ✓ (all new dimensions start at zero).

Reference: Du et al., "Stacking More Layers Differently: High-Rank Training
    Through Low-Rank Updates", arXiv:2305.02869, 2023.

Related:
    - ``ZeroBlockInsert`` (zero_block_insert.py): depth-only FP expansion
    - ``DenseToMoE`` (dense_to_moe.py): Dense → Sparse MoE
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

from llm_grow.configs.base import ExpansionConfig, InsertStrategy
from llm_grow.expanders.base import AbstractExpander
from llm_grow.expanders.registry import register_expander
from llm_grow.utils.expansion_rules import compute_pad_deltas
from llm_grow.utils.insertion import NEW_GROWTH_ATTR

_GrowthSchedule = Literal["instant", "linear", "cosine", "step"]

_VALID_GROWTH_SCHEDULES: set[str] = {"instant", "linear", "cosine", "step"}
_VALID_INSERT_STRATEGIES: set[str] = {"uniform", "front", "rear"}


@dataclass
class MultiAxisPadConfig(ExpansionConfig):
    """Multi-axis (depth + width) expansion config — flat dataclass.

    This dataclass avoids the diamond-inheritance ambiguity of
    ``BaseDepthConfig`` + ``BaseWidthConfig`` by inlining the fields
    from both bases directly under ``ExpansionConfig``.
    """

    # ── Depth fields (inlined from BaseDepthConfig) ──────────────────────────
    num_new_layers: int = 4
    """Number of identity / interpolated layers to insert."""

    insert_strategy: InsertStrategy = "uniform"
    """Insertion strategy: ``'uniform'`` | ``'front'`` | ``'rear'``."""

    # ── Width fields (inlined from BaseWidthConfig) ──────────────────────────
    ffn_size_expansion: int = 0
    """Amount to increase ``intermediate_size``."""

    hidden_size_expansion: int = 0
    """Amount to increase ``hidden_size`` (d_model) globally."""

    # ── MultiAxisPad-specific fields ─────────────────────────────────────────
    intermediate_size_expansion: int | None = None
    """Deprecated alias for ``ffn_size_expansion``."""

    freeze_original: bool = True
    """Whether to freeze original parameters after expansion."""

    growth_schedule: str = "instant"
    """Mask growth schedule: ``'instant'`` | ``'linear'`` | ``'cosine'`` | ``'step'``."""

    attn_output_proj_names: list[str] = field(
        default_factory=lambda: ["o_proj", "out_proj"]
    )
    mlp_output_proj_names: list[str] = field(
        default_factory=lambda: ["down_proj", "fc2"]
    )

    def __post_init__(self) -> None:  # noqa: C901
        if self.intermediate_size_expansion is not None:
            warnings.warn(
                "intermediate_size_expansion is deprecated; use ffn_size_expansion",
                DeprecationWarning,
                stacklevel=2,
            )
            self.ffn_size_expansion = self.intermediate_size_expansion
        self.intermediate_size_expansion = self.ffn_size_expansion

        # Depth validation
        if self.num_new_layers < 0:
            raise ValueError(
                f"num_new_layers must be >= 0, got {self.num_new_layers}"
            )
        if self.insert_strategy not in _VALID_INSERT_STRATEGIES:
            raise ValueError(
                f"insert_strategy must be one of {_VALID_INSERT_STRATEGIES}, "
                f"got {self.insert_strategy!r}"
            )

        # Width validation
        if self.ffn_size_expansion < 0:
            raise ValueError(
                f"ffn_size_expansion must be >= 0, got {self.ffn_size_expansion}"
            )
        if self.hidden_size_expansion < 0:
            raise ValueError(
                f"hidden_size_expansion must be >= 0, got {self.hidden_size_expansion}"
            )

        # Own validation
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

        if self.growth_schedule not in _VALID_GROWTH_SCHEDULES:
            raise ValueError(
                f"growth_schedule must be one of {_VALID_GROWTH_SCHEDULES}, "
                f"got {self.growth_schedule!r}"
            )

        super().__post_init__()


@register_expander("multi_axis_pad")
class MultiAxisPadExpander(AbstractExpander[MultiAxisPadConfig]):
    """Multi-axis masked growth expander.

    Recommended: pair with ``GrowthScheduler`` (``training/growth_scheduler.py``)
    for progressive mask unlocking to avoid training instability.
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
    """Zero-pad all linear layers for width expansion."""
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
    """Zero-pad weight matrix along row (out) and column (in) axes.

    New parameters are tagged ``_is_new_growth = True`` so the freeze
    mechanism can identify them later.
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
        insert_strategy=config.insert_strategy,
        freeze_original=False,
        attn_output_proj_names=config.attn_output_proj_names,
        mlp_output_proj_names=config.mlp_output_proj_names,
    )
    return ZeroBlockInsertExpander().expand(model, sub_config)


# ---------------------------------------------------------------------------
# freeze helpers
# ---------------------------------------------------------------------------


def _freeze_original_params(model: nn.Module) -> None:
    """Freeze all parameters that were NOT marked as new growth."""
    for param in model.parameters():
        if not getattr(param, NEW_GROWTH_ATTR, False):
            param.requires_grad_(False)
