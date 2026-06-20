"""Shared configuration base classes for memory-level and safetensor-level expanders.

These dataclasses hold parameters that appear in multiple expansion algorithms
across both abstraction layers, reducing duplication and making cross-layer
consistency easier to maintain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExpansionConfig:
    """通用扩增配置基类，各方法可继承并扩展字段。"""

    model_name_or_path: str = ""
    output_dir: str = "expanded_model"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseDepthConfig(ExpansionConfig):
    """Shared depth-expansion parameters.

    Used by both in-memory expanders (``ZeroBlockInsert``, ``SVDInterpInsert``,
    ``MultiAxisPad``) and safetensor expanders (``ZeroBlockInsertSafetensor``,
    ``GenericMoEDepthExpander``, ``LongcatDepthExpander``, ``MoEWidthExpander``).
    """

    num_new_layers: int = 4
    """Number of identity / interpolated layers to insert."""

    insert_strategy: str = "uniform"
    """Insertion strategy: ``'uniform'`` | ``'front'`` | ``'rear'``."""


@dataclass
class BaseZeroSuffixConfig(ExpansionConfig):
    """Shared identity-block zero-suffix configuration for safetensor expanders."""

    attn_zero_suffixes: list[str] = field(
        default_factory=lambda: ["self_attn.o_proj.weight"]
    )
    """Layer suffixes for attention output projections that must be zeroed."""

    mlp_zero_suffixes: list[str] = field(
        default_factory=lambda: ["mlp.down_proj.weight"]
    )
    """Layer suffixes for MLP output projections that must be zeroed."""


@dataclass
class BaseMoEDepthConfig(BaseDepthConfig):
    """Shared depth-expansion parameters for MoE safetensor expanders."""

    extra_attn_zero_suffixes: list[str] = field(
        default_factory=lambda: ["self_attn.o_proj.weight"]
    )
    """Exact layer-suffixes for attention output projections to zero."""

    dense_mlp_zero_suffixes: list[str] = field(
        default_factory=lambda: ["mlp.down_proj.weight"]
    )
    """Exact layer-suffixes for dense MLP outputs to zero (non-MoE layers)."""

    zero_shared_expert_down: bool = True
    """Whether to zero ``mlp.shared_experts.down_proj.weight`` in identity blocks."""


@dataclass
class BaseWidthConfig(ExpansionConfig):
    """Shared width-expansion parameters for safetensor expanders."""

    ffn_size_expansion: int = 0
    """Amount to increase ``intermediate_size`` / ``moe_intermediate_size``."""

    hidden_size_expansion: int = 0
    """Amount to increase ``hidden_size`` (d_model) globally."""
