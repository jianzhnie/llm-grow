"""Shared configuration base classes for memory-level and safetensor-level expanders.

These dataclasses hold parameters that appear in multiple expansion algorithms
across both abstraction layers, reducing duplication and making cross-layer
consistency easier to maintain.

Hierarchy
---------
::

    ExpansionConfig                  ← minimal base (extra dict only)
    ├── BaseDepthConfig              ← num_new_layers + insert_strategy
    │   ├── BaseMoEDepthConfig       ← + zero_suffixes for MoE identity blocks
    │   └── (ZeroBlockInsertConfig, SVDInterpInsertConfig, MultiAxisPadConfig, ...)
    ├── BaseWidthConfig              ← ffn_size_expansion + hidden_size_expansion
    └── ModelExpansionConfig         ← + model_name_or_path / output_dir
        ├── DenseToMoEConfig
        ├── ExpertCloneConfig
        └── OverlapCopyConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

InsertStrategy = Literal["uniform", "front", "rear"]
GrowthStrategy = Literal["linear", "cosine", "step"]

_VALID_INSERT_STRATEGIES: set[str] = {"uniform", "front", "rear"}
_VALID_GROWTH_STRATEGIES: set[str] = {"linear", "cosine", "step"}


@dataclass
class ExpansionConfig:
    """Minimal expansion config base class.

    Safetensor-level configs inherit directly from this — they don't need
    ``model_name_or_path`` / ``output_dir`` because paths are passed to
    ``expand(src_dir=..., dst_dir=...)``.
    """

    extra: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value store for extension / serialization."""

    def __post_init__(self) -> None:
        """No-op base hook; subclasses must call super().__post_init__()."""


@dataclass
class ModelExpansionConfig(ExpansionConfig):
    """Base for in-memory expander configs that reference a model path.

    Only in-memory expanders (those that load ``nn.Module``) need these
    fields.  Safetensor expanders receive paths via method arguments.
    """

    model_name_or_path: str = ""
    """HuggingFace model name or local path."""

    output_dir: str = "expanded_model"
    """Directory to save the expanded model."""


# ── Depth ─────────────────────────────────────────────────────────────────────


@dataclass
class BaseDepthConfig(ExpansionConfig):
    """Shared depth-expansion parameters.

    Used by both in-memory expanders (``ZeroBlockInsert``, ``SVDInterpInsert``,
    ``MultiAxisPad``) and safetensor expanders (``ZeroBlockInsertSafetensor``,
    ``GenericMoEDepthExpander``, ``LongcatDepthExpander``, ``MoEWidthExpander``).
    """

    num_new_layers: int = 4
    """Number of identity / interpolated layers to insert."""

    insert_strategy: InsertStrategy = "uniform"
    """Insertion strategy: ``'uniform'`` | ``'front'`` | ``'rear'``."""

    def __post_init__(self) -> None:
        if self.num_new_layers < 0:
            raise ValueError(f"num_new_layers must be >= 0, got {self.num_new_layers}")
        if self.insert_strategy not in _VALID_INSERT_STRATEGIES:
            raise ValueError(
                f"insert_strategy must be one of {_VALID_INSERT_STRATEGIES}, "
                f"got {self.insert_strategy!r}"
            )
        super().__post_init__()


# ── Zero-suffix (identity block) ─────────────────────────────────────────────


def _default_zero_suffixes() -> list[str]:
    """Lazy import to avoid a circular import with ``utils.expansion_rules``."""
    from llm_grow.utils.expansion_rules import build_identity_zero_suffixes

    return build_identity_zero_suffixes()


@dataclass
class BaseZeroSuffixConfig(ExpansionConfig):
    """Unified identity-block zero-suffix configuration.

    Any expander that inserts function-preserving identity blocks should use
    this to declare which tensor suffixes to zero.  The base class method
    ``_should_zero(suffix)`` on ``SafetensorExpanderBase`` reads these lists.
    """

    zero_suffixes: list[str] = field(default_factory=_default_zero_suffixes)
    """Layer suffixes that must be zeroed in identity blocks.

    Combines attention and MLP output projections into a single list.
    Override in subclasses for architecture-specific suffixes.
    """


# ── MoE Depth ────────────────────────────────────────────────────────────────


@dataclass
class BaseMoEDepthConfig(BaseDepthConfig):
    """Depth-expansion parameters for MoE safetensor expanders.

    MoE models need more granular control over which suffixes to zero because
    they have both per-expert projections and optional shared expert / dense
    layers.
    """

    zero_suffixes: list[str] = field(
        default_factory=lambda: ["self_attn.o_proj.weight"]
    )
    """Exact layer-suffixes to zero (attention + dense MLP projections)."""

    zero_expert_down: bool = True
    """Whether to zero all ``mlp.experts.*.down_proj.weight`` in identity blocks."""

    zero_shared_expert_down: bool = True
    """Whether to zero ``mlp.shared_experts.down_proj.weight`` in identity blocks."""

    def __post_init__(self) -> None:
        super().__post_init__()


# ── Width ─────────────────────────────────────────────────────────────────────


@dataclass
class BaseWidthConfig(ExpansionConfig):
    """Shared width-expansion parameters for safetensor expanders."""

    ffn_size_expansion: int = 0
    """Amount to increase ``intermediate_size`` / ``moe_intermediate_size``."""

    hidden_size_expansion: int = 0
    """Amount to increase ``hidden_size`` (d_model) globally."""

    def __post_init__(self) -> None:
        if self.ffn_size_expansion < 0:
            raise ValueError(
                f"ffn_size_expansion must be >= 0, got {self.ffn_size_expansion}"
            )
        if self.hidden_size_expansion < 0:
            raise ValueError(
                f"hidden_size_expansion must be >= 0, got {self.hidden_size_expansion}"
            )
        super().__post_init__()
