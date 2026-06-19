"""LLaMA-Pro safetensor expander: identity block insertion (arXiv:2401.02415).

No model loading required.  Operates entirely on .safetensors files.
Function-preserving: new blocks have o_proj & down_proj zeroed → Block(x) = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_grow.safetensor.base import ExpansionPlan, SafetensorExpanderBase
from llm_grow.safetensor.utils import ShardIndex, insert_positions


@dataclass
class ZeroBlockInsertSafetensorConfig:
    num_new_layers: int = 8
    """Number of identity blocks to insert."""

    num_new_blocks: int | None = None
    """Deprecated alias for num_new_layers."""

    insert_strategy: str = "uniform"
    """'uniform' | 'front' | 'rear'"""

    attn_zero_suffixes: list[str] = field(
        default_factory=lambda: ["self_attn.o_proj.weight"]
    )
    mlp_zero_suffixes: list[str] = field(
        default_factory=lambda: ["mlp.down_proj.weight"]
    )

    def __post_init__(self):
        if self.num_new_blocks is not None:
            self.num_new_layers = self.num_new_blocks
        self.num_new_blocks = self.num_new_layers


class ZeroBlockInsertSafetensorExpander(SafetensorExpanderBase):
    """Insert identity blocks directly into safetensor weight files.

    Example::

        from llm_grow.safetensor.zero_block_insert import (
            ZeroBlockInsertSafetensorConfig, ZeroBlockInsertSafetensorExpander,
        )
        cfg = ZeroBlockInsertSafetensorConfig(num_new_layers=7)
        ZeroBlockInsertSafetensorExpander(cfg).expand(
            src_dir="Qwen/Qwen3-8B",
            dst_dir="./outputs/qwen3_zero_block_insert",
        )
    """

    def __init__(self, config: ZeroBlockInsertSafetensorConfig | None = None) -> None:
        self.config = config or ZeroBlockInsertSafetensorConfig()
        # Merge user-specified zero suffixes into base set
        self.IDENTITY_ZERO_SUFFIXES = frozenset(
            self.config.attn_zero_suffixes + self.config.mlp_zero_suffixes
        )

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        num_orig = src_index.num_hidden_layers()
        cfg = self.config

        positions = insert_positions(num_orig, cfg.num_new_layers, cfg.insert_strategy)
        pos_set = set(positions)

        # Build ordered layer sequence: (src_orig_idx, is_identity)
        sequence: list[tuple[int, bool]] = []
        for i in range(num_orig):
            sequence.append((i, False))
            if i in pos_set:
                sequence.append((i, True))  # identity copy of layer i

        return self._build_layer_plan(src_index, layer_sequence=sequence)

