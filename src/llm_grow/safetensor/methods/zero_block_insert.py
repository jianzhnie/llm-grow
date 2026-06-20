"""LLaMA-Pro safetensor expander: identity block insertion (arXiv:2401.02415).

No model loading required.  Operates entirely on .safetensors files.
Function-preserving: new blocks have o_proj & down_proj zeroed → Block(x) = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_grow.configs.base import (
    BaseDepthConfig,
    BaseZeroSuffixConfig,
)
from llm_grow.safetensor.base import ExpansionPlan, SafetensorExpanderBase
from llm_grow.safetensor.utils import ShardIndex, insert_positions
from llm_grow.utils.insertion import build_layer_sequence


@dataclass
class ZeroBlockInsertSafetensorConfig(BaseDepthConfig, BaseZeroSuffixConfig):
    """ZeroBlockInsert safetensor 配置。"""

    num_new_layers: int = 8
    """Number of identity blocks to insert."""


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
        self.IDENTITY_ZERO_SUFFIXES = frozenset(self.config.zero_suffixes)

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        num_orig = src_index.num_hidden_layers()
        cfg = self.config

        positions = insert_positions(num_orig, cfg.num_new_layers, cfg.insert_strategy)
        pos_set = set(positions)

        sequence = build_layer_sequence(num_orig, pos_set)

        return self._build_layer_plan(src_index, layer_sequence=sequence)
