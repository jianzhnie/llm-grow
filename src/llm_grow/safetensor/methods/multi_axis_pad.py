"""MSG safetensor expander: depth + width masked structural growth.

Supports three orthogonal expansion axes:
  - num_new_layers      : identity block insertion (same as ZeroBlockInsert)
  - ffn_size_expansion   : zero-pad gate_proj/up_proj rows and down_proj cols
  - hidden_size_expansion: zero-pad all projections, embeddings, and lm_head

All are function-preserving: all new parameters start at zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_grow.configs.base import (
    BaseDepthConfig,
    BaseWidthConfig,
    BaseZeroSuffixConfig,
)
from llm_grow.safetensor.base import ExpansionPlan, SafetensorExpanderBase, TensorRecipe
from llm_grow.safetensor.utils import (
    ShardIndex,
    get_hidden_size_from_index,
    insert_positions,
    parse_layer_idx,
)
from llm_grow.utils.expansion_rules import compute_pad_deltas
from llm_grow.utils.insertion import build_layer_sequence


@dataclass
class MultiAxisPadSafetensorConfig(
    BaseDepthConfig, BaseWidthConfig, BaseZeroSuffixConfig
):
    """MSG-style safetensor configuration combining depth and width growth."""

    num_new_layers: int = 0
    """Number of identity blocks to insert (0 = depth disabled)."""


class MultiAxisPadSafetensorExpander(SafetensorExpanderBase):
    """MSG-style safetensor expander combining depth and FFN-width growth.

    Example — depth only::

        MultiAxisPadSafetensorExpander(MultiAxisPadSafetensorConfig(num_new_layers=4)).expand(...)

    Example — depth + wider FFN::

        MultiAxisPadSafetensorExpander(MultiAxisPadSafetensorConfig(
            num_new_layers=4, ffn_size_expansion=1024,
        )).expand(...)
    """

    def __init__(self, config: MultiAxisPadSafetensorConfig | None = None) -> None:
        self.config = config or MultiAxisPadSafetensorConfig()
        self.IDENTITY_ZERO_SUFFIXES = frozenset(self.config.zero_suffixes)

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        cfg = self.config
        num_orig = src_index.num_hidden_layers()
        wmap = src_index.weight_map
        suffixes = src_index.layer_suffixes()

        # ── depth: build layer sequence ──────────────────────────────────────
        if cfg.num_new_layers > 0:
            positions = set(
                insert_positions(num_orig, cfg.num_new_layers, cfg.insert_strategy)
            )
        else:
            positions = set()

        sequence = build_layer_sequence(num_orig, positions)

        plan = ExpansionPlan(
            new_num_hidden_layers=len(sequence),
            config_patches=self._build_config_patches(src_index),
        )

        # ── per-layer tensors ─────────────────────────────────────────────────
        for new_idx, (src_idx, is_identity) in enumerate(sequence):
            for suf in suffixes:
                src_key = f"model.layers.{src_idx}.{suf}"
                if src_key not in wmap:
                    continue
                new_key = f"model.layers.{new_idx}.{suf}"

                zero = is_identity and suf in self.IDENTITY_ZERO_SUFFIXES
                pad_r, pad_c = compute_pad_deltas(
                    suf,
                    ffn_size_expansion=cfg.ffn_size_expansion,
                    hidden_size_expansion=cfg.hidden_size_expansion,
                )

                plan.add(
                    new_key,
                    TensorRecipe(
                        src_shard=wmap[src_key],
                        src_key=src_key,
                        zero_out=zero,
                        pad_rows=pad_r,
                        pad_cols=pad_c,
                    ),
                )

        # ── non-layer tensors ─────────────────────────────────────────────────
        if cfg.hidden_size_expansion > 0:
            for key, shard in wmap.items():
                if parse_layer_idx(key) is not None:
                    continue
                pad_r = pad_c = 0
                if "embed_tokens" in key or "lm_head" in key:
                    pad_c = cfg.hidden_size_expansion
                elif "norm" in key and ".weight" in key:
                    pad_r = cfg.hidden_size_expansion

                if pad_r or pad_c:
                    plan.add(
                        key,
                        TensorRecipe(
                            src_shard=shard,
                            src_key=key,
                            pad_rows=pad_r,
                            pad_cols=pad_c,
                        ),
                    )
                else:
                    plan.passthrough(key, shard)
        else:
            self._passthrough_non_layer_keys(plan, wmap)

        return plan

    def _build_config_patches(self, src_index: ShardIndex) -> dict:
        cfg = self.config
        patches: dict = {}
        if cfg.ffn_size_expansion > 0:
            patches["intermediate_size"] = (
                _get_intermediate_size(src_index) + cfg.ffn_size_expansion
            )
        if cfg.hidden_size_expansion > 0:
            patches["hidden_size"] = (
                get_hidden_size_from_index(src_index) + cfg.hidden_size_expansion
            )
        return patches


def _get_intermediate_size(src_index: ShardIndex) -> int:
    """Infer intermediate_size from gate_proj shape in layer 0."""
    from llm_grow.safetensor.utils import read_safetensors_header

    for key in src_index.weight_map:
        if key.endswith("mlp.gate_proj.weight") and key.startswith("model.layers.0."):
            shard_path = src_index.model_dir / src_index.weight_map[key]
            header = read_safetensors_header(shard_path)
            if key in header:
                _dtype, shape = header[key]
                return shape[0]
    return 0
