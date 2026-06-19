"""SVDInterpInsert safetensor expander: depth expansion via layer interpolation.

Inserts new layers initialized as α-weighted averages of adjacent layer pairs.
Approximately function-preserving (~80-90% initial accuracy retention).

Based on: Yang et al., "LESA: Learnable LLM Layer Expansion with
    SVD-based Adaptation", arXiv:2502.13794, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_grow.safetensor.base import ExpansionPlan, SafetensorExpanderBase, TensorRecipe
from llm_grow.safetensor.utils import ShardIndex


@dataclass
class SVDInterpInsertSafetensorConfig:
    insert_between: list[tuple[int, int]] = field(default_factory=list)
    """Layer pairs between which to insert. Empty = every adjacent pair."""

    interp_alpha: float = 0.5
    """Interpolation weight: output = alpha * layer_a + (1-alpha) * layer_b."""


class SVDInterpInsertSafetensorExpander(SafetensorExpanderBase):
    """Insert interpolated layers between adjacent layer pairs.

    Example::

        from llm_grow.safetensor.methods.svd_interp_insert import (
            SVDInterpInsertSafetensorConfig, SVDInterpInsertSafetensorExpander,
        )
        cfg = SVDInterpInsertSafetensorConfig(
            insert_between=[(i, i+1) for i in range(4)],
        )
        SVDInterpInsertSafetensorExpander(cfg).dry_run("path/to/model")
    """

    def __init__(self, config: SVDInterpInsertSafetensorConfig | None = None) -> None:
        self.config = config or SVDInterpInsertSafetensorConfig()

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        cfg = self.config
        wmap = src_index.weight_map
        suffixes = src_index.layer_suffixes()
        num_orig = src_index.num_hidden_layers()

        pairs = cfg.insert_between or [(i, i + 1) for i in range(num_orig - 1)]
        insert_after = sorted({p[0] for p in pairs if p[1] < num_orig})

        sequence: list[tuple[int, int | None]] = []
        for i in range(num_orig):
            sequence.append((i, None))
            if i in insert_after:
                sequence.append((i, i + 1))

        plan = ExpansionPlan(new_num_hidden_layers=len(sequence))

        for new_idx, (src_a, src_b) in enumerate(sequence):
            for suf in suffixes:
                key_a = f"model.layers.{src_a}.{suf}"
                if key_a not in wmap:
                    continue
                new_key = f"model.layers.{new_idx}.{suf}"

                if src_b is not None:
                    key_b = f"model.layers.{src_b}.{suf}"
                    if key_b in wmap:
                        plan.add(
                            new_key,
                            TensorRecipe(
                                src_shard=wmap[key_a],
                                src_key=key_a,
                                interp_src_shard=wmap[key_b],
                                interp_src_key=key_b,
                                interp_alpha=cfg.interp_alpha,
                            ),
                        )
                    else:
                        plan.passthrough(key_a, wmap[key_a])
                else:
                    plan.add(
                        new_key,
                        TensorRecipe(src_shard=wmap[key_a], src_key=key_a),
                    )

        self._passthrough_non_layer_keys(plan, wmap)
        return plan
