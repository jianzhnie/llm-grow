"""ZeroBlockInsert: Identity-block depth expansion (arXiv:2401.02415, LLaMA-Pro).

Core idea: insert identity blocks at regular intervals (zeroing o_proj
and down_proj).  The expanded model is strictly function-preserving.

Reference: Wu et al., "LLaMA Pro: Progressive LLaMA with Block Expansion",
    arXiv:2401.02415, 2024.

Related:
    - ``OverlapCopy`` (overlap_copy.py): non-FP layer-copy depth expansion
    - ``SVDInterpInsert`` (svd_interp_insert.py): SVD-based approximate FP
    - ``MultiAxisPad`` (multi_axis_pad.py): combined depth+width FP expansion
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch.nn as nn

from llm_grow.configs.base import BaseDepthConfig
from llm_grow.configs.constants import ATTN_OUTPUT_PROJ_NAMES, MLP_OUTPUT_PROJ_NAMES
from llm_grow.expanders.base import AbstractExpander
from llm_grow.expanders.registry import register_expander
from llm_grow.initializers.identity import zero_output_projections
from llm_grow.utils import (
    get_decoder_layers,
    insert_positions,
    set_decoder_layers,
    update_num_hidden_layers,
)
from llm_grow.utils.insertion import NEW_GROWTH_ATTR


@dataclass
class ZeroBlockInsertConfig(BaseDepthConfig):
    """Configuration for ZeroBlockInsert identity-block expansion."""

    num_new_layers: int = 8
    """Number of new layers to insert.  Recommended: ``num_orig // 4``."""

    freeze_original: bool = True
    """If True, freeze original layers during phase-1 training."""

    attn_output_proj_names: list[str] = field(
        default_factory=lambda: list(ATTN_OUTPUT_PROJ_NAMES)
    )
    mlp_output_proj_names: list[str] = field(
        default_factory=lambda: list(MLP_OUTPUT_PROJ_NAMES)
    )


@register_expander("zero_block_insert")
class ZeroBlockInsertExpander(AbstractExpander[ZeroBlockInsertConfig]):
    """ZeroBlockInsert identity-block expander.

    Usage::

        from llm_grow import ZeroBlockInsertExpander
        from llm_grow.expanders.depth.zero_block_insert import ZeroBlockInsertConfig

        config = ZeroBlockInsertConfig(num_new_layers=9)
        expander = ZeroBlockInsertExpander()
        expanded_model = expander(original_model, config)
        expander.verify(original_model, expanded_model)
    """

    def expand(self, model: nn.Module, config: ZeroBlockInsertConfig) -> nn.Module:
        layers = get_decoder_layers(model)
        num_orig = len(layers)
        positions = insert_positions(
            num_orig, config.num_new_layers, config.insert_strategy
        )

        new_layers = nn.ModuleList()
        insert_set = set(positions)

        for i, layer in enumerate(layers):
            new_layers.append(layer)
            if i in insert_set:
                identity_block = _make_identity_block(
                    layer,
                    config.attn_output_proj_names,
                    config.mlp_output_proj_names,
                )
                if config.freeze_original:
                    for param in layer.parameters():
                        param.requires_grad_(False)
                new_layers.append(identity_block)

        set_decoder_layers(model, new_layers)
        update_num_hidden_layers(model, len(new_layers))
        return model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_identity_block(
    source_block: nn.Module,
    attn_proj_names: list[str],
    mlp_proj_names: list[str],
) -> nn.Module:
    """Create an identity block (deep copy + zero projections).  All parameters
    are marked as new growth.
    """
    block = copy.deepcopy(source_block)
    zero_output_projections(block, attn_proj_names, mlp_proj_names)
    for param in block.parameters():
        setattr(param, NEW_GROWTH_ATTR, True)
    return block
