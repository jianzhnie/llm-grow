"""OverlapCopy: Depth up-scaling via overlapping layer copy (arXiv:2312.15166, SOLAR DUS).

Core idea: duplicate the model, keep the upper portion (first N layers) and
lower portion (last N layers), and concatenate them.  The overlap region
ensures smooth distribution at the splice point.

.. warning::
    NOT function-preserving — requires ~100B+ tokens of continued pretraining.

Reference: Kim et al., "SOLAR 10.7B: Scaling Large Language Models
    with Simple yet Effective Depth Up-Scaling", arXiv:2312.15166, 2023.

Related:
    - ``ZeroBlockInsert`` (zero_block_insert.py): FP identity-block insertion
    - ``SVDInterpInsert`` (svd_interp_insert.py): SVD-based approximate FP expansion
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch.nn as nn

from llm_grow.configs.base import ModelExpansionConfig
from llm_grow.expanders.base import AbstractExpander
from llm_grow.expanders.registry import register_expander
from llm_grow.utils import (
    get_decoder_layers,
    set_decoder_layers,
    update_num_hidden_layers,
)
from llm_grow.utils.expansion_rules import build_overlap_sequence
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class OverlapCopyConfig(ModelExpansionConfig):
    num_overlap: int = 8
    """Number of overlapping layers.  Upper portion keeps the first
    ``L - num_overlap`` layers; lower portion starts from layer
    ``num_overlap``.  Total: ``len(upper) + len(lower) = 2*(L - num_overlap)``.
    """


@register_expander("overlap_copy")
class OverlapCopyExpander(AbstractExpander[OverlapCopyConfig]):
    """OverlapCopy depth up-scaling expander.

    .. warning::
        NOT function-preserving — ``verify()`` always returns ``False``.
        Requires substantial continued pretraining (~100B+ tokens recommended).
    """

    def expand(self, model: nn.Module, config: OverlapCopyConfig) -> nn.Module:
        layers = get_decoder_layers(model)
        num_layers = len(layers)
        overlap = config.num_overlap

        sequence = build_overlap_sequence(num_layers, overlap)

        new_layers = nn.ModuleList()
        for src_idx, _ in sequence:
            new_layers.append(copy.deepcopy(layers[src_idx]))

        set_decoder_layers(model, new_layers)
        update_num_hidden_layers(model, len(new_layers))
        return model

    def verify(self, original: nn.Module, expanded: nn.Module, **kwargs) -> bool:
        logger.info("OverlapCopy is NOT function-preserving — skipping output check.")
        return False
