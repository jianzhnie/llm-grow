"""OverlapCopy: 层重叠拼接深度扩增 (arXiv:2312.15166, SOLAR DUS).

核心思路：将原模型复制两份，取上段前 N 层与下段后 N 层拼接，
中间重叠区保证拼接点分布平滑。非 function-preserving，需要 100B+ CPT。

原始论文: Kim et al., "SOLAR 10.7B: Scaling Large Language Models
    with Simple yet Effective Depth Up-Scaling", arXiv:2312.15166, 2023.

Related:
    - ``ZeroBlockInsert`` (zero_block_insert.py): FP 恒等块嫁接
    - ``SVDInterpInsert`` (svd_interp_insert.py): SVD 插值近似 FP 扩增
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch.nn as nn

from llm_grow.configs.base import ModelExpansionConfig
from llm_grow.expanders.base import AbstractExpander
from llm_grow.utils import (
    get_decoder_layers,
    set_decoder_layers,
    update_num_hidden_layers,
)
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class OverlapCopyConfig(ModelExpansionConfig):
    num_overlap: int = 8
    """重叠层数。上段保留前 (L - num_overlap) 层；下段从第 num_overlap 层开始。
    实际: len(upper) + len(lower) = 2*(L - num_overlap)
    """


class OverlapCopyExpander(AbstractExpander):
    """OverlapCopy 层重叠拼接扩增器。

    WARNING: 非 function-preserving，verify() 始终返回 False。
    扩增后需要大量 continued pretraining（建议 100B+ tokens）。
    """

    def expand(self, model: nn.Module, config: OverlapCopyConfig) -> nn.Module:
        layers = get_decoder_layers(model)
        num_layers = len(layers)
        overlap = config.num_overlap

        if overlap >= num_layers:
            raise ValueError(
                f"num_overlap ({overlap}) must be < num_layers ({num_layers})."
            )

        upper_end = num_layers - overlap
        lower_start = overlap

        new_layers = nn.ModuleList()
        for i in range(upper_end):
            new_layers.append(copy.deepcopy(layers[i]))
        for i in range(lower_start, num_layers):
            new_layers.append(copy.deepcopy(layers[i]))

        set_decoder_layers(model, new_layers)
        update_num_hidden_layers(model, len(new_layers))
        return model

    def verify(self, original: nn.Module, expanded: nn.Module, **kwargs) -> bool:
        logger.info("OverlapCopy is NOT function-preserving — skipping output check.")
        return False
