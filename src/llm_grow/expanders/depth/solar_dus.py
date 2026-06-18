"""SOLAR DUS: Depth Up-Scaling via layer overlap-copy (arXiv:2312.15166).

核心思路：将原模型复制两份，取上段前 N 层与下段后 N 层拼接，
中间重叠区保证拼接点分布平滑。非 function-preserving，需要 100B+ CPT。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch.nn as nn

from llm_grow.expanders.base import AbstractExpander, ExpansionConfig
from llm_grow.expanders.depth.llama_pro import (
    _get_decoder_layers,
    _set_decoder_layers,
    _update_num_hidden_layers,
)
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SolarDUSConfig(ExpansionConfig):
    num_overlap: int = 8
    """重叠层数。上段保留前 (L - num_overlap) 层；下段从第 num_overlap 层开始。
    实际: len(upper) + len(lower) = 2*(L - num_overlap)
    """


class SolarDUSExpander(AbstractExpander):
    """SOLAR Depth Up-Scaling 扩增器。

    WARNING: 非 function-preserving，verify() 始终返回 False。
    扩增后需要大量 continued pretraining（建议 100B+ tokens）。
    """

    def expand(self, model: nn.Module, config: SolarDUSConfig) -> nn.Module:
        layers = _get_decoder_layers(model)
        num_layers = len(layers)
        overlap = config.num_overlap

        if overlap >= num_layers:
            raise ValueError(
                f"num_overlap ({overlap}) must be < num_layers ({num_layers})."
            )

        upper_end = num_layers - overlap
        lower_start = overlap

        upper = [copy.deepcopy(layers[i]) for i in range(upper_end)]
        lower = [copy.deepcopy(layers[i]) for i in range(lower_start, num_layers)]

        new_layers = nn.ModuleList(upper + lower)
        _set_decoder_layers(model, new_layers)
        _update_num_hidden_layers(model, len(new_layers))
        return model

    def verify(self, original: nn.Module, expanded: nn.Module, **kwargs) -> bool:
        logger.info("SOLAR DUS is NOT function-preserving — skipping output check.")
        return False
