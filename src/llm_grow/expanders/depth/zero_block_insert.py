"""ZeroBlockInsert: 恒等块嫁接深度扩增 (arXiv:2401.02415, LLaMA-Pro).

核心思路：在均匀间隔处插入恒等块（o_proj & down_proj 置零），
扩增后模型与原始模型函数完全一致（function-preserving）。

原始论文: Wu et al., "LLaMA Pro: Progressive LLaMA with Block Expansion",
    arXiv:2401.02415, 2024.

Related:
    - ``OverlapCopy`` (overlap_copy.py): 非 FP 的层重叠拷贝深度扩增
    - ``SVDInterpInsert`` (svd_interp_insert.py): SVD 插值近似 FP 深度扩增
    - ``MultiAxisPad`` (multi_axis_pad.py): 深度+宽度联合 FP 扩增
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch.nn as nn

from llm_grow.configs.base import BaseDepthConfig
from llm_grow.configs.constants import ATTN_OUTPUT_PROJ_NAMES, MLP_OUTPUT_PROJ_NAMES
from llm_grow.expanders.base import AbstractExpander
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
    """ZeroBlockInsert 恒等块插入配置。"""

    num_new_layers: int = 8
    """插入的新层数量。建议 = 原层数 // 4。"""

    freeze_original: bool = True
    """Phase-1 训练时是否冻结原始块（仅训练新增块）。"""

    attn_output_proj_names: list[str] = field(
        default_factory=lambda: list(ATTN_OUTPUT_PROJ_NAMES)
    )
    mlp_output_proj_names: list[str] = field(
        default_factory=lambda: list(MLP_OUTPUT_PROJ_NAMES)
    )


class ZeroBlockInsertExpander(AbstractExpander[ZeroBlockInsertConfig]):
    """ZeroBlockInsert 恒等块插入扩增器。

    用法::

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
    """创建恒等块（deep copy + zero projections），并标记所有参数为新增。"""
    block = copy.deepcopy(source_block)
    zero_output_projections(block, attn_proj_names, mlp_proj_names)
    for param in block.parameters():
        setattr(param, NEW_GROWTH_ATTR, True)
    return block
