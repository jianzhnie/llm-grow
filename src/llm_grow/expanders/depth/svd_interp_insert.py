"""SVDInterpInsert: SVD 插值嫁接扩层 (arXiv:2502.13794, LESA).

核心思路：对相邻层权重矩阵做 SVD 分解，训练轻量预测网络预测插入层参数，
新层从"有意义"的初始化出发，收敛速度优于 OverlapCopy (DUS)。
即时精度约 80-90%（非严格 FP），CPT 需求量约为 OverlapCopy 的 50%。

原始论文: Yang et al., "LESA: Learnable LLM Layer Expansion with
    SVD-based Adaptation", arXiv:2502.13794, 2025.

Related:
    - ``ZeroBlockInsert`` (zero_block_insert.py): 严格 FP 恒等块嫁接
    - ``OverlapCopy`` (overlap_copy.py): 非 FP 层重叠拷贝
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from llm_grow.configs.base import BaseDepthConfig
from llm_grow.expanders.base import AbstractExpander
from llm_grow.utils import (
    get_decoder_layers,
    insert_positions,
    set_decoder_layers,
    update_num_hidden_layers,
)
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SVDInterpInsertConfig(BaseDepthConfig):
    num_new_layers: int = 0
    """新增层数。当 ``insert_between`` 为空且该值大于 0 时，
    使用均匀策略在相邻层之间插入 ``num_new_layers`` 个新层。
    """

    insert_between: list[tuple[int, int]] = field(default_factory=list)
    """显式指定在哪两层之间插入新层，格式 ``[(i, i+1), ...]``。
    若提供该列表，则 ``num_new_layers`` 被忽略。
    """

    svd_rank: int = 64
    """用于特征提取的 SVD 秩（越大信息越多，预测网络越慢）。"""

    predictor_hidden: int = 256
    """预测网络的隐藏维度。"""

    use_predictor: bool = False
    """False 时退化为相邻层线性插值（快速 baseline）；
    True 时使用 MLP 预测网络（需额外训练步骤）。"""


class SVDInterpInsertExpander(AbstractExpander):
    """SVDInterpInsert SVD 插值扩增器。

    当 use_predictor=False 时，使用相邻层参数的简单算术平均作为插入层初始化，
    可快速获得比 DUS 更好的初始精度。完整 LESA 实现（训练预测网络）请设置
    use_predictor=True 并先调用 train_predictor()。
    """

    def expand(self, model: nn.Module, config: SVDInterpInsertConfig) -> nn.Module:
        layers = get_decoder_layers(model)
        num_layers = len(layers)

        if config.insert_between:
            pairs = config.insert_between
        elif config.num_new_layers > 0:
            positions = insert_positions(
                num_layers, config.num_new_layers, config.insert_strategy
            )
            pairs = [(i, i + 1) for i in positions if i + 1 < num_layers]
        else:
            pairs = []

        if not pairs:
            logger.warning(
                "SVDInterpInsert: no insertion points specified. "
                "Set num_new_layers or insert_between to expand the model."
            )
            return model

        insert_after = sorted({p[0] for p in pairs})

        predictors = None
        if config.use_predictor:
            from llm_grow.initializers.svd_interp import train_predictor

            logger.info("Training LESA predictors (svd_rank=%d)...", config.svd_rank)
            predictors = train_predictor(
                layers,
                svd_rank=config.svd_rank,
                predictor_hidden=config.predictor_hidden,
            )

        new_layers: list[nn.Module] = []
        for i, layer in enumerate(layers):
            new_layers.append(layer)
            if i in insert_after and i + 1 < num_layers:
                if predictors is not None:
                    from llm_grow.initializers.svd_interp import predict_layer

                    interpolated = predict_layer(
                        layers[i],
                        layers[i + 1],
                        predictors,
                        svd_rank=config.svd_rank,
                    )
                else:
                    interpolated = _interpolate_layers(layers[i], layers[i + 1], config)
                new_layers.append(interpolated)

        layer_list = nn.ModuleList(new_layers)
        set_decoder_layers(model, layer_list)
        update_num_hidden_layers(model, len(layer_list))
        return model

    def verify(self, original: nn.Module, expanded: nn.Module, **kwargs) -> bool:
        logger.info(
            "SVDInterpInsert is approximately FP (~80-90%%)"
            " — running output diff check."
        )
        return super().verify(original, expanded, atol=0.5, **kwargs)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _interpolate_layers(
    layer_a: nn.Module,
    layer_b: nn.Module,
    config: SVDInterpInsertConfig,
    alpha: float = 0.5,
) -> nn.Module:
    """简单算术平均插值（use_predictor=False 的 baseline）。"""
    new_layer = copy.deepcopy(layer_a)
    state_a = dict(layer_a.named_parameters())
    state_b = dict(layer_b.named_parameters())

    with torch.no_grad():
        for name, param in new_layer.named_parameters():
            if name in state_b and state_a[name].shape == state_b[name].shape:
                param.copy_(alpha * state_a[name] + (1 - alpha) * state_b[name])
    return new_layer
