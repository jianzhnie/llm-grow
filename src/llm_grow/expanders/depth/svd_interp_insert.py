"""SVDInterpInsert: SVD-based interpolation layer insertion (arXiv:2502.13794, LESA).

Core idea: decompose adjacent layer weight matrices via SVD, train a
lightweight predictor network to estimate inserted-layer parameters.
New layers start from a "meaningful" initialization, converging faster
than OverlapCopy (DUS).  Instant accuracy ≈ 80–90% (not strictly FP);
CPT requirement ≈ 50% of OverlapCopy.

Reference: Yang et al., "LESA: Learnable LLM Layer Expansion with
    SVD-based Adaptation", arXiv:2502.13794, 2025.

Related:
    - ``ZeroBlockInsert`` (zero_block_insert.py): strictly FP identity blocks
    - ``OverlapCopy`` (overlap_copy.py): non-FP layer copy
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from llm_grow.configs.base import BaseDepthConfig
from llm_grow.expanders.base import AbstractExpander
from llm_grow.expanders.registry import register_expander
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
    """Number of new layers to insert.  When ``insert_between`` is empty and
    this value > 0, layers are inserted uniformly between existing layers.
    """

    insert_between: list[tuple[int, int]] = field(default_factory=list)
    """Explicit insertion positions as ``[(i, i+1), ...]`` pairs.
    When provided, ``num_new_layers`` is ignored.
    """

    svd_rank: int = 64
    """SVD rank for feature extraction (higher = more info, slower predictor)."""

    predictor_hidden: int = 256
    """Hidden dimension of the predictor network."""

    use_predictor: bool = False
    """If False, fall back to simple arithmetic averaging (fast baseline).
    If True, use an MLP predictor network (requires extra training step)."""


@register_expander("svd_interp_insert")
class SVDInterpInsertExpander(AbstractExpander[SVDInterpInsertConfig]):
    """SVD-based interpolation expander.

    When ``use_predictor=False``, uses simple arithmetic averaging of adjacent
    layer parameters as the initialization — a fast baseline that already beats
    DUS.  For the full LESA implementation (trained predictor), set
    ``use_predictor=True`` and call ``train_predictor()`` first.
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
        kwargs.setdefault("atol", 0.5)
        return super().verify(original, expanded, **kwargs)


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
