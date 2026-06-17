"""LLaMA-Pro: Progressive LLaMA with Block Expansion (arXiv:2401.02415).

核心思路：在均匀间隔处插入恒等块（o_proj & down_proj 置零），
扩增后模型与原始模型函数完全一致（function-preserving）。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch.nn as nn

from llm_grow.expanders.base import AbstractExpander, ExpansionConfig
from llm_grow.initializers.identity import zero_output_projections


@dataclass
class LlamaProConfig(ExpansionConfig):
    num_new_blocks: int = 8
    """插入的新块数量。建议 = 原层数 // 4。"""

    insert_strategy: str = "uniform"
    """插入策略：
    - 'uniform'  : 均匀分布（论文默认，效果最好）
    - 'front'    : 集中在前端
    - 'rear'     : 集中在后端
    """

    freeze_original: bool = True
    """Phase-1 训练时是否冻结原始块（仅训练新增块）。"""

    attn_output_proj_names: list[str] = field(
        default_factory=lambda: ["o_proj", "out_proj"]
    )
    mlp_output_proj_names: list[str] = field(
        default_factory=lambda: ["down_proj", "fc2"]
    )


class LlamaProExpander(AbstractExpander):
    """恒等块插入扩增器。

    用法::

        from llm_grow import LlamaProExpander
        from llm_grow.expanders.depth.llama_pro import LlamaProConfig

        config = LlamaProConfig(num_new_blocks=9)
        expander = LlamaProExpander()
        expanded_model = expander(original_model, config)
        expander.verify(original_model, expanded_model)
    """

    def expand(self, model: nn.Module, config: LlamaProConfig) -> nn.Module:
        layers = _get_decoder_layers(model)
        num_orig = len(layers)
        insert_positions = _compute_insert_positions(
            num_orig, config.num_new_blocks, config.insert_strategy
        )

        new_layers = nn.ModuleList()
        insert_set = set(insert_positions)
        insert_idx = 0

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
                insert_idx += 1

        _set_decoder_layers(model, new_layers)
        _update_num_hidden_layers(model, len(new_layers))
        return model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_identity_block(
    source_block: nn.Module,
    attn_proj_names: list[str],
    mlp_proj_names: list[str],
) -> nn.Module:
    block = copy.deepcopy(source_block)
    zero_output_projections(block, attn_proj_names, mlp_proj_names)
    return block


def _compute_insert_positions(
    num_orig: int, num_new: int, strategy: str
) -> list[int]:
    if strategy == "uniform":
        step = num_orig / (num_new + 1)
        return sorted(set(int(round(step * (i + 1))) - 1 for i in range(num_new)))
    if strategy == "front":
        return list(range(num_new))
    if strategy == "rear":
        return list(range(num_orig - num_new, num_orig))
    raise ValueError(f"Unknown insert_strategy: {strategy!r}")


def _get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    for attr in ("layers", "model.layers", "transformer.h", "decoder.layers"):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return obj
    raise AttributeError("Cannot locate decoder layer list in model.")


def _set_decoder_layers(model: nn.Module, new_layers: nn.ModuleList) -> None:
    for attr in ("layers", "model.layers", "transformer.h", "decoder.layers"):
        parts = attr.split(".")
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], new_layers)
            return
    raise AttributeError("Cannot set decoder layer list in model.")


def _update_num_hidden_layers(model: nn.Module, new_num: int) -> None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, new_num)
            break
