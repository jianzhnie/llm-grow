"""IdentityGraft: 恒等块嫁接深度扩增 (arXiv:2401.02415, LLaMA-Pro).

核心思路：在均匀间隔处插入恒等块（o_proj & down_proj 置零），
扩增后模型与原始模型函数完全一致（function-preserving）。

原始论文: Wu et al., "LLaMA Pro: Progressive LLaMA with Block Expansion",
    arXiv:2401.02415, 2024.

Related:
    - ``OverlapSplit`` (overlap_split.py): 非 FP 的层重叠拷贝深度扩增
    - ``InterpGraft`` (interp_graft.py): SVD 插值近似 FP 深度扩增
    - ``MultiAxisGrow`` (multi_axis_grow.py): 深度+宽度联合 FP 扩增
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch.nn as nn

from llm_grow.expanders.base import AbstractExpander, ExpansionConfig
from llm_grow.initializers.identity import zero_output_projections


@dataclass
class IdentityGraftConfig(ExpansionConfig):
    num_new_layers: int = 8
    """插入的新层数量。建议 = 原层数 // 4。
    向后兼容：也可使用 num_new_blocks 传参（等效别名）。
    """

    num_new_blocks: int | None = None
    """Deprecated alias for num_new_layers. 优先使用 num_new_layers。"""

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

    def __post_init__(self):
        if self.num_new_blocks is not None:
            self.num_new_layers = self.num_new_blocks
        self.num_new_blocks = self.num_new_layers


class IdentityGraftExpander(AbstractExpander):
    """IdentityGraft 恒等块插入扩增器。

    用法::

        from llm_grow import IdentityGraftExpander
        from llm_grow.expanders.depth.llama_pro import IdentityGraftConfig

        config = IdentityGraftConfig(num_new_layers=9)
        expander = IdentityGraftExpander()
        expanded_model = expander(original_model, config)
        expander.verify(original_model, expanded_model)
    """

    def expand(self, model: nn.Module, config: IdentityGraftConfig) -> nn.Module:
        layers = _get_decoder_layers(model)
        num_orig = len(layers)
        insert_positions = _compute_insert_positions(
            num_orig, config.num_new_layers, config.insert_strategy
        )

        new_layers = nn.ModuleList()
        insert_set = set(insert_positions)

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
    """创建恒等块（deep copy + zero projections），并标记所有参数为新增。"""
    block = copy.deepcopy(source_block)
    zero_output_projections(block, attn_proj_names, mlp_proj_names)
    for param in block.parameters():
        param._is_new_growth = True
    return block


def _compute_insert_positions(num_orig: int, num_new: int, strategy: str) -> list[int]:
    if num_new <= 0:
        return []
    if num_new > num_orig:
        raise ValueError(
            f"num_new_layers ({num_new}) cannot exceed num_orig_layers ({num_orig})."
        )
    if strategy == "uniform":
        step = num_orig / (num_new + 1)
        positions = sorted({round(step * (i + 1)) - 1 for i in range(num_new)})
        if len(positions) < num_new:
            import warnings

            warnings.warn(
                f"Uniform insertion produced {len(positions)} unique positions "
                f"(requested {num_new}). Consider reducing num_new_layers.",
                stacklevel=2,
            )
        return positions
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


# Backward-compatible aliases
LlamaProConfig = IdentityGraftConfig
LlamaProExpander = IdentityGraftExpander
