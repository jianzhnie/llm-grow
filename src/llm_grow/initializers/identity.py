"""Identity initializer: zero o_proj / down_proj for function-preserving expansion."""
from __future__ import annotations

import torch.nn as nn


def zero_output_projections(
    block: nn.Module,
    attn_proj_names: list[str] | None = None,
    mlp_proj_names: list[str] | None = None,
) -> nn.Module:
    """将 block 内的 Attention 输出投影和 MLP 输出投影权重置零。

    置零后，block(x) = 0，残差连接保证 output = x + 0 = x，
    即严格的恒等映射（function-preserving）。

    Args:
        block: 待初始化的 Transformer block（含 self_attn 和 mlp 子模块）。
        attn_proj_names: Attention 输出投影的模块名列表（前缀匹配）。
        mlp_proj_names:  MLP 输出投影的模块名列表（前缀匹配）。

    Returns:
        原地修改后的 block。
    """
    if attn_proj_names is None:
        attn_proj_names = ["o_proj", "out_proj"]
    if mlp_proj_names is None:
        mlp_proj_names = ["down_proj", "fc2"]

    target_names = set(attn_proj_names) | set(mlp_proj_names)

    for name, module in block.named_modules():
        leaf_name = name.split(".")[-1]
        if leaf_name in target_names and isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return block


def is_identity_block(
    block: nn.Module,
    attn_proj_names: list[str] | None = None,
    mlp_proj_names: list[str] | None = None,
    atol: float = 1e-8,
) -> bool:
    """检查 block 是否已被初始化为恒等块（输出投影权重全零）。"""
    if attn_proj_names is None:
        attn_proj_names = ["o_proj", "out_proj"]
    if mlp_proj_names is None:
        mlp_proj_names = ["down_proj", "fc2"]

    target_names = set(attn_proj_names) | set(mlp_proj_names)
    found = False
    for name, module in block.named_modules():
        leaf_name = name.split(".")[-1]
        if leaf_name in target_names and isinstance(module, nn.Linear):
            found = True
            if module.weight.abs().max().item() > atol:
                return False
    return found
