"""SVD interpolation initializer for LESA-style layer prediction."""
from __future__ import annotations

import torch
import torch.nn as nn


def svd_features(
    weight: torch.Tensor,
    rank: int = 64,
) -> torch.Tensor:
    """对权重矩阵做截断 SVD，返回前 rank 个奇异值加权的右奇异向量拼接特征。

    Args:
        weight: 形状 (out, in) 的权重矩阵。
        rank:   保留的奇异值数量。

    Returns:
        形状 (rank * in,) 的特征向量。
    """
    w = weight.float()
    try:
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    except RuntimeError:
        U, S, Vh = torch.svd(w)
        Vh = Vh.T

    k = min(rank, S.shape[0])
    weighted = S[:k].unsqueeze(-1) * Vh[:k]
    return weighted.reshape(-1)


def interpolate_weights(
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """对两个形状相同的权重矩阵做线性插值。

    若形状不匹配，对齐到较小尺寸后插值。
    """
    if w_a.shape != w_b.shape:
        min_out = min(w_a.shape[0], w_b.shape[0])
        min_in = min(w_a.shape[1], w_b.shape[1])
        w_a = w_a[:min_out, :min_in]
        w_b = w_b[:min_out, :min_in]
    return alpha * w_a + (1.0 - alpha) * w_b


def init_layer_by_interpolation(
    new_layer: nn.Module,
    layer_a: nn.Module,
    layer_b: nn.Module,
    alpha: float = 0.5,
) -> nn.Module:
    """用 layer_a 和 layer_b 的参数插值初始化 new_layer（in-place）。

    new_layer 应已是 layer_a 的深拷贝。
    """
    params_a = dict(layer_a.named_parameters())
    params_b = dict(layer_b.named_parameters())

    with torch.no_grad():
        for name, param in new_layer.named_parameters():
            if name in params_b and params_a[name].shape == params_b[name].shape:
                interp = interpolate_weights(params_a[name], params_b[name], alpha)
                param.copy_(interp)
    return new_layer
