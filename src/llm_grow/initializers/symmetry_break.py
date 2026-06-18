"""Symmetry-breaking initializers for MoE expert copies."""

from __future__ import annotations

import torch
import torch.nn as nn


def add_noise_to_experts(
    experts: nn.ModuleList,
    std: float = 0.01,
    skip_first: bool = True,
) -> None:
    """向每个专家的全部参数添加高斯噪声，打破副本间的对称性。

    Args:
        experts:    专家模块列表。
        std:        噪声标准差（相对于参数值量级，建议 0.001 ~ 0.02）。
        skip_first: 是否跳过第一个专家（保留原始权重作为"锚点"）。
    """
    start = 1 if skip_first else 0
    for expert in experts[start:]:
        with torch.no_grad():
            for param in expert.parameters():
                param.add_(torch.randn_like(param) * std)


def drop_upcycling(
    experts: nn.ModuleList,
    drop_ratio: float = 0.1,
    skip_first: bool = True,
) -> None:
    """Drop-Upcycling：随机将专家参数的一部分置零，强迫副本专业化。

    比加噪更激进，有助于专家快速分化，但初始精度损失更大。

    Args:
        experts:    专家模块列表。
        drop_ratio: 置零比例（建议 0.05 ~ 0.2）。
        skip_first: 是否跳过第一个专家。
    """
    start = 1 if skip_first else 0
    for expert in experts[start:]:
        with torch.no_grad():
            for param in expert.parameters():
                mask = torch.rand_like(param) > drop_ratio
                param.mul_(mask.float())


def router_noise_init(router: nn.Linear, std: float = 0.001) -> None:
    """对 Router 权重矩阵加小噪声，防止所有 token 路由到同一专家。"""
    with torch.no_grad():
        router.weight.add_(torch.randn_like(router.weight) * std)


def cluster_aware_upcycling(
    experts: nn.ModuleList,
    cluster_assignments: list[int],
    skip_first: bool = True,
) -> None:
    """Cluster-Aware Upcycling（占位实现，arXiv:2604.13508）。

    完整实现需要先对训练数据做 token 聚类，再让不同副本在不同
    cluster 的子集上微调。此处仅做接口预留。

    .. warning::
        此函数尚未实现，调用将抛出 NotImplementedError。

    Args:
        experts:             专家模块列表。
        cluster_assignments: 每个副本分配的 cluster id 列表。
        skip_first:          是否跳过第一个专家。
    """
    # TODO: 实现 Cluster-Aware Upcycling（需要外部 token 聚类结果）
    raise NotImplementedError(
        "Cluster-Aware Upcycling 需要外部 token 聚类结果。"
        "请参考 arXiv:2604.13508 实现完整流程。"
    )
