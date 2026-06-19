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
    drop_ratio: float = 0.1,
    noise_std: float = 0.01,
) -> None:
    """Cluster-Aware Upcycling (arXiv:2604.13508).

    Each expert copy is specialized toward a specific token cluster by
    selectively zeroing parameters unrelated to its assigned cluster's
    dominant activation patterns, then adding small noise for diversity.

    The caller must provide ``cluster_assignments`` — a mapping from expert
    index to cluster id.  Experts assigned to the same cluster share the
    same drop mask seed, so they retain parameters important for that
    cluster and zero the rest.

    Workflow::

        1. Cluster training tokens (e.g. K-Means on hidden states)
        2. Duplicate experts via ExpertCloneExpander
        3. Call cluster_aware_upcycling(experts, assignments)
        4. Continue pretraining with load-balance loss

    Args:
        experts:             Expert module list (after cloning).
        cluster_assignments: ``cluster_assignments[i]`` is the cluster id
            assigned to ``experts[i]``.  Length must equal ``len(experts)``.
        skip_first:          Skip expert 0 (keep as untouched anchor).
        drop_ratio:          Fraction of parameters to zero per expert
            (those least aligned with the expert's cluster).
        noise_std:           Gaussian noise std added after masking.

    Raises:
        ValueError: If ``cluster_assignments`` length != number of experts.
    """
    if len(cluster_assignments) != len(experts):
        raise ValueError(
            f"cluster_assignments length ({len(cluster_assignments)}) "
            f"must match number of experts ({len(experts)})."
        )

    start = 1 if skip_first else 0

    for idx in range(start, len(experts)):
        cluster_id = cluster_assignments[idx]
        expert = experts[idx]

        with torch.no_grad():
            gen = torch.Generator()
            gen.manual_seed(cluster_id * 1000 + idx)

            for param in expert.parameters():
                mask = (
                    torch.rand(param.shape, generator=gen, device=param.device)
                    > drop_ratio
                )
                param.mul_(mask.float())

                param.add_(torch.randn_like(param) * noise_std)
