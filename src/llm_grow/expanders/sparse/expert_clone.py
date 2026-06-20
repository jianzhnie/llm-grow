"""ExpertClone: MoE 专家克隆扩展 (arXiv:2604.19835, ExpertClone).

核心思路（M1 方案）：将已有 E 个专家各复制一份得到 mE 个专家，
保持 Top-K 路由不变，推理激活参数量不变，总参数量线性增长。
关键：必须打破对称性，否则副本无法专业化。

原始论文: Amazon AI, "ExpertClone: Densely Activated
    Mixture-of-Experts Pre-Training", arXiv:2604.19835, 2026.

Related:
    - ``DenseToMoE`` (dense_to_moe.py): Dense 转 MoE（从零创建专家）
    - ``ZeroBlockInsert`` (zero_block_insert.py): MoE 模型的深度扩增
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from llm_grow.configs.base import ModelExpansionConfig
from llm_grow.expanders.base import AbstractExpander
from llm_grow.initializers.symmetry_break import (
    add_noise_to_experts,
    drop_upcycling,
)
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


class ExpertSelectionStrategy(str, Enum):
    UNIFORM = "uniform"
    UTILITY = "utility"
    RANDOM_SUBSET = "random_subset"


@dataclass
class ExpertCloneConfig(ModelExpansionConfig):
    expand_factor: int = 2
    """专家数扩展倍数（原 E 个专家 → expand_factor × E 个专家）。"""

    selection_strategy: ExpertSelectionStrategy = ExpertSelectionStrategy.UNIFORM
    """专家选择策略：
    - 'uniform'       : 每个专家等概率复制（简单基线）
    - 'utility'       : 按梯度重要性优先复制高价值专家（论文推荐，差距闭合 3x）
    - 'random_subset' : 随机选部分专家复制（效果不稳定）
    """

    symmetry_break: str = "noise"
    """对称性破坏方式：'noise' | 'drop' | 'cluster'（cluster 需额外实现）。"""

    noise_std: float = 0.01
    drop_ratio: float = 0.1

    router_noise_std: float = 0.001
    """Router 新增列的扰动强度。"""

    moe_layer_cls_name: str = ""
    """目标 MoE 层的类名（用于显式按类名匹配）。
    默认空字符串，此时通过 ``hasattr(module, 'experts') and hasattr(module, 'router')``
    判断，更加稳健。
    """


class ExpertCloneExpander(AbstractExpander):
    """ExpertClone MoE 专家克隆扩展器（M1 方案）。

    适用于基座已经是 MoE 架构的模型（Qwen3-MoE、DeepSeek-V3、Mixtral 等）。
    扩展后推理激活参数量几乎不变（Router 计算量微增），总参数线性增长。
    """

    def expand(self, model: nn.Module, config: ExpertCloneConfig) -> nn.Module:
        expanded_count = 0
        for _name, module in model.named_modules():
            is_target_cls = (
                bool(config.moe_layer_cls_name)
                and type(module).__name__ == config.moe_layer_cls_name
            )
            has_moe_interface = hasattr(module, "experts") and hasattr(module, "router")
            if not is_target_cls and not has_moe_interface:
                continue

            new_experts, new_router_weight = _expand_experts(module, config)
            module.experts = new_experts
            _update_router(module, new_router_weight)
            expanded_count += 1

        logger.info(
            "Expanded %d MoE layers, factor=%dx, strategy=%s.",
            expanded_count,
            config.expand_factor,
            config.selection_strategy,
        )
        return model

    def verify(self, original: nn.Module, expanded: nn.Module, **kwargs) -> bool:
        logger.info(
            "ExpertClone requires symmetry breaking — "
            "output will differ; skipping strict FP check."
        )
        return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _expand_experts(
    moe_module: nn.Module,
    config: ExpertCloneConfig,
) -> tuple[nn.ModuleList, torch.Tensor]:
    experts: nn.ModuleList = moe_module.experts
    num_orig = len(experts)
    factor = config.expand_factor
    strategy = config.selection_strategy

    if strategy == ExpertSelectionStrategy.UNIFORM:
        src_indices = list(range(num_orig)) * (factor - 1)
    elif strategy == ExpertSelectionStrategy.UTILITY:
        scores = _compute_utility_scores(experts)
        src_indices = _utility_select(scores, num_orig * (factor - 1))
    else:
        src_indices = torch.randint(0, num_orig, (num_orig * (factor - 1),)).tolist()

    new_experts = nn.ModuleList(list(experts))
    for src_idx in src_indices:
        clone = copy.deepcopy(experts[src_idx])
        if config.symmetry_break == "noise":
            add_noise_to_experts(nn.ModuleList([clone]), std=config.noise_std)
        elif config.symmetry_break == "drop":
            drop_upcycling(nn.ModuleList([clone]), drop_ratio=config.drop_ratio)
        new_experts.append(clone)

    router_w = moe_module.router.weight.data
    new_router_w = _expand_router_weight(
        router_w, src_indices, num_orig, config.router_noise_std
    )
    return new_experts, new_router_w


def _expand_router_weight(
    router_w: torch.Tensor,
    src_indices: list[int],
    num_orig: int,
    noise_std: float,
) -> torch.Tensor:
    """Router 权重：shape (num_experts, hidden_size)。
    为每个副本新增行（从源专家行初始化 + 小噪声）。"""
    new_rows = []
    for src_idx in src_indices:
        row = router_w[src_idx, :].clone()  # (hidden_size,)
        row += torch.randn_like(row) * noise_std
        new_rows.append(row.unsqueeze(0))  # (1, hidden_size)
    return torch.cat([router_w, *new_rows], dim=0)  # (num_new_experts, hidden_size)


def _update_router(moe_module: nn.Module, new_weight: torch.Tensor) -> None:
    """new_weight: (num_new_experts, hidden_size)"""
    old_router = moe_module.router
    num_new_experts, hidden_size = new_weight.shape
    new_router = nn.Linear(
        hidden_size, num_new_experts, bias=old_router.bias is not None
    )
    new_router.weight = nn.Parameter(new_weight)  # shape 已对齐，无需转置
    moe_module.router = new_router


def _compute_utility_scores(experts: nn.ModuleList) -> list[float]:
    """粗略的参数 L2 范数作为效用得分（完整实现应使用梯度重要性）。"""
    scores = []
    for expert in experts:
        total = sum(p.data.norm().item() for p in expert.parameters())
        scores.append(total)
    return scores


def _utility_select(scores: list[float], n: int) -> list[int]:
    """按效用得分降序选取 n 个索引（允许重复）。"""
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    result: list[int] = []
    while len(result) < n:
        result.extend(ranked[: n - len(result)])
    return result[:n]
