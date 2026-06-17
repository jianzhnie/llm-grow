"""MoE Upcycling: Dense → Sparse MoE (arXiv:2212.05055, Komatsuzaki et al., ICLR 2023).

核心思路：将 Dense FFN 复制 num_experts 份作为专家初始权重，
新增 Router（随机初始化）；每个 token 通过 Top-K 路由激活 K 个专家。
推理激活参数量 ≈ 原 Dense 模型（top-1 时几乎不变）。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_grow.expanders.base import AbstractExpander, ExpansionConfig
from llm_grow.initializers.symmetry_break import add_noise_to_experts


@dataclass
class MoEUpcyclingConfig(ExpansionConfig):
    num_experts: int = 8
    """每层的专家数量（Dense FFN 被复制的份数）。"""

    top_k: int = 2
    """每个 token 激活的专家数。top-1 时推理成本最小。"""

    noise_std: float = 0.01
    """打破对称性的初始噪声标准差。"""

    ffn_module_pattern: str = "mlp"
    """用于定位 FFN 模块的名称模式（前缀匹配）。"""

    router_init: str = "random"
    """Router 初始化方式：'random' | 'uniform_noise'。"""


class MoELayer(nn.Module):
    """替换 Dense FFN 的 MoE 层。"""

    def __init__(
        self,
        experts: nn.ModuleList,
        hidden_size: int,
        num_experts: int,
        top_k: int,
    ):
        super().__init__()
        self.experts = experts
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = hidden_states.shape
        flat = hidden_states.view(-1, hidden)

        router_logits = self.router(flat)
        scores = F.softmax(router_logits, dim=-1)
        topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(flat)
        for k in range(self.top_k):
            expert_ids = topk_ids[:, k]
            weights = topk_weights[:, k].unsqueeze(-1)
            for expert_idx in range(self.num_experts):
                mask = expert_ids == expert_idx
                if mask.any():
                    expert_out = self.experts[expert_idx](flat[mask])
                    output[mask] += weights[mask] * expert_out

        return output.view(bsz, seq_len, hidden)


class MoEUpcyclingExpander(AbstractExpander):
    """Dense → MoE 扩增器（Sparse Upcycling）。

    WARNING: 非 function-preserving（Router 随机初始化）。
    扩增后需要 50-100B tokens CPT + load balancing loss。
    """

    def expand(self, model: nn.Module, config: MoEUpcyclingConfig) -> nn.Module:
        hidden_size = _get_hidden_size(model)

        replaced = 0
        for name, module in list(model.named_modules()):
            if config.ffn_module_pattern not in name.split(".")[-1]:
                continue
            if not _is_ffn_module(module):
                continue

            experts = nn.ModuleList(
                [copy.deepcopy(module) for _ in range(config.num_experts)]
            )
            add_noise_to_experts(experts, std=config.noise_std)

            moe_layer = MoELayer(
                experts=experts,
                hidden_size=hidden_size,
                num_experts=config.num_experts,
                top_k=config.top_k,
            )
            _replace_submodule(model, name, moe_layer)
            replaced += 1

        print(f"[MoEUpcycling] Replaced {replaced} FFN layers with MoE layers "
              f"({config.num_experts} experts, top-{config.top_k}).")
        return model

    def verify(self, original: nn.Module, expanded: nn.Module, **kwargs) -> bool:
        print("[FP verify] MoE Upcycling is NOT function-preserving — skipping.")
        return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_ffn_module(module: nn.Module) -> bool:
    """判断是否是叶子 FFN 模块（含 gate_proj / up_proj 的 SwiGLU MLP）。"""
    child_names = {n for n, _ in module.named_children()}
    return bool({"gate_proj", "up_proj", "down_proj"} & child_names) or bool(
        {"fc1", "fc2"} & child_names
    )


def _get_hidden_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "d_model", "n_embd"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
    for _, param in model.named_parameters():
        if param.dim() == 2:
            return min(param.shape)
    return 4096


def _replace_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
