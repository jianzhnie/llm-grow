"""DenseToMoE: Dense FFN 转稀疏 MoE (arXiv:2212.05055, Sparse Upcycling).

核心思路：将 Dense FFN 复制 num_experts 份作为专家初始权重，
新增 Router（随机初始化）；每个 token 通过 Top-K 路由激活 K 个专家。
推理激活参数量 ≈ 原 Dense 模型（top-1 时几乎不变）。

原始论文: Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts
    from Dense Checkpoints", arXiv:2212.05055, ICLR 2023.

Related:
    - ``ExpertClone`` (expert_clone.py): 已有 MoE 专家数扩展
    - ``MultiAxisPad`` (multi_axis_grow.py): Dense 宽度/深度扩增
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_grow.expanders.base import AbstractExpander, ExpansionConfig
from llm_grow.initializers.symmetry_break import add_noise_to_experts
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DenseToMoEConfig(ExpansionConfig):
    num_experts: int = 8
    """每层的专家数量（Dense FFN 被复制的份数）。"""

    top_k: int = 2
    """每个 token 激活的专家数。top-1 时推理成本最小。"""

    noise_std: float = 0.01
    """打破对称性的初始噪声标准差。"""

    ffn_module_pattern: str = "mlp"
    """用于定位 FFN 模块的名称模式（前缀匹配）。"""


class MoELayer(nn.Module):
    """替换 Dense FFN 的 MoE 层。

    使用 scatter-gather 批量分发策略：将 token 按路由结果分组后批量送入
    对应专家，避免 per-token Python 循环，显著提升训练和推理吞吐。
    """

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
        num_tokens = bsz * seq_len
        flat = hidden_states.view(num_tokens, hidden)

        router_logits = self.router(flat)
        scores = F.softmax(router_logits, dim=-1)
        topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        flat_ids = topk_ids.view(-1)
        flat_weights = topk_weights.view(-1, 1)
        token_indices = torch.arange(num_tokens, device=flat.device).unsqueeze(1)
        token_indices = token_indices.expand(-1, self.top_k).reshape(-1)

        expert_outputs = torch.zeros(
            num_tokens, hidden, dtype=flat.dtype, device=flat.device
        )

        for expert_idx in range(self.num_experts):
            mask = flat_ids == expert_idx
            if not mask.any():
                continue
            selected_tokens = token_indices[mask]
            expert_input = flat[selected_tokens]
            expert_out = self.experts[expert_idx](expert_input)
            weighted_out = flat_weights[mask] * expert_out
            expert_outputs.index_add_(0, selected_tokens, weighted_out)

        return expert_outputs.view(bsz, seq_len, hidden)


class DenseToMoEExpander(AbstractExpander):
    """DenseToMoE 扩增器（Dense → Sparse MoE）。

    WARNING: 非 function-preserving（Router 随机初始化）。
    扩增后需要 50-100B tokens CPT + load balancing loss。
    """

    def expand(self, model: nn.Module, config: DenseToMoEConfig) -> nn.Module:
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

        logger.info(
            "Replaced %d FFN layers with MoE layers (%d experts, top-%d).",
            replaced,
            config.num_experts,
            config.top_k,
        )
        return model

    def verify(self, original: nn.Module, expanded: nn.Module, **kwargs) -> bool:
        logger.info("DenseToMoE is NOT function-preserving — skipping.")
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
    for name, param in model.named_parameters():
        if param.dim() == 2 and "embed" not in name:
            return min(param.shape)
    return 4096


def _replace_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
