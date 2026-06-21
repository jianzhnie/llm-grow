"""DenseToMoE: Dense FFN 转稀疏 MoE (arXiv:2212.05055, Sparse Upcycling).

核心思路：将 Dense FFN 复制 num_experts 份作为专家初始权重，
新增 Router（随机初始化）；每个 token 通过 Top-K 路由激活 K 个专家。
推理激活参数量 ≈ 原 Dense 模型（top-1 时几乎不变）。

原始论文: Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts
    from Dense Checkpoints", arXiv:2212.05055, ICLR 2023.

Related:
    - ``ExpertClone`` (expert_clone.py): 已有 MoE 专家数扩展
    - ``MultiAxisPad`` (multi_axis_pad.py): Dense 宽度/深度扩增
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_grow.configs.base import ModelExpansionConfig
from llm_grow.expanders.base import AbstractExpander
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DenseToMoEConfig(ModelExpansionConfig):
    num_experts: int = 8
    """每层的专家数量（Dense FFN 被复制的份数）。"""

    top_k: int = 2
    """每个 token 激活的专家数。top-1 时推理成本最小。"""

    noise_std: float = 0.01
    """打破对称性的初始噪声标准差。"""

    ffn_module_pattern: str = "mlp"
    """用于定位 FFN 模块的名称模式（前缀匹配）。"""

    hidden_size: int | None = None
    """显式指定 hidden_size。若未提供，则从 model.config 或参数推断；
    推断失败时抛出 ValueError，不再使用魔法默认值。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.num_experts, int) or self.num_experts <= 0:
            raise ValueError(
                f"num_experts must be a positive integer, got {self.num_experts!r}"
            )
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k!r}")
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})"
            )
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")
        if not isinstance(self.ffn_module_pattern, str) or not self.ffn_module_pattern:
            raise ValueError(
                "ffn_module_pattern must be a non-empty string, "
                f"got {self.ffn_module_pattern!r}"
            )
        if self.hidden_size is not None and (
            not isinstance(self.hidden_size, int) or self.hidden_size <= 0
        ):
            raise ValueError(
                "hidden_size must be a positive integer or None, "
                f"got {self.hidden_size!r}"
            )


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
        """Scatter-gather MoE forward with per-expert batched computation."""
        bsz, seq_len, hidden = hidden_states.shape
        num_tokens = bsz * seq_len
        flat = hidden_states.view(num_tokens, hidden)

        router_logits = self.router(flat)
        scores = F.softmax(router_logits, dim=-1)
        topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        flat_ids = topk_ids.view(-1)
        flat_weights = topk_weights.view(-1, 1)
        token_indices = (
            torch.arange(num_tokens, device=flat.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )

        expert_outputs = torch.zeros(
            num_tokens, hidden, dtype=flat.dtype, device=flat.device
        )

        sorted_expert_ids, sort_idx = flat_ids.sort()
        sorted_token_indices = token_indices[sort_idx]
        sorted_weights = flat_weights[sort_idx]

        expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
        splits = expert_counts.tolist()
        token_splits = sorted_token_indices.split(splits)
        weight_splits = sorted_weights.split(splits)

        for expert_idx in range(self.num_experts):
            if splits[expert_idx] == 0:
                continue
            selected_tokens = token_splits[expert_idx]
            expert_input = flat[selected_tokens]
            expert_out = self.experts[expert_idx](expert_input)
            weighted_out = weight_splits[expert_idx] * expert_out
            expert_outputs.index_add_(0, selected_tokens, weighted_out)

        return expert_outputs.view(bsz, seq_len, hidden)


class DenseToMoEExpander(AbstractExpander[DenseToMoEConfig]):
    """DenseToMoE 扩增器（Dense → Sparse MoE）。

    WARNING: 非 function-preserving（Router 随机初始化）。
    扩增后需要 50-100B tokens CPT + load balancing loss。
    """

    def expand(self, model: nn.Module, config: DenseToMoEConfig) -> nn.Module:
        hidden_size = _get_hidden_size(model, config.hidden_size)

        replaced = 0
        for name, module in list(model.named_modules()):
            if name.split(".")[-1] != config.ffn_module_pattern:
                continue
            if not _is_ffn_module(module):
                continue

            experts = nn.ModuleList()
            for i in range(config.num_experts):
                expert = copy.deepcopy(module)
                if i > 0:
                    with torch.no_grad():
                        for param in expert.parameters():
                            param.add_(torch.randn_like(param) * config.noise_std)
                experts.append(expert)

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

        cfg = getattr(model, "config", None)
        if cfg is not None:
            for attr in ("num_experts", "n_routed_experts"):
                if hasattr(cfg, attr):
                    setattr(cfg, attr, config.num_experts)
            for attr in ("top_k", "num_experts_per_tok", "moe_topk"):
                if hasattr(cfg, attr):
                    setattr(cfg, attr, config.top_k)

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


def _get_hidden_size(model: nn.Module, explicit_hidden_size: int | None = None) -> int:
    """推断或返回 hidden_size。无法推断时抛出 ValueError。"""
    if explicit_hidden_size is not None:
        return explicit_hidden_size

    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "d_model", "n_embd"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))

    for name, param in model.named_parameters():
        if param.dim() == 2 and "embed" not in name:
            return min(param.shape)

    raise ValueError(
        "Cannot infer hidden_size from config or parameters. "
        "Please provide DenseToMoEConfig(hidden_size=...)."
    )


def _replace_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
