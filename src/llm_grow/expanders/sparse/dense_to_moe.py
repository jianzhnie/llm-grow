"""DenseToMoE: Dense FFN → Sparse MoE (arXiv:2212.05055, Sparse Upcycling).

Core idea: clone the Dense FFN ``num_experts`` times as initial expert
weights, add a randomly initialised Router.  Each token activates ``top_k``
experts.  Inference active parameters ≈ original Dense model (unchanged
at top-1).

Reference: Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts
    from Dense Checkpoints", arXiv:2212.05055, ICLR 2023.

Related:
    - ``ExpertClone`` (expert_clone.py): MoE expert-count expansion
    - ``MultiAxisPad`` (multi_axis_pad.py): Dense width/depth expansion
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_grow.configs.base import ModelExpansionConfig
from llm_grow.expanders.base import AbstractExpander
from llm_grow.expanders.registry import register_expander
from llm_grow.initializers.noise import DEFAULT_NOISE, NoiseStrategy
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DenseToMoEConfig(ModelExpansionConfig):
    num_experts: int = 8
    """Number of experts per layer (Dense FFN is cloned this many times)."""

    top_k: int = 2
    """Number of experts activated per token.  top-1 minimises inference cost."""

    noise_std: float = 0.01
    """Noise standard deviation for symmetry breaking."""

    noise: NoiseStrategy | None = None
    """Pluggable noise strategy.  ``None`` → :class:`GaussianNoise` (default)."""

    ffn_module_pattern: str = "mlp"
    """Name pattern for locating FFN modules (prefix match)."""

    hidden_size: int | None = None
    """Explicit hidden_size.  Inferred from config or parameters if not given.
    Raises ValueError when inference fails.
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
    """MoE layer replacing a Dense FFN.

    Uses scatter-gather batching: tokens are grouped by routing result and
    fed to experts in bulk, avoiding per-token Python loops for throughput.
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


@register_expander("dense_to_moe")
class DenseToMoEExpander(AbstractExpander[DenseToMoEConfig]):
    """DenseToMoE expander (Dense → Sparse MoE).

    .. warning::
        NOT function-preserving (Router is randomly initialised).
        Requires 50–100B tokens CPT + load-balancing loss after expansion.
    """

    def expand(self, model: nn.Module, config: DenseToMoEConfig) -> nn.Module:
        hidden_size = _get_hidden_size(model, config.hidden_size)
        noise_strategy = config.noise or DEFAULT_NOISE

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
                            noise_strategy.apply(param.data, scale=config.noise_std)
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
                    try:
                        setattr(cfg, attr, config.num_experts)
                    except AttributeError:
                        logger.warning(
                            "Cannot set config.%s (read-only property); "
                            "the expanded model config may need manual updating.",
                            attr,
                        )
            for attr in ("top_k", "num_experts_per_tok", "moe_topk"):
                if hasattr(cfg, attr):
                    try:
                        setattr(cfg, attr, config.top_k)
                    except AttributeError:
                        logger.warning(
                            "Cannot set config.%s (read-only property); "
                            "the expanded model config may need manual updating.",
                            attr,
                        )

        return model

    def verify(self, original: nn.Module, expanded: nn.Module, **kwargs) -> bool:
        logger.info("DenseToMoE is NOT function-preserving — skipping.")
        return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_ffn_module(module: nn.Module) -> bool:
    """Check whether *module* is a leaf FFN module (SwiGLU MLP with
    gate_proj / up_proj, or a legacy fc1 / fc2 MLP).
    """
    child_names = {n for n, _ in module.named_children()}
    return bool({"gate_proj", "up_proj", "down_proj"} & child_names) or bool(
        {"fc1", "fc2"} & child_names
    )


def _get_hidden_size(model: nn.Module, explicit_hidden_size: int | None = None) -> int:
    """Infer or return hidden_size.  Raises ValueError when inference fails."""
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
