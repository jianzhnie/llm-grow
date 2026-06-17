"""Load balancing loss for MoE training."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def load_balance_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int = 2,
    coeff: float = 1e-2,
) -> torch.Tensor:
    """Switch Transformer 负载均衡损失（arXiv:2101.03961）。

    L_balance = coeff * num_experts * sum_i(f_i * p_i)

    其中：
      f_i = 路由到专家 i 的 token 比例（实际负载）
      p_i = 所有 token 对专家 i 的平均路由概率

    Args:
        router_logits: 形状 (num_tokens, num_experts) 的 Router 原始输出。
        num_experts:   专家总数。
        top_k:         每个 token 激活的专家数。
        coeff:         损失系数（建议 1e-3 ~ 1e-2）。

    Returns:
        标量负载均衡损失。
    """
    probs = F.softmax(router_logits, dim=-1)
    _, indices = torch.topk(router_logits, top_k, dim=-1)

    num_tokens = router_logits.shape[0]
    dispatch_mask = torch.zeros_like(probs)
    dispatch_mask.scatter_(1, indices, 1.0)

    f = dispatch_mask.mean(dim=0)
    p = probs.mean(dim=0)

    return coeff * num_experts * (f * p).sum()


def z_loss(router_logits: torch.Tensor, coeff: float = 1e-3) -> torch.Tensor:
    """Router z-loss（ST-MoE, arXiv:2202.08906）。

    惩罚 Router logits 的绝对值，防止数值不稳定和专家 collapse。

    L_z = coeff * mean(log(sum_j(exp(logit_j)))^2)
    """
    log_z = torch.logsumexp(router_logits, dim=-1)
    return coeff * (log_z ** 2).mean()


def combined_moe_loss(
    lm_loss: torch.Tensor,
    router_logits_list: list[torch.Tensor],
    num_experts: int,
    top_k: int = 2,
    balance_coeff: float = 1e-2,
    z_coeff: float = 1e-3,
) -> torch.Tensor:
    """LM loss + 所有 MoE 层的负载均衡 + z-loss 总和。

    Args:
        lm_loss:           语言模型主损失。
        router_logits_list: 每个 MoE 层的 router_logits 列表。
        num_experts:       专家数。
        top_k:             激活专家数。
        balance_coeff:     负载均衡损失系数。
        z_coeff:           z-loss 系数。
    """
    total = lm_loss
    for logits in router_logits_list:
        total = total + load_balance_loss(logits, num_experts, top_k, balance_coeff)
        total = total + z_loss(logits, z_coeff)
    return total
