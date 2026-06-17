"""Net2Net: Function-preserving model widening (arXiv:1511.05641, ICLR 2016).

核心思路：
  - Net2WiderNet：复制 k 列（输入权重），同步缩放对应行（输出权重），输出不变。
  - Net2DeeperNet：插入恒等层。

在 LLM 时代主要作为 MSG 的理论基础；独立使用时复制神经元存在对称性问题。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from llm_grow.expanders.base import AbstractExpander, ExpansionConfig


@dataclass
class Net2NetConfig(ExpansionConfig):
    width_multiplier: float = 2.0
    """目标宽度倍数（hidden_size 和 intermediate_size 统一缩放）。"""

    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
    )
    """需要做宽度扩展的模块名称（前缀匹配）。"""

    add_noise: bool = True
    """对复制列加入小噪声，缓解梯度对称性问题。"""

    noise_std: float = 1e-3


class Net2NetExpander(AbstractExpander):
    """Net2WiderNet 宽度扩增器。

    注意：直接在 LLM 上使用时效果受限（SwiGLU / GQA 的非线性使严格 FP 需额外适配），
    建议优先使用 MSG（更完整的 LLM 宽度扩增实现）。
    """

    def expand(self, model: nn.Module, config: Net2NetConfig) -> nn.Module:
        raise NotImplementedError(
            "Net2NetExpander.expand() 需要针对具体架构实现列复制逻辑。"
            "对于 Transformer LLM，建议直接使用 MSGExpander。"
        )

    def wider(
        self,
        w_in: torch.Tensor,
        w_out: torch.Tensor,
        new_width: int,
        add_noise: bool = True,
        noise_std: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Net2WiderNet 核心变换（函数工具，可独立调用）。

        Args:
            w_in:  形状 (out_features, in_features)，待扩展层的权重。
            w_out: 形状 (*, out_features)，下一层需配套缩放的权重。
            new_width: 扩展后的 out_features。

        Returns:
            (w_in_new, w_out_new)  满足 w_out_new @ w_in_new ≈ w_out @ w_in。
        """
        old_width = w_in.shape[0]
        extra = new_width - old_width
        if extra <= 0:
            return w_in, w_out

        indices = torch.randint(0, old_width, (extra,))
        copies = w_in[indices].clone()
        if add_noise:
            copies += torch.randn_like(copies) * noise_std
        w_in_new = torch.cat([w_in, copies], dim=0)

        scale = torch.ones(old_width, device=w_out.device)
        for idx in indices:
            scale[idx] += 1
        scale_out = torch.cat([scale, torch.ones(extra, device=w_out.device)])
        w_out_new = w_out / scale_out.unsqueeze(0) if w_out.dim() == 2 else w_out

        return w_in_new, w_out_new
