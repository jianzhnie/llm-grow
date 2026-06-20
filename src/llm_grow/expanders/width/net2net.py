"""Net2Net: Function-preserving model widening (arXiv:1511.05641, ICLR 2016).

.. deprecated::
    ``Net2NetExpander`` is deprecated.  Net2WiderNet's assumptions do not map
    cleanly onto modern Transformer LLMs (SwiGLU / GQA / RMSNorm make strict
    function preservation non-trivial).  For production width expansion use
    :class:`llm_grow.expanders.width.multi_axis_pad.MultiAxisPadExpander`.

    The low-level ``wider()`` math is preserved here as a thin wrapper around
    :func:`llm_grow.initializers.net2net.net2wider_net` for backward
    compatibility and for research use-cases that implement their own
    architecture-specific wiring.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from llm_grow.expanders.base import AbstractExpander, ExpansionConfig
from llm_grow.initializers.net2net import net2wider_net


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
    """Deprecated Net2WiderNet expander.

    .. deprecated::
        ``Net2NetExpander`` is deprecated and will be removed in a future
        release.  Use ``MultiAxisPadExpander`` for width expansion, or call
        ``net2wider_net`` from ``llm_grow.initializers`` for the low-level
        transformation.
    """

    def __init__(self) -> None:
        warnings.warn(
            "Net2NetExpander is deprecated. Use MultiAxisPadExpander for width "
            "expansion, or llm_grow.initializers.net2wider_net for the low-level "
            "transformation.",
            DeprecationWarning,
            stacklevel=2,
        )

    def expand(self, model: nn.Module, config: Net2NetConfig) -> nn.Module:
        raise NotImplementedError(
            "Net2NetExpander.expand() is deprecated and not implemented for "
            "Transformer LLMs. Use MultiAxisPadExpander for production width "
            "expansion."
        )

    def wider(
        self,
        w_in: torch.Tensor,
        w_out: torch.Tensor,
        new_width: int,
        add_noise: bool = True,
        noise_std: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deprecated wrapper around :func:`net2wider_net`."""
        return net2wider_net(w_in, w_out, new_width, add_noise, noise_std)
