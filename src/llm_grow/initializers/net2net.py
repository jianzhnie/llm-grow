"""Net2WiderNet core transformation as a standalone utility.

Net2Net (arXiv:1511.05641) widens a layer by replicating neurons and
symmetrically scaling the next layer's weights, preserving the function
computed by the network.  In modern Transformer LLMs, SwiGLU / GQA make a
strict function-preserving widening non-trivial, so this is exposed as a
low-level building block rather than a full-model expander.
"""

from __future__ import annotations

import torch


def net2wider_net(
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    new_width: int,
    add_noise: bool = True,
    noise_std: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Net2WiderNet: widen ``w_in`` to ``new_width`` while keeping
    ``w_out @ w_in`` approximately unchanged.

    Args:
        w_in:  Shape ``(out_features, in_features)``.
        w_out: Shape ``(*, out_features)``; the layer that consumes ``w_in``'s
               output.
        new_width: Target ``out_features`` for ``w_in``.
        add_noise: Whether to add Gaussian noise to copied rows.
        noise_std: Noise standard deviation.

    Returns:
        ``(w_in_new, w_out_new)`` with shapes ``(new_width, in_features)``
        and ``(*, new_width)``.
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

    counts = torch.bincount(indices, minlength=old_width)
    scale = 1.0 + counts.float().to(w_out.device)
    scale_out = torch.cat([scale, scale[indices]])
    w_out_expanded = (
        torch.cat([w_out, w_out[:, indices]], dim=1) if w_out.dim() == 2 else w_out
    )
    w_out_new = (
        w_out_expanded / scale_out.unsqueeze(0) if w_out.dim() == 2 else w_out
    )

    return w_in_new, w_out_new
