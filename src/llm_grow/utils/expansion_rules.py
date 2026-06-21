"""Pure, I/O-free expansion rules shared by in-memory and safetensor expanders.

This module bridges the two expansion layers by centralising position/sequence
arithmetic, linear-layer classification, and padding-delta logic that is
otherwise duplicated between ``expanders/`` and ``safetensor/``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

LinearRole = Literal["hidden_to_hidden", "hidden_to_inter", "inter_to_hidden", "skip"]


def build_identity_zero_suffixes(
    attn_output_proj_names: Sequence[str] = ("o_proj", "out_proj"),
    mlp_output_proj_names: Sequence[str] = ("down_proj", "fc2"),
) -> list[str]:
    """Build the weight-suffix list that must be zeroed in identity blocks.

    Args:
        attn_output_proj_names: attention output projection leaf names.
        mlp_output_proj_names:  MLP output projection leaf names.

    Returns:
        Suffix strings such as ``self_attn.o_proj.weight``,
        ``mlp.down_proj.weight``.
    """
    suffixes: list[str] = []
    for name in attn_output_proj_names:
        suffixes.append(f"self_attn.{name}.weight")
    for name in mlp_output_proj_names:
        suffixes.append(f"mlp.{name}.weight")
    return suffixes


def classify_linear_suffix(suffix: str) -> LinearRole:
    """Classify a linear layer's semantic role from its layer-relative suffix.

    Accepts either module names (e.g. ``self_attn.q_proj``) or parameter
    names ending in ``.weight`` (e.g. ``self_attn.q_proj.weight``).

    Args:
        suffix: layer-relative identifier for the linear layer/weight.

    Returns:
        One of ``hidden_to_hidden``, ``hidden_to_inter``, ``inter_to_hidden``,
        or ``skip``.
    """
    if suffix.endswith(".weight"):
        suffix = suffix[:-7]

    if any(k in suffix for k in ("lm_head", "embed", "layernorm", "norm")):
        return "skip"

    proj_name = suffix.split(".")[-1]
    if proj_name in ("gate_proj", "up_proj"):
        return "hidden_to_inter"
    if proj_name == "down_proj":
        return "inter_to_hidden"
    return "hidden_to_hidden"


def compute_pad_deltas(
    suffix: str,
    *,
    ffn_size_expansion: int = 0,
    hidden_size_expansion: int = 0,
) -> tuple[int, int]:
    """Compute (pad_rows, pad_cols) for a layer tensor suffix.

    The rules encode how MSG-style width growth maps onto Transformer
    weight matrices:

    * FFN width (``intermediate_size``) expansion:
      - gate_proj / up_proj : rows grow
      - down_proj           : cols grow
    * Hidden-size expansion:
      - q/k/v/o projections : both rows and cols grow
      - gate_proj / up_proj : cols grow
      - down_proj           : rows grow

    Args:
        suffix: layer-relative tensor suffix (may end with ``.weight``).
        ffn_size_expansion: amount to grow ``intermediate_size``.
        hidden_size_expansion: amount to grow ``hidden_size``.

    Returns:
        ``(pad_rows, pad_cols)`` to apply to the source tensor.
    """
    pad_r = pad_c = 0
    if ffn_size_expansion <= 0 and hidden_size_expansion <= 0:
        return pad_r, pad_c

    role = classify_linear_suffix(suffix)
    if role == "skip":
        return pad_r, pad_c

    if ffn_size_expansion > 0:
        if role == "hidden_to_inter":
            pad_r += ffn_size_expansion
        elif role == "inter_to_hidden":
            pad_c += ffn_size_expansion

    if hidden_size_expansion > 0:
        if role == "hidden_to_hidden":
            pad_r += hidden_size_expansion
            pad_c += hidden_size_expansion
        elif role == "hidden_to_inter":
            pad_c += hidden_size_expansion
        elif role == "inter_to_hidden":
            pad_r += hidden_size_expansion

    return pad_r, pad_c


def build_overlap_sequence(
    num_orig: int,
    overlap: int,
) -> list[tuple[int, bool]]:
    """Build the layer sequence for OverlapCopy depth up-scaling.

    Args:
        num_orig: original number of layers.
        overlap: number of overlapping layers.

    Returns:
        List of ``(src_layer_idx, is_identity)`` tuples describing the
        expanded layer stack.  All entries have ``is_identity=False``.

    Raises:
        ValueError: if ``overlap`` is negative or not smaller than ``num_orig``.
    """
    if not isinstance(overlap, int):
        raise TypeError(f"overlap must be an integer, got {type(overlap)}")
    if overlap < 0 or overlap >= num_orig:
        raise ValueError(
            f"overlap ({overlap}) must be in [0, num_orig) = [0, {num_orig})."
        )

    upper_end = num_orig - overlap
    lower_start = overlap
    return [(i, False) for i in range(upper_end)] + [
        (i, False) for i in range(lower_start, num_orig)
    ]
