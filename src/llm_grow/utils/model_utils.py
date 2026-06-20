"""Generic model traversal and mutation helpers for Transformer decoder stacks."""

from __future__ import annotations

import torch.nn as nn

from llm_grow.utils.insertion import DECODER_LAYER_ATTRS


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder layer list in a HuggingFace-style causal LM.

    Tries common attribute paths such as ``model.layers``, ``transformer.h``,
    and ``decoder.layers``.
    """
    for attr in DECODER_LAYER_ATTRS:
        obj: nn.Module | None = model
        for part in attr.split("."):
            if obj is None:
                break
            obj = getattr(obj, part, None)
        if isinstance(obj, nn.ModuleList):
            return obj
    raise AttributeError("Cannot locate decoder layer list in model.")


def set_decoder_layers(model: nn.Module, new_layers: nn.ModuleList) -> None:
    """Replace the decoder layer list in a model with ``new_layers``."""
    for attr in DECODER_LAYER_ATTRS:
        parts = attr.split(".")
        obj: nn.Module | None = model
        for part in parts[:-1]:
            if obj is None:
                break
            obj = getattr(obj, part, None)
        if obj is not None and hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], new_layers)
            return
    raise AttributeError("Cannot set decoder layer list in model.")


def update_num_hidden_layers(model: nn.Module, new_num: int) -> None:
    """Update the layer count attribute on ``model.config``.

    Prefers the standard ``num_hidden_layers`` key; falls back to ``num_layers``
    or ``n_layer`` depending on what the source config actually uses.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    for attr in ("num_hidden_layers", "num_layers", "n_layer"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, new_num)
            return
