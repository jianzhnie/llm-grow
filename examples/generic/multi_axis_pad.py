#!/usr/bin/env python
"""MSG multi-dimensional growth: depth + width mask expansion.

Usage:
    python examples/generic/multi_axis_pad.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --num-new-layers 4 --ffn-size-expansion 512 \\
        --output-dir ./expanded_msg --verify
"""

from __future__ import annotations

import argparse
import copy

import torch

from llm_grow.expanders.registry import get_expander
from llm_grow.expanders.width.multi_axis_pad import MultiAxisPadConfig
from llm_grow.utils.arch_info import param_diff_report
from llm_grow.utils.model_io import load_model, load_tokenizer, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MSG multi-dimensional growth")
    p.add_argument("--model", required=True, help="HuggingFace model name or local path")
    p.add_argument("--num-new-layers", type=int, default=0, help="Identity blocks to insert")
    p.add_argument("--hidden-size-expansion", type=int, default=0)
    p.add_argument("--ffn-size-expansion", type=int, default=0)
    p.add_argument("--no-freeze", action="store_true", help="Do not freeze original layers")
    p.add_argument("--output-dir", default="./expanded_msg")
    p.add_argument("--verify", action="store_true", help="Run FP verification after expand")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    print(f"[MultiAxisPad] Loading {args.model} ...")
    model = load_model(args.model, dtype=dtype)
    tokenizer = load_tokenizer(args.model)

    original_for_verify = copy.deepcopy(model) if args.verify else None

    config = MultiAxisPadConfig(
        num_new_layers=args.num_new_layers,
        hidden_size_expansion=args.hidden_size_expansion,
        ffn_size_expansion=args.ffn_size_expansion,
        freeze_original=not args.no_freeze,
    )

    expander = get_expander("multi_axis_pad")()
    expanded = expander.expand(model, config)

    param_diff_report(original_for_verify or model, expanded)

    if args.verify and original_for_verify is not None:
        expander.verify(original_for_verify, expanded)

    save_model(expanded, args.output_dir, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
