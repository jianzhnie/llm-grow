#!/usr/bin/env python
"""MSG multi-dimensional growth expansion script.

用法:
    python examples/generic/multi_axis_pad.py \
        --model Qwen/Qwen3-8B \
        --num-new-layers 10 \
        --hidden-size-expansion 512 \
        --intermediate-size-expansion 3072 \
        --output-dir ./expanded_msg \
        --verify
"""

from __future__ import annotations

import argparse
import copy

import torch

from llm_grow.expanders.width.multi_axis_pad import (
    MultiAxisPadConfig,
    MultiAxisPadExpander,
)
from llm_grow.utils.arch_info import param_diff_report
from llm_grow.utils.model_io import load_model, load_tokenizer, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MSG multi-dimensional growth")
    p.add_argument("--model", required=True)
    p.add_argument(
        "--num-new-layers", "--depth-expansion", type=int, default=0, help="新增层数"
    )
    p.add_argument("--hidden-size-expansion", type=int, default=0)
    p.add_argument("--intermediate-size-expansion", type=int, default=0)
    p.add_argument("--no-freeze", action="store_true")
    p.add_argument("--output-dir", default="./expanded_msg")
    p.add_argument("--verify", action="store_true")
    p.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
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
        intermediate_size_expansion=args.intermediate_size_expansion,
        freeze_original=not args.no_freeze,
    )

    expander = MultiAxisPadExpander()
    expanded = expander.expand(model, config)

    param_diff_report(original_for_verify or model, expanded)

    if args.verify and original_for_verify is not None:
        expander.verify(original_for_verify, expanded)

    save_model(expanded, args.output_dir, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
