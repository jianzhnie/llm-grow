#!/usr/bin/env python
"""LLaMA-Pro identity block expansion (arXiv:2401.02415).

Usage:
    python examples/generic/zero_block_insert.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --num-new-layers 9 \\
        --output-dir ./expanded_llama_pro --verify
"""

from __future__ import annotations

import argparse
import copy

import torch

from llm_grow.expanders.depth.zero_block_insert import ZeroBlockInsertConfig
from llm_grow.expanders.registry import get_expander
from llm_grow.utils.arch_info import param_diff_report
from llm_grow.utils.model_io import load_model, load_tokenizer, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLaMA-Pro block expansion")
    p.add_argument("--model", required=True, help="HuggingFace model name or local path")
    p.add_argument("--num-new-layers", type=int, default=8, help="Identity blocks to insert")
    p.add_argument(
        "--insert-strategy", default="uniform", choices=["uniform", "front", "rear"],
        help="uniform: evenly spaced | front: beginning | rear: end",
    )
    p.add_argument("--no-freeze", action="store_true", help="Do not freeze original layers")
    p.add_argument("--output-dir", default="./expanded_zero_block_insert")
    p.add_argument("--verify", action="store_true", help="Run FP verification after expand")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    print(f"[ZeroBlockInsert] Loading {args.model} ...")
    model = load_model(args.model, dtype=dtype)
    tokenizer = load_tokenizer(args.model)

    original_for_verify = copy.deepcopy(model) if args.verify else None

    config = ZeroBlockInsertConfig(
        num_new_layers=args.num_new_layers,
        insert_strategy=args.insert_strategy,
        freeze_original=not args.no_freeze,
    )

    expander = get_expander("zero_block_insert")()
    expanded = expander.expand(model, config)

    if original_for_verify is not None:
        param_diff_report(original_for_verify, expanded)
    else:
        param_diff_report(model, expanded)

    if args.verify and original_for_verify is not None:
        expander.verify(original_for_verify, expanded)

    save_model(expanded, args.output_dir, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
