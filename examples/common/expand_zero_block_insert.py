#!/usr/bin/env python
"""LLaMA-Pro block expansion script.

用法:
    python examples/common/expand_zero_block_insert.py \
        --model Qwen/Qwen3-8B \
        --num-new-layers 9 \
        --output-dir ./expanded_zero_block_insert \
        --verify
"""

from __future__ import annotations

import argparse

import torch

from llm_grow.expanders.depth.zero_block_insert import (
    ZeroBlockInsertConfig,
    ZeroBlockInsertExpander,
)
from llm_grow.utils.arch_info import param_diff_report
from llm_grow.utils.model_io import load_model, load_tokenizer, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLaMA-Pro block expansion")
    p.add_argument("--model", required=True, help="原始模型路径或 HuggingFace model id")
    p.add_argument(
        "--num-new-layers",
        "--num-new-blocks",
        type=int,
        default=8,
        help="插入的恒等块数量",
    )
    p.add_argument(
        "--insert-strategy", default="uniform", choices=["uniform", "front", "rear"]
    )
    p.add_argument("--no-freeze", action="store_true", help="不冻结原始层")
    p.add_argument("--output-dir", default="./expanded_zero_block_insert")
    p.add_argument(
        "--verify", action="store_true", help="扩增后验证 function-preserving"
    )
    p.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    print(f"[ZeroBlockInsert] Loading {args.model} ...")
    model = load_model(args.model, dtype=dtype)
    tokenizer = load_tokenizer(args.model)

    if args.verify:
        import copy

        original_for_verify = copy.deepcopy(model)

    config = ZeroBlockInsertConfig(
        num_new_layers=args.num_new_layers,
        insert_strategy=args.insert_strategy,
        freeze_original=not args.no_freeze,
    )

    expander = ZeroBlockInsertExpander()
    expanded = expander.expand(model, config)

    param_diff_report(original_for_verify if args.verify else model, expanded)

    if args.verify:
        expander.verify(original_for_verify, expanded)

    save_model(expanded, args.output_dir, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
