#!/usr/bin/env python
"""MoE Upcycling script: Dense FFN → Sparse MoE.

用法:
    python examples/generic/dense_to_moe.py \
        --model Qwen/Qwen3-8B \
        --num-experts 8 \
        --top-k 2 \
        --noise-std 0.01 \
        --output-dir ./expanded_moe
"""

from __future__ import annotations

import argparse

import torch

from llm_grow.expanders.sparse.dense_to_moe import (
    DenseToMoEConfig,
    DenseToMoEExpander,
)
from llm_grow.utils.arch_info import param_diff_report
from llm_grow.utils.model_io import load_model, load_tokenizer, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense → MoE Upcycling")
    p.add_argument("--model", required=True)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--ffn-pattern", default="mlp", help="用于定位 FFN 模块的名称模式")
    p.add_argument("--output-dir", default="./expanded_moe")
    p.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    print(f"[DenseToMoE] Loading {args.model} ...")
    model = load_model(args.model, dtype=dtype)
    tokenizer = load_tokenizer(args.model)

    config = DenseToMoEConfig(
        num_experts=args.num_experts,
        top_k=args.top_k,
        noise_std=args.noise_std,
        ffn_module_pattern=args.ffn_pattern,
    )

    import copy

    original_ref = copy.deepcopy(model)

    expander = DenseToMoEExpander()
    expanded = expander.expand(model, config)

    param_diff_report(original_ref, expanded)
    save_model(expanded, args.output_dir, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
