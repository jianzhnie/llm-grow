#!/usr/bin/env python
"""Expert Upcycling script: expand MoE expert count (M1 strategy).

用法:
    python examples/generic/expert_clone.py \
        --model path/to/qwen3-moe \
        --expand-factor 2 \
        --selection-strategy utility \
        --symmetry-break noise \
        --output-dir ./expanded_experts
"""

from __future__ import annotations

import argparse
import copy

import torch

from llm_grow.expanders.sparse.expert_clone import (
    ExpertCloneConfig,
    ExpertCloneExpander,
    ExpertSelectionStrategy,
)
from llm_grow.utils.arch_info import param_diff_report
from llm_grow.utils.model_io import load_model, load_tokenizer, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE Expert Upcycling (M1)")
    p.add_argument("--model", required=True)
    p.add_argument("--expand-factor", type=int, default=2)
    p.add_argument(
        "--selection-strategy",
        default="uniform",
        choices=["uniform", "utility", "random_subset"],
    )
    p.add_argument("--symmetry-break", default="noise", choices=["noise", "drop"])
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--drop-ratio", type=float, default=0.1)
    p.add_argument("--output-dir", default="./expanded_experts")
    p.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    print(f"[ExpertClone] Loading {args.model} ...")
    model = load_model(args.model, dtype=dtype)
    tokenizer = load_tokenizer(args.model)
    original_ref = copy.deepcopy(model)

    config = ExpertCloneConfig(
        expand_factor=args.expand_factor,
        selection_strategy=ExpertSelectionStrategy(args.selection_strategy),
        symmetry_break=args.symmetry_break,
        noise_std=args.noise_std,
        drop_ratio=args.drop_ratio,
    )

    expander = ExpertCloneExpander()
    expanded = expander.expand(model, config)

    param_diff_report(original_ref, expanded)
    save_model(expanded, args.output_dir, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
