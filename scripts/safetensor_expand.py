#!/usr/bin/env python
"""Safetensor-level model expansion — no full model load required.

Supports an ``auto`` mode that detects Dense vs MoE and picks the
correct expander automatically.

Usage examples
--------------
# Auto-detect (recommended)
python scripts/safetensor_expand.py auto \\
    --src  /path/to/model \\
    --dst  ./output \\
    --method depth --num-new-layers 4

python scripts/safetensor_expand.py auto \\
    --src  /path/to/moe_model \\
    --dst  ./output \\
    --method expert --expand-factor 2

# Explicit expanders
python scripts/safetensor_expand.py zero_block_insert \\
    --src  /path/to/Qwen3-8B \\
    --dst  ./outputs/qwen3_zbi \\
    --num-new-layers 7

python scripts/safetensor_expand.py overlap_copy \\
    --src  /path/to/Qwen3-8B \\
    --dst  ./outputs/qwen3_oc \\
    --num-overlap 8

python scripts/safetensor_expand.py multi_axis_pad \\
    --src  /path/to/Qwen3-8B \\
    --dst  ./outputs/qwen3_map \\
    --num-new-layers 4 \\
    --ffn-size-expansion 1024

python scripts/safetensor_expand.py expert_clone \\
    --src  /path/to/Qwen3-30B-A3B \\
    --dst  ./outputs/qwen3_30b_2x \\
    --expand-factor 2
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Safetensor-level LLM expansion (no full model load)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "expander",
        choices=[
            "auto",
            "zero_block_insert",
            "overlap_copy",
            "multi_axis_pad",
            "expert_clone",
        ],
        help=(
            "auto              : detect Dense/MoE and select the right method\n"
            "zero_block_insert : identity block insertion (Dense depth)\n"
            "overlap_copy      : layer overlap-copy (Dense depth)\n"
            "multi_axis_pad    : depth + FFN-width (Dense)\n"
            "expert_clone      : expert count expansion (MoE only)"
        ),
    )
    p.add_argument("--src", required=True, help="Source model directory")
    p.add_argument("--dst", required=True, help="Output directory")
    p.add_argument(
        "--dry-run", action="store_true", help="Build plan without writing files"
    )
    p.add_argument(
        "--target-shard-gb",
        type=float,
        default=None,
        help="Output shard size in GB (default: auto-detect from source)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel writer processes (0=CPU count, 1=serial)",
    )
    p.add_argument("--quiet", action="store_true")

    # auto params
    p.add_argument(
        "--method",
        default="depth",
        choices=["depth", "expert", "width"],
        help="[auto] Expansion axis",
    )

    # depth / ZeroBlockInsert / MultiAxisPad
    p.add_argument(
        "--num-new-layers",
        "--num-new-blocks",
        type=int,
        default=4,
        help="[auto depth / zero_block_insert] layers to insert",
    )
    p.add_argument(
        "--insert-strategy", default="uniform", choices=["uniform", "front", "rear"]
    )

    # OverlapCopy
    p.add_argument(
        "--num-overlap", type=int, default=8, help="[overlap_copy] overlapping layers"
    )

    # MultiAxisPad width
    p.add_argument(
        "--ffn-size-expansion",
        type=int,
        default=0,
        help="[multi_axis_pad / auto width] intermediate_size increment",
    )

    # expert
    p.add_argument(
        "--expand-factor",
        type=int,
        default=2,
        help="[auto expert / expert_clone] expert count multiplier",
    )
    p.add_argument(
        "--noise-scale",
        type=float,
        default=1e-6,
        help="[auto expert / expert_clone] router noise scale",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    target_bytes = (
        int(args.target_shard_gb * 1024**3)
        if args.target_shard_gb is not None
        else None
    )
    verbose = not args.quiet

    # ── auto mode: detect + dispatch ─────────────────────────────────────────
    if args.expander == "auto":
        from llm_grow.safetensor.auto import auto_expand

        auto_expand(
            src_dir=args.src,
            dst_dir=args.dst,
            method=args.method,
            num_new_layers=args.num_new_layers,
            insert_strategy=args.insert_strategy,
            expand_factor=args.expand_factor,
            noise_scale=args.noise_scale,
            ffn_size_expansion=args.ffn_size_expansion,
            target_shard_gb=args.target_shard_gb or 4.0,
            verbose=verbose,
            dry_run=args.dry_run,
            workers=args.workers,
        )
        return

    # ── explicit expanders ────────────────────────────────────────────────────
    if args.expander == "zero_block_insert":
        from llm_grow.safetensor.zero_block_insert import (
            ZeroBlockInsertSafetensorConfig,
            ZeroBlockInsertSafetensorExpander,
        )

        expander = ZeroBlockInsertSafetensorExpander(
            ZeroBlockInsertSafetensorConfig(
                num_new_layers=args.num_new_layers,
                insert_strategy=args.insert_strategy,
            )
        )

    elif args.expander == "overlap_copy":
        from llm_grow.safetensor.overlap_copy import (
            OverlapCopySafetensorConfig,
            OverlapCopySafetensorExpander,
        )

        expander = OverlapCopySafetensorExpander(
            OverlapCopySafetensorConfig(num_overlap=args.num_overlap)
        )

    elif args.expander == "multi_axis_pad":
        from llm_grow.safetensor.multi_axis_pad import (
            MultiAxisPadSafetensorConfig,
            MultiAxisPadSafetensorExpander,
        )

        expander = MultiAxisPadSafetensorExpander(
            MultiAxisPadSafetensorConfig(
                num_new_layers=args.num_new_layers,
                insert_strategy=args.insert_strategy,
                ffn_size_expansion=args.ffn_size_expansion,
            )
        )

    elif args.expander == "expert_clone":
        from llm_grow.safetensor.auto import auto_expand

        auto_expand(
            src_dir=args.src,
            dst_dir=args.dst,
            method="expert",
            expand_factor=args.expand_factor,
            noise_scale=args.noise_scale,
            target_shard_gb=args.target_shard_gb or 4.0,
            verbose=verbose,
            dry_run=args.dry_run,
        )
        return

    else:
        print(f"Unknown expander: {args.expander}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        expander.dry_run(args.src)
    else:
        expander.expand(
            src_dir=args.src,
            dst_dir=args.dst,
            target_shard_bytes=target_bytes,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
