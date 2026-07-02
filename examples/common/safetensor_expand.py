#!/usr/bin/env python
"""Safetensor-level model expansion — no full model load required.

All expansion is dispatched through :func:`auto_expand`, which auto-detects
the model architecture (Dense / MoE / LongCat / DeepSeek-MoE) and selects
the correct expander automatically.

Usage examples
--------------
# Depth expansion (Dense or MoE)
python examples/common/safetensor_expand.py \\
    --src /path/to/model --dst ./output \\
    --method depth --num-new-layers 4

# Expert upcycling (MoE only)
python examples/common/safetensor_expand.py \\
    --src /path/to/moe_model --dst ./output \\
    --method expert --expand-factor 2

# Width expansion (Dense only)
python examples/common/safetensor_expand.py \\
    --src /path/to/model --dst ./output \\
    --method width --ffn-size-expansion 1024

# Dry-run only — build plan without writing files
python examples/common/safetensor_expand.py \\
    --src /path/to/model --dst ./output \\
    --method depth --num-new-layers 4 --dry-run

# With parallel writing and output validation
python examples/common/safetensor_expand.py \\
    --src /path/to/model --dst ./output \\
    --method depth --num-new-layers 4 --workers 4 --validate-output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_grow.safetensor.auto import auto_expand


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Safetensor-level LLM expansion (auto-detect + dispatch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src", required=True, help="Source model directory")
    p.add_argument("--dst", required=True, help="Output directory")
    p.add_argument(
        "--method",
        default="depth",
        choices=["depth", "expert", "width"],
        help="Expansion axis (default: depth)",
    )

    # Depth params
    p.add_argument("--num-new-layers", type=int, default=4, help="Layers to insert")
    p.add_argument(
        "--insert-strategy", default="uniform", choices=["uniform", "front", "rear"]
    )

    # Expert params
    p.add_argument(
        "--expand-factor", type=int, default=2, help="Expert count multiplier"
    )
    p.add_argument(
        "--noise-scale", type=float, default=1e-6, help="Router noise scale"
    )

    # Width params
    p.add_argument(
        "--ffn-size-expansion",
        type=int,
        default=0,
        help="intermediate_size increment",
    )

    # I/O
    p.add_argument("--dry-run", action="store_true", help="Plan only, no file writes")
    p.add_argument(
        "--target-shard-gb", type=float, default=4.0, help="Output shard size in GB"
    )
    p.add_argument("--workers", type=int, default=1, help="Parallel writer threads")
    p.add_argument(
        "--validate-output",
        action="store_true",
        help="Verify output config and tensor keys after writing",
    )
    p.add_argument("--resume", action="store_true", help="Resume interrupted run")
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not Path(args.src).exists():
        print(f"Error: source directory not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    auto_expand(
        src_dir=args.src,
        dst_dir=args.dst,
        method=args.method,
        num_new_layers=args.num_new_layers,
        insert_strategy=args.insert_strategy,
        expand_factor=args.expand_factor,
        noise_scale=args.noise_scale,
        ffn_size_expansion=args.ffn_size_expansion,
        target_shard_gb=args.target_shard_gb,
        verbose=not args.quiet,
        dry_run=args.dry_run,
        workers=args.workers,
        validate_output=args.validate_output,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
