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

# Explicit expanders (legacy)
python scripts/safetensor_expand.py llama_pro \\
    --src  /path/to/Qwen3-8B \\
    --dst  ./outputs/qwen3_llama_pro \\
    --num-new-blocks 7

python scripts/safetensor_expand.py solar_dus \\
    --src  /path/to/Qwen3-8B \\
    --dst  ./outputs/qwen3_solar_dus \\
    --num-overlap 8

python scripts/safetensor_expand.py msg \\
    --src  /path/to/Qwen3-8B \\
    --dst  ./outputs/qwen3_msg \\
    --depth-expansion 4 \\
    --ffn-size-expansion 1024

python scripts/safetensor_expand.py moe_expert \\
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
        choices=["auto", "llama_pro", "solar_dus", "msg", "moe_expert"],
        help=(
            "auto       : detect Dense/MoE and select the right method\n"
            "llama_pro  : identity block insertion (Dense depth)\n"
            "solar_dus  : layer overlap-copy (Dense depth)\n"
            "msg        : depth + FFN-width (Dense)\n"
            "moe_expert : expert count expansion (MoE only)"
        ),
    )
    p.add_argument("--src", required=True, help="Source model directory")
    p.add_argument("--dst", required=True, help="Output directory")
    p.add_argument("--dry-run", action="store_true", help="Build plan without writing files")
    p.add_argument("--target-shard-gb", type=float, default=4.0)
    p.add_argument("--quiet", action="store_true")

    # auto params
    p.add_argument(
        "--method",
        default="depth",
        choices=["depth", "expert", "width"],
        help="[auto] Expansion axis",
    )

    # depth / LLaMA-Pro / MSG
    p.add_argument(
        "--num-new-layers",
        "--num-new-blocks",
        type=int,
        default=4,
        help="[auto depth / llama_pro / msg] layers / blocks to insert",
    )
    p.add_argument("--insert-strategy", default="uniform", choices=["uniform", "front", "rear"])

    # SOLAR DUS
    p.add_argument("--num-overlap", type=int, default=8, help="[solar_dus] overlapping layers")

    # MSG width
    p.add_argument(
        "--ffn-size-expansion",
        type=int,
        default=0,
        help="[msg / auto width] intermediate_size increment",
    )

    # expert
    p.add_argument(
        "--expand-factor",
        type=int,
        default=2,
        help="[auto expert / moe_expert] expert count multiplier",
    )
    p.add_argument(
        "--noise-scale",
        type=float,
        default=1e-6,
        help="[auto expert / moe_expert] router noise scale",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    target_bytes = int(args.target_shard_gb * 1024**3)
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
            target_shard_gb=args.target_shard_gb,
            verbose=verbose,
            dry_run=args.dry_run,
        )
        return

    # ── explicit expanders ────────────────────────────────────────────────────
    if args.expander == "llama_pro":
        from llm_grow.safetensor.llama_pro import (
            LlamaProSafetensorConfig,
            LlamaProSafetensorExpander,
        )

        expander = LlamaProSafetensorExpander(
            LlamaProSafetensorConfig(
                num_new_blocks=args.num_new_layers,
                insert_strategy=args.insert_strategy,
            )
        )

    elif args.expander == "solar_dus":
        from llm_grow.safetensor.solar_dus import (
            SolarDUSSafetensorConfig,
            SolarDUSSafetensorExpander,
        )

        expander = SolarDUSSafetensorExpander(SolarDUSSafetensorConfig(num_overlap=args.num_overlap))

    elif args.expander == "msg":
        from llm_grow.safetensor.msg import MSGSafetensorConfig, MSGSafetensorExpander

        expander = MSGSafetensorExpander(
            MSGSafetensorConfig(
                depth_expansion=args.num_new_layers,
                insert_strategy=args.insert_strategy,
                ffn_size_expansion=args.ffn_size_expansion,
            )
        )

    elif args.expander == "moe_expert":
        from llm_grow.safetensor.auto import auto_expand

        auto_expand(
            src_dir=args.src,
            dst_dir=args.dst,
            method="expert",
            expand_factor=args.expand_factor,
            noise_scale=args.noise_scale,
            target_shard_gb=args.target_shard_gb,
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
