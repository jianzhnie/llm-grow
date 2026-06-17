#!/usr/bin/env python
"""Safetensor-level model expansion — no full model load required.

Usage examples
--------------
# LLaMA-Pro: insert 7 identity blocks (uniform)
python scripts/safetensor_expand.py llama_pro \\
    --src  /path/to/Qwen3-8B \\
    --dst  ./outputs/qwen3_llama_pro \\
    --num-new-blocks 7

# SOLAR DUS: depth up-scaling with overlap=8
python scripts/safetensor_expand.py solar_dus \\
    --src  /path/to/Qwen3-8B \\
    --dst  ./outputs/qwen3_solar_dus \\
    --num-overlap 8

# MSG: depth=4 new blocks + FFN width +1024
python scripts/safetensor_expand.py msg \\
    --src  /path/to/Qwen3-8B \\
    --dst  ./outputs/qwen3_msg \\
    --depth-expansion 4 \\
    --ffn-size-expansion 1024

# Override default 4GB shard size (e.g. 2GB for smaller machines)
python scripts/safetensor_expand.py llama_pro \\
    --src  ... --dst  ... \\
    --target-shard-gb 2
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Safetensor-level LLM expansion (no full model load)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("method", choices=["llama_pro", "solar_dus", "msg"],
                   help="Expansion method")
    p.add_argument("--src", required=True, help="Source model directory")
    p.add_argument("--dst", required=True, help="Output directory")
    p.add_argument("--target-shard-gb", type=float, default=4.0,
                   help="Target output shard size in GB (default: 4)")
    p.add_argument("--quiet", action="store_true")

    # LLaMA-Pro / MSG depth
    p.add_argument("--num-new-blocks", type=int, default=8,
                   help="[llama_pro / msg] Identity blocks to insert")
    p.add_argument("--insert-strategy", default="uniform",
                   choices=["uniform", "front", "rear"],
                   help="[llama_pro / msg] Block insertion strategy")

    # SOLAR DUS
    p.add_argument("--num-overlap", type=int, default=8,
                   help="[solar_dus] Overlapping layers")

    # MSG width
    p.add_argument("--ffn-size-expansion", type=int, default=0,
                   help="[msg] intermediate_size increment per layer")
    return p


def main() -> None:
    args = build_parser().parse_args()
    target_bytes = int(args.target_shard_gb * 1024 ** 3)

    if args.method == "llama_pro":
        from llm_grow.safetensor.llama_pro import (
            LlamaProSafetensorConfig, LlamaProSafetensorExpander,
        )
        cfg = LlamaProSafetensorConfig(
            num_new_blocks=args.num_new_blocks,
            insert_strategy=args.insert_strategy,
        )
        expander = LlamaProSafetensorExpander(cfg)

    elif args.method == "solar_dus":
        from llm_grow.safetensor.solar_dus import (
            SolarDUSSafetensorConfig, SolarDUSSafetensorExpander,
        )
        cfg = SolarDUSSafetensorConfig(num_overlap=args.num_overlap)
        expander = SolarDUSSafetensorExpander(cfg)

    elif args.method == "msg":
        from llm_grow.safetensor.msg import MSGSafetensorConfig, MSGSafetensorExpander
        cfg = MSGSafetensorConfig(
            depth_expansion=args.num_new_blocks,
            insert_strategy=args.insert_strategy,
            ffn_size_expansion=args.ffn_size_expansion,
        )
        expander = MSGSafetensorExpander(cfg)

    else:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)

    expander.expand(
        src_dir=args.src,
        dst_dir=args.dst,
        target_shard_bytes=target_bytes,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
