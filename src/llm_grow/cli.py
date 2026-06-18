"""CLI entry point for llm-grow.

Usage::

    llm-grow expand --method depth --src /path/to/model --dst ./output --num-new-layers 4
    llm-grow expand --method expert --src /path/to/moe --dst ./output --expand-factor 2
    llm-grow expand --method width --src /path/to/model --dst ./output --ffn-size-expansion 1024
    llm-grow verify --src /path/to/original --dst /path/to/expanded
    llm-grow info --src /path/to/model
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Main CLI entry point registered as `llm-grow` console script."""
    parser = argparse.ArgumentParser(
        prog="llm-grow",
        description="Modular toolkit for LLM parameter expansion",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── expand ────────────────────────────────────────────────────────────────
    expand_p = subparsers.add_parser("expand", help="Expand a model (safetensor-level)")
    expand_p.add_argument("--src", required=True, help="Source model directory")
    expand_p.add_argument("--dst", required=True, help="Output directory")
    expand_p.add_argument(
        "--method",
        default="depth",
        choices=["depth", "expert", "width"],
        help="Expansion method (default: depth)",
    )
    expand_p.add_argument(
        "--num-new-layers", type=int, default=4, help="[depth/width] Layers to insert"
    )
    expand_p.add_argument(
        "--insert-strategy", default="uniform", choices=["uniform", "front", "rear"]
    )
    expand_p.add_argument(
        "--expand-factor", type=int, default=2, help="[expert] Expert count multiplier"
    )
    expand_p.add_argument(
        "--noise-scale", type=float, default=1e-6, help="[expert] Router noise scale"
    )
    expand_p.add_argument(
        "--ffn-size-expansion",
        type=int,
        default=0,
        help="[width] intermediate_size increment",
    )
    expand_p.add_argument(
        "--target-shard-gb", type=float, default=4.0, help="Output shard size in GB"
    )
    expand_p.add_argument(
        "--workers", type=int, default=1, help="Parallel writer processes"
    )
    expand_p.add_argument(
        "--dry-run", action="store_true", help="Plan only, no file writes"
    )
    expand_p.add_argument("--quiet", action="store_true")

    # ── verify ────────────────────────────────────────────────────────────────
    verify_p = subparsers.add_parser("verify", help="Verify an expanded model")
    verify_p.add_argument("--src", required=True, help="Original model directory")
    verify_p.add_argument("--dst", required=True, help="Expanded model directory")
    verify_p.add_argument(
        "--fp", action="store_true", help="Run full FP logit check (loads models)"
    )
    verify_p.add_argument("--fp-atol", type=float, default=1e-4)

    # ── info ──────────────────────────────────────────────────────────────────
    info_p = subparsers.add_parser("info", help="Detect and display model architecture")
    info_p.add_argument("--src", required=True, help="Model directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "expand":
        _cmd_expand(args)
    elif args.command == "verify":
        _cmd_verify(args)
    elif args.command == "info":
        _cmd_info(args)


def _cmd_expand(args) -> None:
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
        verbose=not args.quiet,
        dry_run=args.dry_run,
        workers=args.workers,
    )


def _cmd_verify(args) -> None:
    from pathlib import Path

    from llm_grow.eval.structural import StructuralVerifier

    verifier = StructuralVerifier(src_dir=args.src, dst_dir=args.dst)
    results = verifier.run_all()

    if args.fp:
        from llm_grow.eval.structural import check_fp

        results["fp_logit_check"] = check_fp(
            Path(args.src),
            Path(args.dst),
            seq_len=32,
            samples=4,
            atol=args.fp_atol,
        )

    all_ok = all(results.values())
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    for name, ok in results.items():
        icon = "pass" if ok else "FAIL"
        print(f"  [{icon}] {name}")

    sys.exit(0 if all_ok else 1)


def _cmd_info(args) -> None:
    from llm_grow.safetensor.detect import detect_model

    profile = detect_model(args.src)
    print(profile.summary())


if __name__ == "__main__":
    main()
