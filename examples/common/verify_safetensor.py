#!/usr/bin/env python
"""Verify an expanded safetensor model against its source.

Leverages :class:`llm_grow.eval.structural.StructuralVerifier` for
fast structural checks (no model load) and optionally loads both
models for a full function-preserving logit comparison (``--fp``).

Usage
-----
# Fast structural checks only (no model load, works for 100B+ models)
python examples/common/verify_safetensor.py \\
    --src /path/to/original --dst /path/to/expanded

# Full FP verification (loads both models — requires < ~80 GB RAM)
python examples/common/verify_safetensor.py \\
    --src /path/to/original --dst /path/to/expanded --fp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_grow.eval.structural import StructuralVerifier, check_fp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify expanded safetensor model")
    p.add_argument("--src", required=True, help="Original model directory")
    p.add_argument("--dst", required=True, help="Expanded model directory")
    p.add_argument(
        "--fp",
        action="store_true",
        help="Load both models and run function-preserving logit check",
    )
    p.add_argument("--fp-atol", type=float, default=1e-4)
    return p


def main() -> None:
    args = build_parser().parse_args()
    src_dir, dst_dir = Path(args.src), Path(args.dst)

    print(f"src: {src_dir}")
    print(f"dst: {dst_dir}\n")

    verifier = StructuralVerifier(src_dir=src_dir, dst_dir=dst_dir)
    results = verifier.run_all()

    if args.fp:
        results["fp_logit_check"] = check_fp(
            src_dir, dst_dir, atol=args.fp_atol,
        )

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    for name, ok in results.items():
        icon = "pass" if ok else "FAIL"
        print(f"  [{icon}] {name}")

    all_ok = all(results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
