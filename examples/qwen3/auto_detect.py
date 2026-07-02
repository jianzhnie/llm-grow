#!/usr/bin/env python
"""Auto-detect and auto-dispatch example for Qwen3 Dense + MoE models.

Verifies detection of both ``dense`` (Qwen3-0.6B) and ``standard_moe``
(Qwen3-30B-A3B) families, and that auto_expand correctly dispatches
(including expected errors for invalid combinations).
"""

from __future__ import annotations

import sys

from common.helpers import run_tests
from common.model_paths import QWEN3_06B, QWEN3_30B, require_path

DENSE_SRC = require_path("QWEN3_06B", QWEN3_06B)
MOE_SRC = require_path("QWEN3_30B", QWEN3_30B)


def check_detect():
    from llm_grow.safetensor.detect import detect_model

    results = {}

    p_dense = detect_model(DENSE_SRC)
    print(p_dense.summary())
    results["dense/family"] = p_dense.family == "dense"
    results["dense/is_moe"] = p_dense.is_moe is False
    print(f"  [{'OK' if all(results.values()) else 'FAIL'}] Dense")

    p_moe = detect_model(MOE_SRC)
    print(p_moe.summary())
    results["moe/family"] = p_moe.family == "standard_moe"
    results["moe/is_moe"] = p_moe.is_moe is True
    results["moe/experts"] = p_moe.experts_per_moe_layer == 128
    print(f"  [{'OK' if results['moe/family'] else 'FAIL'}] MoE")

    return all(results.values())


def check_auto_dispatch():
    from llm_grow.safetensor.auto import auto_expand

    scenarios = [
        (DENSE_SRC, "dense", "depth", {}, True),
        (DENSE_SRC, "dense", "expert", {}, "expected"),
        (DENSE_SRC, "dense", "width", {"ffn_size_expansion": 256}, True),
        (MOE_SRC, "moe", "depth", {"num_new_layers": 4}, True),
        (MOE_SRC, "moe", "expert", {"expand_factor": 2}, True),
        (MOE_SRC, "moe", "width", {}, "expected"),
    ]

    all_ok = True
    for path, label, method, kwargs, want in scenarios:
        print(f"\n  -> auto_expand({label}, method={method!r}, dry_run=True)")
        try:
            auto_expand(
                path, f"/tmp/auto_test/{label}",
                method=method, verbose=False, dry_run=True, **kwargs,
            )
            ok = want is True
            print(f"  [{'OK' if ok else 'FAIL'}] {label} / {method}")
        except (ValueError, NotImplementedError) as e:
            ok = want == "expected"
            print(f"  [{'OK' if ok else 'FAIL'}] {label} / {method} (expected: {e})")
        except Exception as e:
            ok = False
            print(f"  [FAIL] {label} / {method}: {e}")
        all_ok = all_ok and ok

    return all_ok


if __name__ == "__main__":
    sys.exit(run_tests([
        ("detect", check_detect),
        ("auto_dispatch", check_auto_dispatch),
    ]))
