#!/usr/bin/env python
"""Auto-detect and auto_expand dispatch tests for Kimi-K2-Base (DeepSeek)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.model_paths import KIMI_K2, require_path

SRC = require_path("KIMI_K2", KIMI_K2)


def test_detect():
    from llm_grow.safetensor.detect import detect_model

    p = detect_model(SRC)
    print(p.summary())

    results = {}
    results["family"] = p.family == "deepseek_moe"
    results["is_moe"] = p.is_moe is True
    results["experts"] = p.experts_per_moe_layer == 384
    results["has_fp8"] = p.has_fp8 is True
    results["dense_layer_0"] = 0 in p.dense_only_layers

    for k, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {k}")

    return all(results.values())


def test_auto_dispatch():
    from llm_grow.safetensor.auto import auto_expand

    scenarios = [
        ("depth", {"num_new_layers": 4}, True),
        ("expert", {"expand_factor": 2}, True),
    ]

    all_ok = True
    for method, kwargs, want_ok in scenarios:
        print(f"\n  -> auto_expand(method={method!r}, dry_run=True)")
        try:
            auto_expand(
                SRC,
                f"/tmp/auto_test/kimi_k2",
                method=method,
                verbose=False,
                dry_run=True,
                **kwargs,
            )
            print(f"  [OK] kimi_k2 / {method}")
        except Exception as e:
            print(f"  [FAIL] {e}")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    from common.helpers import run_tests

    sys.exit(run_tests([
        ("detect", test_detect),
        ("auto_dispatch", test_auto_dispatch),
    ]))
