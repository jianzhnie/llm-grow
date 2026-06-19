#!/usr/bin/env python
"""Auto-detect and auto_expand dispatch tests for Qwen3 models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.helpers import print_summary
from common.model_paths import QWEN3_06B, QWEN3_30B, require_path

DENSE_SRC = require_path("QWEN3_06B", QWEN3_06B)
MOE_SRC = require_path("QWEN3_30B", QWEN3_30B)


def test_detect():
    from llm_grow.safetensor.detect import detect_model

    results = {}

    p_dense = detect_model(DENSE_SRC)
    print(p_dense.summary())
    results["dense/family"] = p_dense.family == "dense"
    results["dense/is_moe"] = p_dense.is_moe is False
    print(f"  [{'OK' if all(results.values()) else 'FAIL'}] Dense: family={p_dense.family}, is_moe={p_dense.is_moe}")

    p_moe = detect_model(MOE_SRC)
    print(p_moe.summary())
    results["moe/family"] = p_moe.family == "standard_moe"
    results["moe/is_moe"] = p_moe.is_moe is True
    results["moe/experts"] = p_moe.experts_per_moe_layer == 128
    print(f"  [{'OK' if results['moe/family'] else 'FAIL'}] MoE: family={p_moe.family}, experts={p_moe.experts_per_moe_layer}")

    return all(results.values())


def test_auto_dispatch():
    from llm_grow.safetensor.auto import auto_expand

    scenarios = [
        (DENSE_SRC, "dense_qwen3", "depth", {}, True),
        (DENSE_SRC, "dense_qwen3", "expert", {}, "expected"),
        (DENSE_SRC, "dense_qwen3", "width", {"ffn_size_expansion": 256}, True),
        (MOE_SRC, "moe_qwen3", "depth", {"num_new_layers": 4}, True),
        (MOE_SRC, "moe_qwen3", "expert", {"expand_factor": 2}, True),
        (MOE_SRC, "moe_qwen3", "width", {}, "expected"),
    ]

    all_ok = True
    for path, label, method, kwargs, want in scenarios:
        print(f"\n  -> auto_expand({label}, method={method!r}, dry_run=True)")
        try:
            auto_expand(
                path,
                f"/tmp/auto_test/{label}",
                method=method,
                verbose=False,
                dry_run=True,
                **kwargs,
            )
            ok = want is True
            print(f"  [{'OK' if ok else 'FAIL'}] {label} / {method}")
        except (ValueError, NotImplementedError) as e:
            ok = want == "expected"
            print(f"  [{'OK' if ok else 'FAIL'}] {label} / {method} (expected error: {e})")
        except Exception as e:
            ok = False
            print(f"  [FAIL] {label} / {method}: {e}")
        all_ok = all_ok and ok

    return all_ok


if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("detect", test_detect),
        ("auto_dispatch", test_auto_dispatch),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    sys.exit(print_summary(results))
