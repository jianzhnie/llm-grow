#!/usr/bin/env python
"""Auto-detect and auto_expand dispatch tests for LongCat-Flash-Chat."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.helpers import print_summary
from common.model_paths import LONGCAT, require_path

MODEL_DIR = require_path("LONGCAT", LONGCAT)


def test_detect():
    from llm_grow.safetensor.detect import detect_model

    p = detect_model(MODEL_DIR)
    print(p.summary())

    assert p.family == "longcat", f"Expected 'longcat', got {p.family!r}"
    assert p.has_dual_attn is True, f"Expected has_dual_attn=True, got {p.has_dual_attn}"
    print("  [OK] Detection: family=longcat, has_dual_attn=True")
    return True


def test_auto_dispatch():
    from llm_grow.safetensor.auto import auto_expand

    scenarios = [
        ("depth", {"num_new_layers": 4}, True),
        ("expert", {"expand_factor": 2}, True),
    ]

    for method, kwargs, want_ok in scenarios:
        print(f"\n  -> auto_expand(method={method!r}, dry_run=True)")
        try:
            auto_expand(
                MODEL_DIR,
                f"/tmp/auto_test/longcat",
                method=method,
                verbose=False,
                dry_run=True,
                **kwargs,
            )
            print(f"  [OK] longcat / {method}")
        except Exception as e:
            print(f"  [FAIL] {e}")
            return False
    return True


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
