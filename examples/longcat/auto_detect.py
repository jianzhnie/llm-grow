#!/usr/bin/env python
"""Auto-detect and auto_expand dispatch example for LongCat-Flash-Chat."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.model_paths import LONGCAT, require_path

MODEL_DIR = require_path("LONGCAT", LONGCAT)


def check_detect():
    from llm_grow.safetensor.detect import detect_model

    p = detect_model(MODEL_DIR)
    print(p.summary())

    assert p.family == "longcat", f"Expected 'longcat', got {p.family!r}"
    assert p.has_dual_attn is True, (
        f"Expected has_dual_attn=True, got {p.has_dual_attn}"
    )
    print("  [OK] Detection: family=longcat, has_dual_attn=True")
    return True


def check_auto_dispatch():
    from llm_grow.safetensor.auto import auto_expand

    scenarios = [
        ("depth", {"num_new_layers": 4}, True),
        ("expert", {"expand_factor": 2}, True),
    ]

    for method, kwargs, _want_ok in scenarios:
        print(f"\n  -> auto_expand(method={method!r}, dry_run=True)")
        try:
            auto_expand(
                MODEL_DIR,
                "/tmp/auto_test/longcat",
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
    from common.helpers import run_tests

    sys.exit(
        run_tests(
            [
                ("detect", check_detect),
                ("auto_dispatch", check_auto_dispatch),
            ]
        )
    )
