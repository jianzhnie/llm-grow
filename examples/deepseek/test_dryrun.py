#!/usr/bin/env python
"""Dry-run expansion tests for Kimi-K2-Base (DeepseekV3ForCausalLM).

Tests (no weight files needed, index JSON only):
  1. Expert upcycling  384 -> 768
  2. Depth expansion   61  -> 65  (+4 layers)
  3. Dry-run plan verification via auto-detect
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.helpers import count_new_keys, count_zero_recipes, print_summary
from common.model_paths import KIMI_K2, require_path

SRC = require_path("KIMI_K2", KIMI_K2)


def test_expert_clone():
    print("\n" + "=" * 62)
    print("  [1] Kimi-K2-Base  Expert Upcycling  (384 -> 768 experts)")
    print("=" * 62)
    from llm_grow.safetensor.moe_generic import make_kimik2_expert_clone
    from llm_grow.safetensor.utils import ShardIndex

    exp = make_kimik2_expert_clone(expand_factor=2)
    plan = exp.dry_run(SRC)

    src = ShardIndex.load(SRC)
    wmap = src.weight_map

    l0_expert_keys = [
        k
        for k in plan.recipes
        if k.startswith("model.layers.0.") and "mlp.experts." in k
    ]
    assert len(l0_expert_keys) == 0, (
        f"Layer 0 should be dense, got {len(l0_expert_keys)} expert keys"
    )
    print("  [OK] Layer 0 is dense — no expert tensors")

    l1_experts = len(
        {
            int(re.search(r"experts\.(\d+)", k).group(1))
            for k in plan.recipes
            if k.startswith("model.layers.1.") and "mlp.experts." in k
        }
    )
    assert l1_experts == 768, f"Expected 768 experts in layer 1, got {l1_experts}"
    print("  [OK] experts/MoE-layer: 384 -> 768")

    scale_keys = [k for k in plan.recipes if k.endswith("weight_scale_inv")]
    print(f"  [OK] FP8 scale tensors copied: {len(scale_keys):,}")

    router_w = [k for k in plan.recipes if k.endswith("mlp.gate.weight")]
    assert all(plan.recipes[k].dup_rows for k in router_w)
    print(f"  [OK] {len(router_w)} router weights: dup_rows=True")

    router_b = [
        k for k in plan.recipes if k.endswith("mlp.gate.e_score_correction_bias")
    ]
    assert all(
        plan.recipes[k].dup_rows and plan.recipes[k].dup_rows_noise_scale == 0.0
        for k in router_b
    )
    print(f"  [OK] {len(router_b)} router biases: dup_rows=True, noise=0")

    shared_keys_src = [k for k in wmap if "shared_experts" in k]
    shared_keys_out = [k for k in plan.recipes if "shared_experts" in k]
    assert len(shared_keys_src) == len(shared_keys_out)
    print(f"  [OK] Shared expert tensors preserved: {len(shared_keys_src)}")

    assert plan.config_patches.get("n_routed_experts") == 768
    assert plan.config_patches.get("num_experts_per_tok") == 16
    print("  [OK] Config: n_routed_experts=768, num_experts_per_tok=16")

    new_keys = count_new_keys(plan, wmap)
    print(f"  New tensors : {new_keys:,}")
    print(f"  Total output: {len(plan.recipes):,}  (src: {len(wmap):,})")
    return True


def test_depth():
    print("\n" + "=" * 62)
    print("  [2] Kimi-K2-Base  Depth Expansion  (61 -> 65 layers)")
    print("=" * 62)
    from llm_grow.safetensor.moe_generic import make_kimik2_zero_block_insert
    from llm_grow.safetensor.utils import ShardIndex

    exp = make_kimik2_zero_block_insert(num_new_layers=4)
    plan = exp.dry_run(SRC)

    assert plan.new_num_hidden_layers == 65
    print("  [OK] num_hidden_layers: 61 -> 65")

    zero_keys = [k for k, r in plan.recipes.items() if r.zero_out]

    print("  Zero breakdown:")
    o_proj_n = sum(1 for k in zero_keys if "o_proj.weight" in k and "scale" not in k)
    expert_n = sum(1 for k in zero_keys if "experts" in k and "down_proj.weight" in k)
    shared_n = sum(1 for k in zero_keys if "shared_experts.down_proj" in k)
    dense_n = sum(
        1
        for k in zero_keys
        if "mlp.down_proj.weight" in k and "experts" not in k and "shared" not in k
    )
    print(f"    o_proj zeros    : {o_proj_n}")
    print(f"    expert dp zeros : {expert_n}")
    print(f"    shared dp zeros : {shared_n}")
    print(f"    dense dp zeros  : {dense_n}")
    print(f"    Total zero      : {len(zero_keys)}")

    src = ShardIndex.load(SRC)
    new_keys = count_new_keys(plan, src.weight_map)
    print(f"  New tensors : {new_keys:,}")
    print(f"  Total output: {len(plan.recipes):,}")
    return True


def test_dryrun_plan():
    print("\n" + "=" * 62)
    print("  [3] Kimi-K2-Base  Dry-run plan verification")
    print("=" * 62)
    from llm_grow.safetensor.auto import _build_expander
    from llm_grow.safetensor.detect import detect_model
    from llm_grow.safetensor.utils import ShardIndex

    checks = [
        ("expert", {"expand_factor": 2}, {"n_routed_experts": 768}),
    ]

    for method, kwargs, expected_patches in checks:
        profile = detect_model(SRC)
        exp = _build_expander(
            method,
            profile,
            kwargs.get("num_new_layers", 4),
            "uniform",
            kwargs.get("expand_factor", 2),
            1e-6,
            0,
        )
        plan = exp._build_plan(ShardIndex.load(SRC))

        ok = all(
            plan.config_patches.get(k) == v for k, v in expected_patches.items()
        )
        print(f"  [{'OK' if ok else 'FAIL'}] Kimi-K2/{method} config: {plan.config_patches}")

        src_count = len(ShardIndex.load(SRC).all_keys)
        ok2 = len(plan.recipes) > src_count
        print(f"  [{'OK' if ok2 else 'FAIL'}] tensors: {src_count} -> {len(plan.recipes)}")

    return True


if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("expert_clone", test_expert_clone),
        ("depth", test_depth),
        ("dryrun_plan", test_dryrun_plan),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    sys.exit(print_summary(results))
