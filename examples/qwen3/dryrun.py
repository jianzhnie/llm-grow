#!/usr/bin/env python
"""Dry-run expansion example for Qwen3-30B-A3B (Qwen3MoeForCausalLM).

Tests (no weight files needed, index JSON only):
  1. Expert upcycling  128 -> 256
  2. Depth expansion   48  -> 56  (+8 layers)
  3. Dry-run plan verification via auto-detect
"""

from __future__ import annotations

import sys

from common.helpers import (
    count_experts_in_layer,
    count_new_keys,
    count_zero_recipes,
    run_tests,
    verify_dryrun_plan,
)
from common.model_paths import QWEN3_30B, require_path

SRC = require_path("QWEN3_30B", QWEN3_30B)


def check_expert_clone():
    print("\n" + "=" * 62)
    print("  [1] Qwen3-30B-A3B  Expert Upcycling  (128 -> 256 experts)")
    print("=" * 62)
    from llm_grow.safetensor.models.moe_generic import make_qwen3moe_expert_clone
    from llm_grow.safetensor.utils import ShardIndex

    exp = make_qwen3moe_expert_clone(expand_factor=2)
    plan = exp.dry_run(SRC)

    src = ShardIndex.load(SRC)
    wmap = src.weight_map

    new_experts = count_experts_in_layer(plan, 0)
    assert new_experts == 256, f"Expected 256 experts in layer 0, got {new_experts}"
    print(f"  [OK] experts/layer: 128 -> {new_experts}")

    router_keys = [k for k in plan.recipes if k.endswith("mlp.gate.weight")]
    assert all(plan.recipes[k].dup_rows for k in router_keys)
    print(f"  [OK] {len(router_keys)} router weights use dup_rows=True")

    bias_keys = [k for k in plan.recipes if "e_score_correction_bias" in k]
    assert len(bias_keys) == 0
    print("  [OK] No e_score_correction_bias (expected for Qwen3MoE)")

    assert plan.config_patches.get("num_experts") == 256
    assert plan.config_patches.get("num_experts_per_tok") == 16
    print("  [OK] Config: num_experts=256, num_experts_per_tok=16")

    new_keys = count_new_keys(plan, wmap)
    print(f"  New tensors : {new_keys:,}")
    print(f"  Total output: {len(plan.recipes):,}  (src: {len(wmap):,})")
    return True


def check_depth():
    print("\n" + "=" * 62)
    print("  [2] Qwen3-30B-A3B  Depth Expansion  (48 -> 56 layers)")
    print("=" * 62)
    from llm_grow.safetensor.models.moe_generic import make_qwen3moe_zero_block_insert
    from llm_grow.safetensor.utils import ShardIndex

    exp = make_qwen3moe_zero_block_insert(num_new_layers=8)
    plan = exp.dry_run(SRC)

    assert plan.new_num_hidden_layers == 56
    print("  [OK] num_hidden_layers: 48 -> 56")

    zero_n = count_zero_recipes(plan)
    expected_zero = 8 * (1 + 128)
    assert zero_n == expected_zero, f"Expected {expected_zero} zeros, got {zero_n}"
    print(f"  [OK] zero tensors: {zero_n}  (8 layers x {1 + 128} = {expected_zero})")

    src = ShardIndex.load(SRC)
    new_keys = count_new_keys(plan, src.weight_map)
    print(f"  New tensors : {new_keys:,}")
    print(f"  Total output: {len(plan.recipes):,}")
    return True


def check_dryrun_plan():
    print("\n" + "=" * 62)
    print("  [3] Qwen3-30B-A3B  Dry-run plan verification")
    print("=" * 62)

    return verify_dryrun_plan(
        SRC,
        "Qwen3-30B",
        [
            ("expert", {"expand_factor": 2}, {"num_experts": 256}),
            ("depth", {"num_new_layers": 4}, {}),
        ],
    )


if __name__ == "__main__":
    sys.exit(
        run_tests(
            [
                ("expert_clone", check_expert_clone),
                ("depth", check_depth),
                ("dryrun_plan", check_dryrun_plan),
            ]
        )
    )
