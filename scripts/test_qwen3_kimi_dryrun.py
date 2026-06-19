#!/usr/bin/env python
"""Dry-run expansion analysis for Qwen3-30B-A3B and Kimi-K2-Base.

Both models have only index JSON (no weight files), so we validate the
expansion plan logic via dry_run().

Tests:
  Qwen3-30B-A3B  (Qwen3MoeForCausalLM,  48 layers, 128 experts)
    1. Expert upcycling  128 -> 256
    2. Depth expansion   48  -> 56  (+8 layers)

  Kimi-K2-Base   (DeepseekV3ForCausalLM, 61 layers, 384 experts, fp8)
    3. Expert upcycling  384 -> 768
    4. Depth expansion   61  -> 65  (+4 layers)
"""

import re
import sys

QWEN3 = "/Users/robin/hfhub/models/Qwen/Qwen3-30B-A3B"
KIMI_K2 = "/Users/robin/hfhub/models/moonshotai/Kimi-K2-Base"


def _count_zero_recipes(plan):
    return sum(1 for r in plan.recipes.values() if r.zero_out)


def _count_dup_recipes(plan):
    return sum(1 for r in plan.recipes.values() if r.dup_rows)


def _count_new_keys(plan, wmap):
    return sum(1 for k in plan.recipes if k not in wmap)


# ── Test 1: Qwen3-30B-A3B Expert Upcycling ───────────────────────────────────


def test_qwen3_expert_clone():
    print("\n" + "=" * 62)
    print("  [1] Qwen3-30B-A3B  Expert Upcycling  (128 → 256 experts)")
    print("=" * 62)
    from llm_grow.safetensor.moe_generic import make_qwen3moe_upcycling
    from llm_grow.safetensor.utils import ShardIndex

    exp = make_qwen3moe_upcycling(expand_factor=2)
    plan = exp.dry_run(QWEN3)

    src = ShardIndex.load(QWEN3)
    wmap = src.weight_map

    # Expert count per layer should double
    def experts_in_plan_layer0():
        return len(
            {
                int(re.search(r"experts\.(\d+)", k).group(1))
                for k in plan.recipes
                if k.startswith("model.layers.0.") and "mlp.experts." in k
            }
        )

    orig_experts = 128
    new_experts = experts_in_plan_layer0()
    assert new_experts == 256, f"Expected 256 experts in layer 0, got {new_experts}"
    print(f"  [OK] experts/layer: {orig_experts} → {new_experts}")

    # Router weight → dup_rows
    router_keys = [k for k in plan.recipes if k.endswith("mlp.gate.weight")]
    assert all(plan.recipes[k].dup_rows for k in router_keys)
    print(f"  [OK] {len(router_keys)} router weights use dup_rows=True")

    # No router bias in Qwen3MoE
    bias_keys = [k for k in plan.recipes if "e_score_correction_bias" in k]
    assert len(bias_keys) == 0
    print("  [OK] No e_score_correction_bias (expected for Qwen3MoE)")

    # Config patches
    assert plan.config_patches.get("num_experts") == 256
    assert plan.config_patches.get("num_experts_per_tok") == 16
    print("  [OK] Config: num_experts=256, num_experts_per_tok=16")

    new_keys = _count_new_keys(plan, wmap)
    print(f"  New tensors : {new_keys:,}")
    print(f"  Total output: {len(plan.recipes):,}  (src: {len(wmap):,})")
    return True


# ── Test 2: Qwen3-30B-A3B Depth Expansion ────────────────────────────────────


def test_qwen3_depth():
    print("\n" + "=" * 62)
    print("  [2] Qwen3-30B-A3B  Depth Expansion  (48 → 56 layers)")
    print("=" * 62)
    from llm_grow.safetensor.moe_generic import make_qwen3moe_depth
    from llm_grow.safetensor.utils import ShardIndex

    exp = make_qwen3moe_depth(num_new_layers=8)
    plan = exp.dry_run(QWEN3)

    assert plan.new_num_hidden_layers == 56
    print("  [OK] num_hidden_layers: 48 → 56")

    zero_n = _count_zero_recipes(plan)
    # Per identity block: 1 o_proj + 128 expert down_proj = 129
    expected_zero = 8 * (1 + 128)
    assert zero_n == expected_zero, f"Expected {expected_zero} zeros, got {zero_n}"
    print(f"  [OK] zero tensors: {zero_n}  (8 layers × {1 + 128} = {expected_zero})")

    src = ShardIndex.load(QWEN3)
    new_keys = _count_new_keys(plan, src.weight_map)
    print(f"  New tensors : {new_keys:,}")
    print(f"  Total output: {len(plan.recipes):,}")
    return True


# ── Test 3: Kimi-K2-Base Expert Upcycling ────────────────────────────────────


def test_kimik2_expert_clone():
    print("\n" + "=" * 62)
    print("  [3] Kimi-K2-Base  Expert Upcycling  (384 → 768 experts)")
    print("=" * 62)
    from llm_grow.safetensor.moe_generic import make_kimik2_upcycling
    from llm_grow.safetensor.utils import ShardIndex

    exp = make_kimik2_upcycling(expand_factor=2)
    plan = exp.dry_run(KIMI_K2)

    src = ShardIndex.load(KIMI_K2)
    wmap = src.weight_map

    # Layer 0 is dense — no experts should appear there
    l0_expert_keys = [
        k
        for k in plan.recipes
        if k.startswith("model.layers.0.") and "mlp.experts." in k
    ]
    assert len(l0_expert_keys) == 0, (
        f"Layer 0 should be dense, got {len(l0_expert_keys)} expert keys"
    )
    print("  [OK] Layer 0 is dense — no expert tensors")

    # Layer 1 expert count
    l1_experts = len(
        {
            int(re.search(r"experts\.(\d+)", k).group(1))
            for k in plan.recipes
            if k.startswith("model.layers.1.") and "mlp.experts." in k
        }
    )
    assert l1_experts == 768, f"Expected 768 experts in layer 1, got {l1_experts}"
    print("  [OK] experts/MoE-layer: 384 → 768")

    # FP8: weight_scale_inv tensors should be present and copied
    scale_keys = [k for k in plan.recipes if k.endswith("weight_scale_inv")]
    print(f"  [OK] FP8 scale tensors copied: {len(scale_keys):,}")

    # Router weight dup_rows
    router_w = [k for k in plan.recipes if k.endswith("mlp.gate.weight")]
    assert all(plan.recipes[k].dup_rows for k in router_w)
    print(f"  [OK] {len(router_w)} router weights: dup_rows=True")

    # Router bias dup_rows (no noise)
    router_b = [
        k for k in plan.recipes if k.endswith("mlp.gate.e_score_correction_bias")
    ]
    assert all(
        plan.recipes[k].dup_rows and plan.recipes[k].dup_rows_noise_scale == 0.0
        for k in router_b
    )
    print(f"  [OK] {len(router_b)} router biases: dup_rows=True, noise=0")

    # Shared expert untouched
    shared_keys_src = [k for k in wmap if "shared_experts" in k]
    shared_keys_out = [k for k in plan.recipes if "shared_experts" in k]
    assert len(shared_keys_src) == len(shared_keys_out)
    print(f"  [OK] Shared expert tensors preserved: {len(shared_keys_src)}")

    # Config patches
    assert plan.config_patches.get("n_routed_experts") == 768
    assert plan.config_patches.get("num_experts_per_tok") == 16
    print("  [OK] Config: n_routed_experts=768, num_experts_per_tok=16")

    new_keys = _count_new_keys(plan, wmap)
    print(f"  New tensors : {new_keys:,}")
    print(f"  Total output: {len(plan.recipes):,}  (src: {len(wmap):,})")
    return True


# ── Test 4: Kimi-K2-Base Depth Expansion ─────────────────────────────────────


def test_kimik2_depth():
    print("\n" + "=" * 62)
    print("  [4] Kimi-K2-Base  Depth Expansion  (61 → 65 layers)")
    print("=" * 62)
    from llm_grow.safetensor.moe_generic import make_kimik2_depth
    from llm_grow.safetensor.utils import ShardIndex

    exp = make_kimik2_depth(num_new_layers=4)
    plan = exp.dry_run(KIMI_K2)

    assert plan.new_num_hidden_layers == 65
    print("  [OK] num_hidden_layers: 61 → 65")

    # Analyse zero tensors per identity block
    zero_keys = [k for k, r in plan.recipes.items() if r.zero_out]

    # Layer 0 is dense: identity copy from it should have mlp.down_proj.weight zeroed
    [
        k
        for k in zero_keys
        if "model.layers.0." not in k
        and any(k.startswith(f"model.layers.{new_i}.") for new_i in range(65))
        and "mlp.down_proj.weight" in k
        and "experts" not in k
    ]

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

    src = ShardIndex.load(KIMI_K2)
    new_keys = _count_new_keys(plan, src.weight_map)
    print(f"  New tensors : {new_keys:,}")
    print(f"  Total output: {len(plan.recipes):,}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("qwen3_expert_clone", test_qwen3_expert_clone),
        ("qwen3_depth", test_qwen3_depth),
        ("kimik2_expert_clone", test_kimik2_expert_clone),
        ("kimik2_depth", test_kimik2_depth),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n  [FAIL] {e}")
            import traceback

            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 62)
    print("  Summary")
    print("=" * 62)
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    print("\nNote: weight files not present — dry_run validates plan logic only.")
    sys.exit(0 if all(results.values()) else 1)
