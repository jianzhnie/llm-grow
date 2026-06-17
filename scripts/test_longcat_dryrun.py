#!/usr/bin/env python
"""Dry-run analysis for LongCat-Flash-Chat expansion (no weight files needed).

Tests both:
  1. Expert Upcycling  (512 -> 1024 experts)
  2. Depth Expansion   (28  -> 32  layers)
"""

import sys

MODEL_DIR = "/Users/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat"


def check_expert_upcycling():
    print("\n" + "=" * 60)
    print("  Expert Upcycling  (512 -> 1024 experts, top-k 12 -> 24)")
    print("=" * 60)
    from llm_grow.safetensor.longcat import (
        LongcatExpertUpcyclingConfig,
        LongcatExpertUpcyclingExpander,
    )
    from llm_grow.safetensor.utils import ShardIndex

    cfg = LongcatExpertUpcyclingConfig(expand_factor=2, noise_scale=1e-6)
    expander = LongcatExpertUpcyclingExpander(cfg)
    plan = expander.dry_run(MODEL_DIR)

    src_index = ShardIndex.load(MODEL_DIR)
    wmap = src_index.weight_map

    # Verify expert tensors doubled
    orig_experts = expander._count_experts_per_layer(wmap)
    new_expert_keys = [k for k in plan.recipes if "mlp.experts." in k]
    experts_in_plan = len(
        {
            int(__import__("re").search(r"experts\.(\d+)", k).group(1))
            for k in new_expert_keys
            if k.startswith("model.layers.0.")
        }
    )
    print(f"\n  [Check] Original experts/layer : {orig_experts}")
    print(f"  [Check] Plan experts/layer     : {experts_in_plan}")
    assert experts_in_plan == orig_experts * 2, "Expert count mismatch!"
    print("  [OK] Expert count doubled correctly")

    # Verify router keys present and modified
    router_cls_keys = [k for k in plan.recipes if "mlp.router.classifier.weight" in k]
    router_bias_keys = [k for k in plan.recipes if "e_score_correction_bias" in k]
    print(f"  [Check] Router classifier keys : {len(router_cls_keys)} (dup_rows)")
    print(f"  [Check] Router bias keys       : {len(router_bias_keys)} (dup_rows)")
    for k in router_cls_keys[:1]:
        assert plan.recipes[k].dup_rows, "Classifier should have dup_rows=True"
    print("  [OK] Router classifier dup_rows=True")

    # Verify config patches
    assert plan.config_patches.get("n_routed_experts") == 1024
    assert plan.config_patches.get("zero_expert_num") == 512
    assert plan.config_patches.get("moe_topk") == 24
    print("  [OK] Config: n_routed_experts=1024, zero_expert_num=512, moe_topk=24")

    total_new = len([k for k in plan.recipes if k not in wmap])
    print(f"\n  Total new tensors added : {total_new}")
    print(f"  Total output tensors    : {len(plan.recipes)}")
    return True


def check_depth_expansion():
    print("\n" + "=" * 60)
    print("  Depth Expansion  (28 -> 32 layers, +4 identity blocks)")
    print("=" * 60)
    from llm_grow.safetensor.longcat import LongcatDepthConfig, LongcatDepthExpander
    from llm_grow.safetensor.utils import ShardIndex

    cfg = LongcatDepthConfig(num_new_layers=4, insert_strategy="uniform")
    expander = LongcatDepthExpander(cfg)
    plan = expander.dry_run(MODEL_DIR)

    src_index = ShardIndex.load(MODEL_DIR)

    assert plan.new_num_hidden_layers == 32, f"Expected 32, got {plan.new_num_hidden_layers}"
    print(f"  [OK] new_num_hidden_layers = {plan.new_num_hidden_layers}")

    zero_recipes = [k for k, r in plan.recipes.items() if r.zero_out]
    zero_per_layer = {
        "o_proj": sum(1 for k in zero_recipes if "o_proj" in k),
        "expert_dp": sum(1 for k in zero_recipes if "mlp.experts" in k and "down_proj" in k),
        "dense_dp": sum(1 for k in zero_recipes if "mlps." in k and "down_proj" in k),
    }
    print("  Zero tensors breakdown:")
    for cat, cnt in zero_per_layer.items():
        print(f"    {cat}: {cnt}")

    # 4 identity layers each with:
    #   2 o_proj (attn.0, attn.1)
    #   512 expert down_proj
    #   2 dense MLP down_proj
    expected_zero_per_id_layer = 2 + 512 + 2  # = 516
    expected_total_zero = 4 * expected_zero_per_id_layer
    print(f"  Expected zero tensors: {expected_total_zero}  (4 layers x 516)")
    print(f"  Actual   zero tensors: {len(zero_recipes)}")
    assert len(zero_recipes) == expected_total_zero, (
        f"Zero count mismatch: {len(zero_recipes)} != {expected_total_zero}"
    )
    print("  [OK] Zero tensor count correct")

    # Non-layer tensors (mtp, embed, etc.) must all pass through
    non_layer_orig = sum(
        1 for k in src_index.weight_map if k.split(".")[0] != "model" or k.split(".")[1] not in ("layers",)
    )
    non_layer_plan = sum(1 for k in plan.recipes if not k.startswith("model.layers."))
    print(f"  [Check] Non-layer tensors: {non_layer_orig} (src) -> {non_layer_plan} (plan)")
    assert non_layer_orig == non_layer_plan, "Non-layer tensor count changed!"
    print("  [OK] All non-layer tensors (embed, norm, mtp) pass through")
    return True


def main():
    results = {}
    for name, fn in [
        ("expert_upcycling", check_expert_upcycling),
        ("depth_expansion", check_depth_expansion),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n  [FAIL] {e}")
            import traceback

            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    print("\nNote: weight files not present -- dry_run validated plan logic only.")
    print("      Download weights to run actual expansion.")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
