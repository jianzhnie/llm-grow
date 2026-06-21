#!/usr/bin/env python
"""Dry-run example for LongCat-Flash-Chat expansion (no weight files needed).

Tests both:
  1. Expert Upcycling  (512 -> 1024 experts)
  2. Depth Expansion   (28  -> 32  layers)
  3. Dry-run plan verification via auto-detect
"""

from __future__ import annotations

import sys

from common.helpers import (
    count_experts_in_layer,
    run_tests,
    verify_dryrun_plan,
)
from common.model_paths import LONGCAT, require_path

MODEL_DIR = require_path("LONGCAT", LONGCAT)


def check_expert_clone():
    print("\n" + "=" * 60)
    print("  Expert Upcycling  (512 -> 1024 experts)")
    print("  scale_moe_topk=False: matches expand_experts.py default")
    print("=" * 60)
    from llm_grow.safetensor.models.longcat import (
        LongcatExpertCloneConfig,
        LongcatExpertCloneExpander,
    )
    from llm_grow.safetensor.utils import ShardIndex

    cfg = LongcatExpertCloneConfig(
        expand_factor=2, noise_scale=1e-6, scale_moe_topk=False
    )
    expander = LongcatExpertCloneExpander(cfg)
    plan = expander.dry_run(MODEL_DIR)

    src_index = ShardIndex.load(MODEL_DIR)
    wmap = src_index.weight_map

    orig_experts = expander._count_experts_per_layer(wmap)
    experts_in_plan = count_experts_in_layer(plan, 0)
    print(f"\n  [Check] Original experts/layer : {orig_experts}")
    print(f"  [Check] Plan experts/layer     : {experts_in_plan}")
    assert experts_in_plan == orig_experts * 2, "Expert count mismatch!"
    print("  [OK] Expert count doubled correctly")

    router_cls_keys = [k for k in plan.recipes if "mlp.router.classifier.weight" in k]
    router_bias_keys = [k for k in plan.recipes if "e_score_correction_bias" in k]
    print(f"  [Check] Router classifier keys : {len(router_cls_keys)} (dup_rows)")
    print(f"  [Check] Router bias keys       : {len(router_bias_keys)} (dup_rows)")
    for k in router_cls_keys[:1]:
        assert plan.recipes[k].dup_rows, "Classifier should have dup_rows=True"
    print("  [OK] Router classifier dup_rows=True")

    assert plan.config_patches.get("n_routed_experts") == 1024
    assert plan.config_patches.get("zero_expert_num") == 512
    assert "moe_topk" not in plan.config_patches
    print(
        "  [OK] Config: n_routed_experts=1024, zero_expert_num=512, moe_topk=UNCHANGED"
    )

    cfg2 = LongcatExpertCloneConfig(expand_factor=2, scale_moe_topk=True)
    plan2 = LongcatExpertCloneExpander(cfg2)._build_plan(src_index)
    assert plan2.config_patches.get("moe_topk") == 24
    print("  [OK] scale_moe_topk=True correctly patches moe_topk=24")

    cfg3 = LongcatExpertCloneConfig(expand_factor=2, use_group_routing=True)
    plan3 = LongcatExpertCloneExpander(cfg3)._build_plan(src_index)
    assert plan3.config_patches.get("use_group_routing") is True
    assert plan3.config_patches.get("expert_expansion_factor") == 2
    assert "moe_topk" not in plan3.config_patches
    print("  [OK] use_group_routing=True: adds flags, moe_topk unchanged")

    total_new = len([k for k in plan.recipes if k not in wmap])
    print(f"\n  Total new tensors added : {total_new}")
    print(f"  Total output tensors    : {len(plan.recipes)}")
    return True


def check_depth_expansion():
    print("\n" + "=" * 60)
    print("  Depth Expansion  (28 -> 32 layers, +4 identity blocks)")
    print("=" * 60)
    from llm_grow.safetensor.models.longcat import (
        LongcatDepthConfig,
        LongcatDepthExpander,
    )
    from llm_grow.safetensor.utils import ShardIndex

    cfg = LongcatDepthConfig(num_new_layers=4, insert_strategy="uniform")
    expander = LongcatDepthExpander(cfg)
    plan = expander.dry_run(MODEL_DIR)

    src_index = ShardIndex.load(MODEL_DIR)

    assert plan.new_num_hidden_layers == 32, (
        f"Expected 32, got {plan.new_num_hidden_layers}"
    )
    print(f"  [OK] new_num_hidden_layers = {plan.new_num_hidden_layers}")

    zero_recipes = [k for k, r in plan.recipes.items() if r.zero_out]
    zero_per_layer = {
        "o_proj": sum(1 for k in zero_recipes if "o_proj" in k),
        "expert_dp": sum(
            1 for k in zero_recipes if "mlp.experts" in k and "down_proj" in k
        ),
        "dense_dp": sum(1 for k in zero_recipes if "mlps." in k and "down_proj" in k),
    }
    print("  Zero tensors breakdown:")
    for cat, cnt in zero_per_layer.items():
        print(f"    {cat}: {cnt}")

    expected_zero_per_id_layer = 2 + 512 + 2
    expected_total_zero = 4 * expected_zero_per_id_layer
    print(f"  Expected zero tensors: {expected_total_zero}  (4 layers x 516)")
    print(f"  Actual   zero tensors: {len(zero_recipes)}")
    assert len(zero_recipes) == expected_total_zero, (
        f"Zero count mismatch: {len(zero_recipes)} != {expected_total_zero}"
    )
    print("  [OK] Zero tensor count correct")

    non_layer_orig = sum(
        1
        for k in src_index.weight_map
        if k.split(".")[0] != "model" or k.split(".")[1] not in ("layers",)
    )
    non_layer_plan = sum(1 for k in plan.recipes if not k.startswith("model.layers."))
    print(
        f"  [Check] Non-layer tensors: "
        f"{non_layer_orig} (src) -> {non_layer_plan} (plan)"
    )
    assert non_layer_orig == non_layer_plan, "Non-layer tensor count changed!"
    print("  [OK] All non-layer tensors (embed, norm, mtp) pass through")
    return True


def check_dryrun_plan():
    print("\n" + "=" * 60)
    print("  Dry-run plan verification via auto-detect")
    print("=" * 60)

    return verify_dryrun_plan(
        MODEL_DIR,
        "LongCat",
        [
            ("expert", {"expand_factor": 2}, {"n_routed_experts": 1024}),
            ("depth", {"num_new_layers": 4}, {}),
        ],
    )


def main():
    print("\nNote: weight files not present -- dry_run validated plan logic only.")
    sys.exit(
        run_tests(
            [
                ("expert_clone", check_expert_clone),
                ("depth_expansion", check_depth_expansion),
                ("dryrun_plan", check_dryrun_plan),
            ]
        )
    )


if __name__ == "__main__":
    main()
