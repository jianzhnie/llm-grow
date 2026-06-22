#!/usr/bin/env python
"""Dry-run expansion example for LongCat-Flash-Lite (dual-attn, 256 experts, 14 layers).

All checks use only the safetensors index — no weight files loaded.
Tests expert upcycling, depth expansion, and group routing.
"""

from __future__ import annotations

import sys

from common.helpers import (
    count_experts_in_layer,
    run_tests,
    verify_dryrun_plan,
)
from common.model_paths import LONGCAT, require_path

SRC = require_path("LONGCAT", LONGCAT)


def check_expert_clone():
    print(f"\n{'=' * 60}")
    print("  [1] LongCat  Expert Upcycling  (256 -> 512 experts)")
    print("=" * 60)
    from llm_grow.safetensor.models.longcat import (
        LongcatExpertCloneConfig,
        LongcatExpertCloneExpander,
    )
    from llm_grow.safetensor.utils import ShardIndex

    cfg = LongcatExpertCloneConfig(expand_factor=2, noise_scale=1e-6, scale_moe_topk=False)
    expander = LongcatExpertCloneExpander(cfg)
    plan = expander.dry_run(SRC)

    src_index = ShardIndex.load(SRC)
    wmap = src_index.weight_map

    orig_experts = expander._count_experts_per_layer(wmap)
    experts_in_plan = count_experts_in_layer(plan, 0)
    assert experts_in_plan == orig_experts * 2, "Expert count mismatch!"
    print(f"  [OK] Expert count doubled: {orig_experts} -> {experts_in_plan}")

    router_cls_keys = [k for k in plan.recipes if "mlp.router.classifier.weight" in k]
    for k in router_cls_keys[:1]:
        assert plan.recipes[k].dup_rows, "Classifier should have dup_rows=True"
    print(f"  [OK] {len(router_cls_keys)} router classifier keys: dup_rows=True")

    assert plan.config_patches.get("n_routed_experts") == 1024
    assert plan.config_patches.get("zero_expert_num") == 512
    print("  [OK] Config: n_routed_experts=1024, zero_expert_num=512")

    total_new = len([k for k in plan.recipes if k not in wmap])
    print(f"  New tensors : {total_new}")
    print(f"  Total output: {len(plan.recipes)}")
    return True


def check_depth():
    print(f"\n{'=' * 60}")
    print("  [2] LongCat  Depth Expansion  (14 -> 18 layers, +4 identity blocks)")
    print("=" * 60)
    from llm_grow.safetensor.models.longcat import (
        LongcatDepthConfig,
        LongcatDepthExpander,
    )
    from llm_grow.safetensor.utils import ShardIndex

    cfg = LongcatDepthConfig(num_new_layers=4, insert_strategy="uniform")
    expander = LongcatDepthExpander(cfg)
    plan = expander.dry_run(SRC)

    assert plan.new_num_hidden_layers == 18, f"Expected 18, got {plan.new_num_hidden_layers}"
    print(f"  [OK] new_num_hidden_layers = {plan.new_num_hidden_layers}")

    zero_recipes = [k for k, r in plan.recipes.items() if r.zero_out]
    zero_per_layer = 2 + 512 + 2  # dual o_proj + 256*2 experts + dual mlp down_proj
    expected_total_zero = 4 * zero_per_layer
    assert len(zero_recipes) == expected_total_zero, (
        f"Zero count mismatch: {len(zero_recipes)} != {expected_total_zero}"
    )
    print(f"  [OK] Zero tensor count correct: {len(zero_recipes)}")

    src_index = ShardIndex.load(SRC)
    non_layer_orig = sum(
        1 for k in src_index.weight_map
        if not k.startswith("model.layers.")
    )
    non_layer_plan = sum(1 for k in plan.recipes if not k.startswith("model.layers."))
    assert non_layer_orig == non_layer_plan, "Non-layer tensor count changed!"
    print(f"  [OK] Non-layer tensors preserved: {non_layer_orig}")
    return True


if __name__ == "__main__":
    sys.exit(run_tests([
        ("expert_clone", check_expert_clone),
        ("depth", check_depth),
        ("dryrun_plan", lambda: verify_dryrun_plan(
            SRC, "LongCat", [
                ("expert", {"expand_factor": 2}, {"n_routed_experts": 1024}),
                ("depth", {"num_new_layers": 4}, {}),
            ],
        )),
    ]))
