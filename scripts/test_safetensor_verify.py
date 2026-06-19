#!/usr/bin/env python
"""End-to-end tensor-level verification for all safetensor expansion methods.

Uses Qwen3-0.6B (real weights) for value-level checks and dry_run for
models without weights.  For each expansion type verifies:

  1. Config patches correct
  2. Output tensor count matches plan
  3. Passthrough tensors byte-identical to source
  4. Identity-block tensors (zero_out) are truly all-zero
  5. Padded tensors: original region matches, new region is zero
  6. dup_rows tensors: correct row layout (real/zero split, noise presence)
  7. FP methods: logit-level identity on random input
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

QWEN3_06B = "/Users/robin/hfhub/models/Qwen/Qwen3-0.6B"
LONGCAT = "/Users/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat"
QWEN3_30B = "/Users/robin/hfhub/models/Qwen/Qwen3-30B-A3B"
KIMI_K2 = "/Users/robin/hfhub/models/moonshotai/Kimi-K2-Base"

results: dict[str, bool] = {}


def log_result(name: str, ok: bool, msg: str = "") -> None:
    icon = "PASS" if ok else "FAIL"
    results[name] = ok
    suffix = f"  {msg}" if msg else ""
    print(f"  [{icon}] {name}{suffix}")


def open_tensors(model_dir: str | Path):
    """Open all safetensor shards and return {key: tensor} accessor."""
    from llm_grow.safetensor.utils import ShardIndex

    idx = ShardIndex.load(model_dir)
    handles = idx.open_all_shards()
    return idx, handles


def get_tensor(handles, weight_map, key):
    return handles[weight_map[key]].get_tensor(key)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: compare two model directories tensor-by-tensor
# ═══════════════════════════════════════════════════════════════════════════════


def verify_passthrough_keys(
    src_idx, src_h, dst_idx, dst_h, passthrough_keys: list[str], label: str
) -> bool:
    """Check that passthrough keys are byte-identical."""
    mismatches = []
    for key in passthrough_keys[:20]:
        src_t = get_tensor(src_h, src_idx.weight_map, key)
        dst_t = get_tensor(dst_h, dst_idx.weight_map, key)
        if not torch.equal(src_t, dst_t):
            mismatches.append(key)
    ok = len(mismatches) == 0
    log_result(
        f"{label}/passthrough_identical",
        ok,
        f"checked {min(20, len(passthrough_keys))}",
    )
    return ok


def verify_zero_keys(dst_idx, dst_h, zero_keys: list[str], label: str) -> bool:
    """Check that all zero_out keys are truly all-zero."""
    non_zero = []
    for key in zero_keys[:30]:
        t = get_tensor(dst_h, dst_idx.weight_map, key)
        if t.abs().max().item() > 0:
            non_zero.append(key)
    ok = len(non_zero) == 0
    log_result(
        f"{label}/zero_tensors_all_zero", ok, f"checked {min(30, len(zero_keys))}"
    )
    return ok


def verify_fp_logits(
    src_dir: str, dst_dir: str, label: str, atol: float = 1e-4
) -> bool:
    """Load both models and compare logits on random input."""
    from transformers import AutoModelForCausalLM

    orig = AutoModelForCausalLM.from_pretrained(src_dir, torch_dtype=torch.float32)
    exp = AutoModelForCausalLM.from_pretrained(dst_dir, torch_dtype=torch.float32)
    orig.eval()
    exp.eval()
    ids = torch.randint(0, orig.config.vocab_size, (2, 32))
    with torch.no_grad():
        lo = orig(input_ids=ids).logits
        le = exp(input_ids=ids).logits
    max_err = (lo - le).abs().max().item()
    ok = max_err < atol
    log_result(f"{label}/fp_logits", ok, f"max|Δ|={max_err:.2e}")
    return ok


def verify_config(dst_dir: str, expected_patches: dict, label: str) -> bool:
    """Check that config.json contains expected patches."""
    with open(Path(dst_dir) / "config.json") as f:
        cfg = json.load(f)
    mismatches = []
    for k, v in expected_patches.items():
        if cfg.get(k) != v:
            mismatches.append(f"{k}: expected {v}, got {cfg.get(k)}")
    ok = len(mismatches) == 0
    log_result(
        f"{label}/config_patches", ok, "; ".join(mismatches) if mismatches else ""
    )
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: LLaMA-Pro (Dense depth, FP)
# ═══════════════════════════════════════════════════════════════════════════════


def test_zero_block_insert():
    label = "zero_block_insert"
    print(f"\n{'=' * 60}\n  {label}: Dense depth expansion (+7 blocks)\n{'=' * 60}")

    from llm_grow.safetensor.zero_block_insert import (
        ZeroBlockInsertSafetensorConfig,
        ZeroBlockInsertSafetensorExpander,
    )

    with tempfile.TemporaryDirectory() as dst:
        ZeroBlockInsertSafetensorExpander(
            ZeroBlockInsertSafetensorConfig(num_new_layers=7)
        ).expand(src_dir=QWEN3_06B, dst_dir=dst, verbose=False)

        verify_config(dst, {"num_hidden_layers": 35}, label)

        src_idx, src_h = open_tensors(QWEN3_06B)
        dst_idx, dst_h = open_tensors(dst)

        log_result(
            f"{label}/tensor_count",
            len(dst_idx.all_keys) == 388,
            f"{len(dst_idx.all_keys)}",
        )

        # Non-layer keys must be identical
        non_layer = [k for k in src_idx.all_keys if not k.startswith("model.layers.")]
        verify_passthrough_keys(src_idx, src_h, dst_idx, dst_h, non_layer, label)

        # Identity blocks: o_proj and down_proj must be zero
        zero_keys = [
            k
            for k in dst_idx.all_keys
            if k.endswith(("self_attn.o_proj.weight", "mlp.down_proj.weight"))
        ]
        identity_zero_keys = []
        for k in zero_keys:
            t = get_tensor(dst_h, dst_idx.weight_map, k)
            if t.abs().max().item() == 0:
                identity_zero_keys.append(k)
        log_result(
            f"{label}/identity_blocks_found",
            len(identity_zero_keys) == 14,
            f"{len(identity_zero_keys)}",
        )

        verify_fp_logits(QWEN3_06B, dst, label)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: SOLAR DUS (Dense depth, non-FP)
# ═══════════════════════════════════════════════════════════════════════════════


def test_overlap_copy():
    label = "overlap_copy"
    print(f"\n{'=' * 60}\n  {label}: Dense DUS (overlap=8)\n{'=' * 60}")

    from llm_grow.safetensor.overlap_copy import (
        OverlapCopySafetensorConfig,
        OverlapCopySafetensorExpander,
    )

    with tempfile.TemporaryDirectory() as dst:
        OverlapCopySafetensorExpander(
            OverlapCopySafetensorConfig(num_overlap=8)
        ).expand(src_dir=QWEN3_06B, dst_dir=dst, verbose=False)

        verify_config(dst, {"num_hidden_layers": 40}, label)

        src_idx, src_h = open_tensors(QWEN3_06B)
        dst_idx, dst_h = open_tensors(dst)

        log_result(
            f"{label}/tensor_count",
            len(dst_idx.all_keys) == 443,
            f"{len(dst_idx.all_keys)}",
        )

        # Layer 0 should be identical to source layer 0
        key = "model.layers.0.mlp.gate_proj.weight"
        src_t = get_tensor(src_h, src_idx.weight_map, key)
        dst_t = get_tensor(dst_h, dst_idx.weight_map, key)
        log_result(f"{label}/layer0_identical", torch.equal(src_t, dst_t))

        non_layer = [k for k in src_idx.all_keys if not k.startswith("model.layers.")]
        verify_passthrough_keys(src_idx, src_h, dst_idx, dst_h, non_layer, label)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: MSG (Dense depth + FFN width, FP)
# ═══════════════════════════════════════════════════════════════════════════════


def test_msg():
    label = "msg"
    print(f"\n{'=' * 60}\n  {label}: Dense depth+4 + FFN+512\n{'=' * 60}")

    from llm_grow.safetensor.multi_axis_pad import (
        MultiAxisPadSafetensorConfig,
        MultiAxisPadSafetensorExpander,
    )

    with tempfile.TemporaryDirectory() as dst:
        MultiAxisPadSafetensorExpander(
            MultiAxisPadSafetensorConfig(num_new_layers=4, ffn_size_expansion=512)
        ).expand(src_dir=QWEN3_06B, dst_dir=dst, verbose=False)

        verify_config(dst, {"num_hidden_layers": 32, "intermediate_size": 3584}, label)

        src_idx, src_h = open_tensors(QWEN3_06B)
        dst_idx, dst_h = open_tensors(dst)

        log_result(
            f"{label}/tensor_count",
            len(dst_idx.all_keys) == 355,
            f"{len(dst_idx.all_keys)}",
        )

        # Check gate_proj padding: original rows preserved, new rows zero
        src_gate = get_tensor(
            src_h, src_idx.weight_map, "model.layers.0.mlp.gate_proj.weight"
        )
        dst_gate = get_tensor(
            dst_h, dst_idx.weight_map, "model.layers.0.mlp.gate_proj.weight"
        )
        orig_rows = src_gate.shape[0]
        log_result(
            f"{label}/gate_proj_shape",
            list(dst_gate.shape) == [orig_rows + 512, src_gate.shape[1]],
            f"{list(dst_gate.shape)}",
        )
        log_result(
            f"{label}/gate_proj_orig_preserved",
            torch.equal(dst_gate[:orig_rows], src_gate),
        )
        log_result(
            f"{label}/gate_proj_padding_zero",
            dst_gate[orig_rows:].abs().max().item() == 0,
        )

        # Check down_proj padding: original cols preserved
        src_down = get_tensor(
            src_h, src_idx.weight_map, "model.layers.0.mlp.down_proj.weight"
        )
        dst_down = get_tensor(
            dst_h, dst_idx.weight_map, "model.layers.0.mlp.down_proj.weight"
        )
        orig_cols = src_down.shape[1]
        log_result(
            f"{label}/down_proj_cols_preserved",
            torch.equal(dst_down[:, :orig_cols], src_down),
        )
        log_result(
            f"{label}/down_proj_padding_zero",
            dst_down[:, orig_cols:].abs().max().item() == 0,
        )

        verify_fp_logits(QWEN3_06B, dst, label)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: dup_rows + router_split (unit test, no model load)
# ═══════════════════════════════════════════════════════════════════════════════


def test_dup_rows_router_split():
    label = "dup_rows_router_split"
    print(f"\n{'=' * 60}\n  {label}: Router weight expansion invariants\n{'=' * 60}")

    from llm_grow.safetensor.base import TensorRecipe, _apply_recipe

    torch.manual_seed(42)
    n_real, n_zero, hidden = 512, 256, 16
    src = torch.randn(n_real + n_zero, hidden)
    src[n_real:] = 0.0  # zero expert rows

    # With router_split: real rows get noise, zero rows don't
    recipe = TensorRecipe(
        "", "", dup_rows=True, dup_rows_noise_scale=1e-6, router_split=n_real
    )
    out = _apply_recipe(src, recipe)

    log_result(
        f"{label}/output_shape", list(out.shape) == [2 * (n_real + n_zero), hidden]
    )

    # Layout: [real, real+noise, zeros, zeros]
    real_orig = out[:n_real]
    real_dup = out[n_real : 2 * n_real]
    zero_orig = out[2 * n_real : 2 * n_real + n_zero]
    zero_dup = out[2 * n_real + n_zero :]

    log_result(f"{label}/real_orig_exact", torch.equal(real_orig, src[:n_real]))
    log_result(f"{label}/real_dup_has_noise", not torch.equal(real_dup, src[:n_real]))
    log_result(
        f"{label}/real_dup_close", torch.allclose(real_dup, src[:n_real], atol=1e-3)
    )
    log_result(f"{label}/zero_orig_exact", torch.equal(zero_orig, src[n_real:]))
    log_result(f"{label}/zero_dup_exact", torch.equal(zero_dup, src[n_real:]))
    log_result(f"{label}/zero_rows_still_zero", zero_orig.abs().max().item() == 0)

    # Without router_split: all rows get noise
    recipe2 = TensorRecipe(
        "", "", dup_rows=True, dup_rows_noise_scale=1e-6, router_split=0
    )
    out2 = _apply_recipe(src, recipe2)
    log_result(
        f"{label}/no_split_all_noised", not torch.equal(out2[n_real + n_zero :], src)
    )

    # Bias: noise_scale=0 → exact copies
    recipe3 = TensorRecipe(
        "", "", dup_rows=True, dup_rows_noise_scale=0.0, router_split=n_real
    )
    out3 = _apply_recipe(src, recipe3)
    log_result(
        f"{label}/bias_real_exact_copy",
        torch.equal(out3[n_real : 2 * n_real], src[:n_real]),
    )
    log_result(
        f"{label}/bias_zero_exact_copy",
        torch.equal(out3[2 * n_real + n_zero :], src[n_real:]),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Dry-run plan verification (models without weights)
# ═══════════════════════════════════════════════════════════════════════════════


def test_dryrun_plans():
    label = "dryrun_plans"
    header = f"{label}: Plan verification for MoE models (no weights)"
    print(f"\n{'=' * 60}\n  {header}\n{'=' * 60}")

    from llm_grow.safetensor.detect import detect_model

    checks = [
        (
            "Qwen3-30B/expert",
            QWEN3_30B,
            "expert",
            {"expand_factor": 2},
            {"num_experts": 256},
        ),
        ("Qwen3-30B/depth", QWEN3_30B, "depth", {"num_new_layers": 4}, {}),
        (
            "Kimi-K2/expert",
            KIMI_K2,
            "expert",
            {"expand_factor": 2},
            {"n_routed_experts": 768},
        ),
        (
            "LongCat/expert",
            LONGCAT,
            "expert",
            {"expand_factor": 2},
            {"n_routed_experts": 1024},
        ),
        ("LongCat/depth", LONGCAT, "depth", {"num_new_layers": 4}, {}),
    ]

    for name, src, method, kwargs, expected_patches in checks:
        try:
            profile = detect_model(src)
            from llm_grow.safetensor.auto import _build_expander

            exp = _build_expander(
                method,
                profile,
                kwargs.get("num_new_layers", 4),
                "uniform",
                kwargs.get("expand_factor", 2),
                1e-6,
                0,
            )
            from llm_grow.safetensor.utils import ShardIndex

            plan = exp._build_plan(ShardIndex.load(src))

            # Check config patches
            ok = all(
                plan.config_patches.get(k) == v for k, v in expected_patches.items()
            )
            log_result(f"{label}/{name}/config", ok, str(plan.config_patches))

            # Check tensor count > source
            src_count = len(ShardIndex.load(src).all_keys)
            ok2 = len(plan.recipes) > src_count
            log_result(
                f"{label}/{name}/more_tensors",
                ok2,
                f"{src_count} → {len(plan.recipes)}",
            )

        except Exception as e:
            log_result(f"{label}/{name}", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    test_dup_rows_router_split()
    test_zero_block_insert()
    test_overlap_copy()
    test_msg()
    test_dryrun_plans()

    print(f"\n{'=' * 60}")
    print("  Summary")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    for name, ok in results.items():
        if not ok:
            print(f"  [FAIL] {name}")
    print(f"\n  {passed} passed, {failed} failed, {len(results)} total")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
