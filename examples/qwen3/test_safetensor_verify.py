#!/usr/bin/env python
"""Safetensor-level verification tests for Qwen3-0.6B.

Tests real weight expansion for dense models:
  1. ZeroBlockInsert (LLaMA-Pro, FP)
  2. OverlapCopy (SOLAR DUS, non-FP)
  3. MultiAxisPad (MSG, depth + FFN width, FP)
  4. dup_rows + router_split unit test (no model)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.helpers import (
    get_tensor,
    log_result,
    open_tensors,
    print_summary,
    results,
    verify_config,
    verify_fp_logits,
    verify_passthrough_keys,
)
from common.model_paths import QWEN3_06B, require_path

SRC = require_path("QWEN3_06B", QWEN3_06B)


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
        ).expand(src_dir=SRC, dst_dir=dst, verbose=False)

        verify_config(dst, {"num_hidden_layers": 35}, label)

        src_idx, src_h = open_tensors(SRC)
        dst_idx, dst_h = open_tensors(dst)

        log_result(
            f"{label}/tensor_count",
            len(dst_idx.all_keys) == 388,
            f"{len(dst_idx.all_keys)}",
        )

        non_layer = [k for k in src_idx.all_keys if not k.startswith("model.layers.")]
        verify_passthrough_keys(src_idx, src_h, dst_idx, dst_h, non_layer, label)

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

        verify_fp_logits(SRC, dst, label)


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
        ).expand(src_dir=SRC, dst_dir=dst, verbose=False)

        verify_config(dst, {"num_hidden_layers": 40}, label)

        src_idx, src_h = open_tensors(SRC)
        dst_idx, dst_h = open_tensors(dst)

        log_result(
            f"{label}/tensor_count",
            len(dst_idx.all_keys) == 443,
            f"{len(dst_idx.all_keys)}",
        )

        key = "model.layers.0.mlp.gate_proj.weight"
        src_t = get_tensor(src_h, src_idx.weight_map, key)
        dst_t = get_tensor(dst_h, dst_idx.weight_map, key)
        log_result(f"{label}/layer0_identical", torch.equal(src_t, dst_t))

        non_layer = [k for k in src_idx.all_keys if not k.startswith("model.layers.")]
        verify_passthrough_keys(src_idx, src_h, dst_idx, dst_h, non_layer, label)


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
        ).expand(src_dir=SRC, dst_dir=dst, verbose=False)

        verify_config(dst, {"num_hidden_layers": 32, "intermediate_size": 3584}, label)

        src_idx, src_h = open_tensors(SRC)
        dst_idx, dst_h = open_tensors(dst)

        log_result(
            f"{label}/tensor_count",
            len(dst_idx.all_keys) == 355,
            f"{len(dst_idx.all_keys)}",
        )

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

        verify_fp_logits(SRC, dst, label)


def test_dup_rows_router_split():
    label = "dup_rows_router_split"
    print(f"\n{'=' * 60}\n  {label}: Router weight expansion invariants\n{'=' * 60}")

    from llm_grow.safetensor.base import TensorRecipe, _apply_recipe

    torch.manual_seed(42)
    n_real, n_zero, hidden = 512, 256, 16
    src = torch.randn(n_real + n_zero, hidden)
    src[n_real:] = 0.0

    recipe = TensorRecipe(
        "", "", dup_rows=True, dup_rows_noise_scale=1e-6, router_split=n_real
    )
    out = _apply_recipe(src, recipe)

    log_result(
        f"{label}/output_shape", list(out.shape) == [2 * (n_real + n_zero), hidden]
    )

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

    recipe2 = TensorRecipe(
        "", "", dup_rows=True, dup_rows_noise_scale=1e-6, router_split=0
    )
    out2 = _apply_recipe(src, recipe2)
    log_result(
        f"{label}/no_split_all_noised", not torch.equal(out2[n_real + n_zero :], src)
    )

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


def main():
    test_dup_rows_router_split()
    test_zero_block_insert()
    test_overlap_copy()
    test_msg()

    sys.exit(print_summary(results))


if __name__ == "__main__":
    main()
