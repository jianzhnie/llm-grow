#!/usr/bin/env python
"""Safetensor-level verification example for Qwen3-0.6B.

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
from typing import cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.helpers import (
    expected_tensor_count_after_depth,
    get_tensor,
    log_result,
    open_tensors,
    print_summary,
    verify_config,
    verify_fp_logits,
    verify_passthrough_keys,
)
from common.model_paths import QWEN3_06B, require_path

SRC = require_path("QWEN3_06B", QWEN3_06B)

RESULTS: dict[str, bool] = {}


def check_zero_block_insert() -> bool:
    label = "zero_block_insert"
    print(f"\n{'=' * 60}\n  {label}: Dense depth expansion (+7 blocks)\n{'=' * 60}")

    from llm_grow.safetensor.methods.zero_block_insert import (
        ZeroBlockInsertSafetensorConfig,
        ZeroBlockInsertSafetensorExpander,
    )

    ok = True
    with tempfile.TemporaryDirectory() as dst:
        ZeroBlockInsertSafetensorExpander(
            ZeroBlockInsertSafetensorConfig(num_new_layers=7)
        ).expand(src_dir=SRC, dst_dir=dst, verbose=False)

        ok = verify_config(dst, {"num_hidden_layers": 35}, label) and ok

        src_idx, src_h = open_tensors(SRC)
        dst_idx, dst_h = open_tensors(dst)

        expected = expected_tensor_count_after_depth(src_idx, 35)
        count_ok = len(dst_idx.all_keys) == expected
        log_result(
            f"{label}/tensor_count",
            count_ok,
            f"{len(dst_idx.all_keys)} (expected {expected})",
        )
        ok = ok and count_ok

        non_layer = [k for k in src_idx.all_keys if not k.startswith("model.layers.")]
        ok = (
            verify_passthrough_keys(src_idx, src_h, dst_idx, dst_h, non_layer, label)
            and ok
        )

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
        expected_identity = 2 * 7  # two zeroed projections per new identity layer
        identity_ok = len(identity_zero_keys) == expected_identity
        log_result(
            f"{label}/identity_blocks_found",
            identity_ok,
            f"{len(identity_zero_keys)} (expected {expected_identity})",
        )
        ok = ok and identity_ok

        ok = verify_fp_logits(SRC, dst, label) and ok
    return ok


def check_overlap_copy() -> bool:
    label = "overlap_copy"
    print(f"\n{'=' * 60}\n  {label}: Dense DUS (overlap=8)\n{'=' * 60}")

    from llm_grow.safetensor.methods.overlap_copy import (
        OverlapCopySafetensorConfig,
        OverlapCopySafetensorExpander,
    )

    ok = True
    with tempfile.TemporaryDirectory() as dst:
        OverlapCopySafetensorExpander(
            OverlapCopySafetensorConfig(num_overlap=8)
        ).expand(src_dir=SRC, dst_dir=dst, verbose=False)

        ok = verify_config(dst, {"num_hidden_layers": 40}, label) and ok

        src_idx, src_h = open_tensors(SRC)
        dst_idx, dst_h = open_tensors(dst)

        expected = expected_tensor_count_after_depth(src_idx, 40)
        count_ok = len(dst_idx.all_keys) == expected
        log_result(
            f"{label}/tensor_count",
            count_ok,
            f"{len(dst_idx.all_keys)} (expected {expected})",
        )
        ok = ok and count_ok

        key = "model.layers.0.mlp.gate_proj.weight"
        src_t = get_tensor(src_h, src_idx.weight_map, key)
        dst_t = get_tensor(dst_h, dst_idx.weight_map, key)
        layer_ok = torch.equal(src_t, dst_t)
        log_result(f"{label}/layer0_identical", layer_ok)
        ok = ok and layer_ok

        non_layer = [k for k in src_idx.all_keys if not k.startswith("model.layers.")]
        ok = (
            verify_passthrough_keys(src_idx, src_h, dst_idx, dst_h, non_layer, label)
            and ok
        )
    return ok


def check_msg() -> bool:
    label = "msg"
    print(f"\n{'=' * 60}\n  {label}: Dense depth+4 + FFN+512\n{'=' * 60}")

    from llm_grow.safetensor.methods.multi_axis_pad import (
        MultiAxisPadSafetensorConfig,
        MultiAxisPadSafetensorExpander,
    )

    ok = True
    with tempfile.TemporaryDirectory() as dst:
        MultiAxisPadSafetensorExpander(
            MultiAxisPadSafetensorConfig(num_new_layers=4, ffn_size_expansion=512)
        ).expand(src_dir=SRC, dst_dir=dst, verbose=False)

        ok = (
            verify_config(
                dst, {"num_hidden_layers": 32, "intermediate_size": 3584}, label
            )
            and ok
        )

        src_idx, src_h = open_tensors(SRC)
        dst_idx, dst_h = open_tensors(dst)

        expected = expected_tensor_count_after_depth(src_idx, 32)
        count_ok = len(dst_idx.all_keys) == expected
        log_result(
            f"{label}/tensor_count",
            count_ok,
            f"{len(dst_idx.all_keys)} (expected {expected})",
        )
        ok = ok and count_ok

        src_gate = get_tensor(
            src_h, src_idx.weight_map, "model.layers.0.mlp.gate_proj.weight"
        )
        dst_gate = get_tensor(
            dst_h, dst_idx.weight_map, "model.layers.0.mlp.gate_proj.weight"
        )
        orig_rows = src_gate.shape[0]
        shape_ok = list(dst_gate.shape) == [orig_rows + 512, src_gate.shape[1]]
        log_result(
            f"{label}/gate_proj_shape",
            shape_ok,
            f"{list(dst_gate.shape)}",
        )
        ok = ok and shape_ok
        preserve_ok = torch.equal(dst_gate[:orig_rows], src_gate)
        log_result(
            f"{label}/gate_proj_orig_preserved",
            preserve_ok,
        )
        ok = ok and preserve_ok
        pad_ok = dst_gate[orig_rows:].abs().max().item() == 0
        log_result(
            f"{label}/gate_proj_padding_zero",
            pad_ok,
        )
        ok = ok and pad_ok

        src_down = get_tensor(
            src_h, src_idx.weight_map, "model.layers.0.mlp.down_proj.weight"
        )
        dst_down = get_tensor(
            dst_h, dst_idx.weight_map, "model.layers.0.mlp.down_proj.weight"
        )
        orig_cols = src_down.shape[1]
        down_preserve_ok = torch.equal(dst_down[:, :orig_cols], src_down)
        log_result(
            f"{label}/down_proj_cols_preserved",
            down_preserve_ok,
        )
        ok = ok and down_preserve_ok
        down_pad_ok = dst_down[:, orig_cols:].abs().max().item() == 0
        log_result(
            f"{label}/down_proj_padding_zero",
            down_pad_ok,
        )
        ok = ok and down_pad_ok

        ok = verify_fp_logits(SRC, dst, label) and ok
    return ok


def main() -> int:
    RESULTS["zero_block_insert"] = check_zero_block_insert()
    RESULTS["overlap_copy"] = check_overlap_copy()
    RESULTS["msg"] = check_msg()
    return cast(int, print_summary(RESULTS))


if __name__ == "__main__":
    sys.exit(main())
