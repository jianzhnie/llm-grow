"""Shared test helpers for safetensor verification examples."""

from __future__ import annotations

import json
from pathlib import Path

import torch


results: dict[str, bool] = {}


def log_result(name: str, ok: bool, msg: str = "") -> None:
    icon = "PASS" if ok else "FAIL"
    results[name] = ok
    suffix = f"  {msg}" if msg else ""
    print(f"  [{icon}] {name}{suffix}")


def open_tensors(model_dir: str | Path):
    """Open all safetensor shards and return (index, handles)."""
    from llm_grow.safetensor.utils import ShardIndex

    idx = ShardIndex.load(model_dir)
    handles = idx.open_all_shards()
    return idx, handles


def get_tensor(handles, weight_map, key):
    return handles[weight_map[key]].get_tensor(key)


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


def print_summary(test_results: dict[str, bool]) -> int:
    """Print pass/fail summary and return exit code."""
    print(f"\n{'=' * 60}")
    print("  Summary")
    print("=" * 60)
    passed = sum(1 for v in test_results.values() if v)
    failed = sum(1 for v in test_results.values() if not v)
    for name, ok in test_results.items():
        if not ok:
            print(f"  [FAIL] {name}")
    print(f"\n  {passed} passed, {failed} failed, {len(test_results)} total")
    return 0 if failed == 0 else 1


def count_zero_recipes(plan):
    return sum(1 for r in plan.recipes.values() if r.zero_out)


def count_dup_recipes(plan):
    return sum(1 for r in plan.recipes.values() if r.dup_rows)


def count_new_keys(plan, wmap):
    return sum(1 for k in plan.recipes if k not in wmap)
