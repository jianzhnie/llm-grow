"""Shared helpers for safetensor verification and dry-run examples.

All functions are stateless: they take inputs, print progress, and return
boolean or scalar results. Callers aggregate results into their own dict.
"""

from __future__ import annotations

import contextlib
import json
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import torch


def log_result(name: str, ok: bool, msg: str = "") -> None:
    """Print a single PASS/FAIL line and continue."""
    icon = "PASS" if ok else "FAIL"
    suffix = f"  {msg}" if msg else ""
    print(f"  [{icon}] {name}{suffix}")


def open_tensors(model_dir: str | Path):
    """Open all safetensor shards and return (index, handles).

    Caller is responsible for closing handles via their ``__exit__`` method.
    """
    from llm_grow.safetensor.utils import ShardIndex

    idx = ShardIndex.load(model_dir)
    handles = idx.open_all_shards()
    return idx, handles


def get_tensor(
    handles: dict[str, Any], weight_map: dict[str, str], key: str
) -> torch.Tensor:
    """Fetch a tensor from an open shard handle through the weight map."""
    return cast(torch.Tensor, handles[weight_map[key]].get_tensor(key))


def verify_passthrough_keys(
    src_idx: Any,
    src_h: dict[str, Any],
    dst_idx: Any,
    dst_h: dict[str, Any],
    passthrough_keys: Sequence[str],
    label: str,
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


def verify_zero_keys(
    dst_idx: Any, dst_h: dict[str, Any], zero_keys: Sequence[str], label: str
) -> bool:
    """Check that all zero_out keys are truly all-zero."""
    non_zero = []
    for key in zero_keys[:30]:
        t = get_tensor(dst_h, dst_idx.weight_map, key)
        if t.abs().max().item() > 0:
            non_zero.append(key)
    ok = len(non_zero) == 0
    log_result(
        f"{label}/zero_tensors_all_zero",
        ok,
        f"checked {min(30, len(zero_keys))}",
    )
    return ok


def verify_fp_logits(
    src_dir: str | Path,
    dst_dir: str | Path,
    label: str,
    atol: float = 1e-4,
    seq_len: int = 32,
) -> bool:
    """Load both models and compare logits on random input."""
    from transformers import AutoModelForCausalLM

    orig = AutoModelForCausalLM.from_pretrained(src_dir, torch_dtype=torch.float32)
    exp = AutoModelForCausalLM.from_pretrained(dst_dir, torch_dtype=torch.float32)
    orig.eval()
    exp.eval()
    ids = torch.randint(0, orig.config.vocab_size, (2, seq_len))
    with torch.no_grad():
        lo = orig(input_ids=ids).logits
        le = exp(input_ids=ids).logits
    max_err = (lo - le).abs().max().item()
    ok = bool(max_err < atol)
    log_result(f"{label}/fp_logits", ok, f"max|Δ|={max_err:.2e}")
    return ok


def verify_config(
    dst_dir: str | Path, expected_patches: dict[str, Any], label: str
) -> bool:
    """Check that config.json contains expected patches."""
    with open(Path(dst_dir) / "config.json") as f:
        cfg = json.load(f)
    mismatches = []
    for k, v in expected_patches.items():
        if cfg.get(k) != v:
            mismatches.append(f"{k}: expected {v}, got {cfg.get(k)}")
    ok = len(mismatches) == 0
    log_result(
        f"{label}/config_patches",
        ok,
        "; ".join(mismatches) if mismatches else "",
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


def count_zero_recipes(plan: Any) -> int:
    """Count recipes that produce all-zero tensors."""
    return sum(1 for r in plan.recipes.values() if r.zero_out)


def count_dup_recipes(plan: Any) -> int:
    """Count recipes that duplicate rows."""
    return sum(1 for r in plan.recipes.values() if r.dup_rows)


def count_new_keys(plan: Any, wmap: dict[str, str]) -> int:
    """Count output keys that are not present in the source weight map."""
    return sum(1 for k in plan.recipes if k not in wmap)


def count_experts_in_layer(plan: Any, layer_idx: int) -> int:
    """Count distinct expert indices produced for a given layer."""
    import re

    indices: set[int] = set()
    prefix = f"model.layers.{layer_idx}."
    for key in plan.recipes:
        if not key.startswith(prefix) or "mlp.experts." not in key:
            continue
        match = re.search(r"experts\.(\d+)", key)
        if match:
            indices.add(int(match.group(1)))
    return len(indices)


def run_tests(test_list: Sequence[tuple[str, Callable[[], bool]]]) -> int:
    """Run a list of (name, fn) example pairs and return exit code."""
    test_results: dict[str, bool] = {}
    for name, fn in test_list:
        try:
            test_results[name] = fn()
        except Exception as exc:
            print(f"\n  [FAIL] {exc}")
            traceback.print_exc()
            test_results[name] = False
    return print_summary(test_results)


def verify_dryrun_plan(
    src_dir: str | Path,
    label: str,
    checks: Sequence[tuple[str, dict[str, Any], dict[str, Any]]],
) -> bool:
    """Shared dry-run plan verification logic.

    Args:
        src_dir: Path to the source model directory.
        label: Display label for print output.
        checks: List of (method, kwargs, expected_config_patches).

    Returns:
        True if every check passed.
    """
    from llm_grow.safetensor.auto import _build_expander
    from llm_grow.safetensor.detect import detect_model
    from llm_grow.safetensor.utils import ShardIndex

    all_ok = True
    for method, kwargs, expected_patches in checks:
        profile = detect_model(src_dir)
        exp = _build_expander(
            method,
            profile,
            kwargs.get("num_new_layers", 4),
            "uniform",
            kwargs.get("expand_factor", 2),
            1e-6,
            0,
        )
        plan = exp._build_plan(ShardIndex.load(src_dir))

        ok = all(plan.config_patches.get(k) == v for k, v in expected_patches.items())
        icon = "OK" if ok else "FAIL"
        print(f"  [{icon}] {label}/{method} config: {plan.config_patches}")
        all_ok = all_ok and ok

        src_count = len(ShardIndex.load(src_dir).all_keys)
        ok2 = len(plan.recipes) > src_count
        icon2 = "OK" if ok2 else "FAIL"
        print(f"  [{icon2}] tensors: {src_count} -> {len(plan.recipes)}")
        all_ok = all_ok and ok2

    return all_ok


def expected_tensor_count_after_depth(
    src_idx: Any,
    new_num_hidden_layers: int,
) -> int:
    """Compute expected total tensor count after pure depth expansion.

    Assumes every non-layer tensor passes through unchanged and every layer
    contributes the same number of tensors.
    """
    src_layers = src_idx.num_hidden_layers()
    per_layer = len(src_idx.layer_suffixes())
    non_layer = len(src_idx.all_keys) - src_layers * per_layer
    return cast(int, non_layer + new_num_hidden_layers * per_layer)


def safe_close_handles(handles: dict[str, Any]) -> None:
    """Close all safetensor handles in a dict and clear it."""
    for handle in list(handles.values()):
        with contextlib.suppress(Exception):
            handle.__exit__(None, None, None)
    handles.clear()
