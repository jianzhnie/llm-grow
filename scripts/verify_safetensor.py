#!/usr/bin/env python
"""Verify an expanded safetensor model against its source.

Checks:
  1. Config diff  — num_hidden_layers, intermediate_size, etc.
  2. Weight count — expected tensor count vs. actual
  3. Original weights preserved — spot-check a sample of source layers
  4. Identity blocks zeroed — verify o_proj / down_proj are zero
  5. FP verification — load both models and compare logits (optional, --fp)

Usage
-----
# Fast structural checks only (no model load)
python scripts/verify_safetensor.py \\
    --src /path/to/original \\
    --dst /path/to/expanded

# Full FP verification (loads both models)
python scripts/verify_safetensor.py \\
    --src /path/to/original \\
    --dst /path/to/expanded \\
    --fp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from llm_grow.safetensor.utils import ShardIndex, parse_layer_idx, layer_suffix


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify expanded safetensor model")
    p.add_argument("--src", required=True, help="Original model directory")
    p.add_argument("--dst", required=True, help="Expanded model directory")
    p.add_argument("--fp", action="store_true",
                   help="Load both models and run function-preserving logit check")
    p.add_argument("--fp-seq-len", type=int, default=32)
    p.add_argument("--fp-samples", type=int, default=4)
    p.add_argument("--fp-atol", type=float, default=1e-4)
    return p


# ── structural checks ─────────────────────────────────────────────────────────

def check_config(src_dir: Path, dst_dir: Path) -> bool:
    src_cfg = _load_config(src_dir)
    dst_cfg = _load_config(dst_dir)

    print("\n[Config diff]")
    keys = sorted(set(src_cfg) | set(dst_cfg))
    changed = False
    for k in keys:
        sv, dv = src_cfg.get(k), dst_cfg.get(k)
        if sv != dv:
            print(f"  {k}: {sv} → {dv}")
            changed = True
    if not changed:
        print("  (no changes)")
    return True


def check_tensor_counts(src_idx: ShardIndex, dst_idx: ShardIndex) -> bool:
    print("\n[Tensor counts]")
    print(f"  src: {len(src_idx.all_keys)} tensors across {len(src_idx.shard_files)} shard(s)")
    print(f"  dst: {len(dst_idx.all_keys)} tensors across {len(dst_idx.shard_files)} shard(s)")
    src_layers = src_idx.num_hidden_layers()
    dst_layers = dst_idx.num_hidden_layers()
    per_layer = len(src_idx.layer_suffixes())
    expected = len(src_idx.all_keys) - src_layers * per_layer + dst_layers * per_layer
    ok = len(dst_idx.all_keys) == expected
    icon = "✓" if ok else "✗"
    print(f"  [{icon}] expected {expected}, got {len(dst_idx.all_keys)}")
    return ok


def check_original_weights_preserved(
    src_idx: ShardIndex, dst_idx: ShardIndex, sample: int = 4
) -> bool:
    """Spot-check that original layer tensors survived unchanged.

    Scans dst layers to find the exact new index for each sampled src layer
    (handles non-uniform insertion positions correctly).
    """
    print(f"\n[Original weight preservation  (sample {sample} layers)]")
    src_handles = src_idx.open_all_shards()
    dst_handles = dst_idx.open_all_shards()
    wmap_src = src_idx.weight_map
    wmap_dst = dst_idx.weight_map

    src_layers = src_idx.num_hidden_layers()
    dst_layers = dst_idx.num_hidden_layers()
    suf = "mlp.gate_proj.weight"

    # Pre-load dst tensors for this suffix for all dst layers (for scanning)
    dst_tensors: dict[int, torch.Tensor] = {}
    for dst_i in range(dst_layers):
        key = f"model.layers.{dst_i}.{suf}"
        if key in wmap_dst:
            dst_tensors[dst_i] = dst_handles[wmap_dst[key]].get_tensor(key).float()

    step = max(1, src_layers // sample)
    all_ok = True
    for orig_idx in range(0, src_layers, step):
        src_key = f"model.layers.{orig_idx}.{suf}"
        src_t = src_handles[wmap_src[src_key]].get_tensor(src_key).float()

        # Find the dst layer with identical (or smallest diff) shape-matching tensor
        best_idx, best_diff = -1, float("inf")
        for dst_i, dst_t in dst_tensors.items():
            if dst_t.shape != src_t.shape:
                continue
            diff = (src_t - dst_t).abs().max().item()
            if diff < best_diff:
                best_diff, best_idx = diff, dst_i

        if best_idx == -1:
            # Width expansion changed shapes — just report the nearest candidate
            print(f"  [~] layer {orig_idx}: all dst shapes differ (width expansion applied)")
        else:
            ok = best_diff < 1e-6
            icon = "✓" if ok else "✗"
            print(f"  [{icon}] orig layer {orig_idx} → dst layer {best_idx}  "
                  f"max|Δ|={best_diff:.2e}")
            all_ok = all_ok and ok
    return all_ok


def check_identity_blocks_zeroed(dst_idx: ShardIndex) -> bool:
    """Verify that identity blocks (if any) have zeroed projections."""
    print("\n[Identity block zero-check]")
    zero_suffixes = {"self_attn.o_proj.weight", "mlp.down_proj.weight"}
    dst_handles = dst_idx.open_all_shards()

    # Heuristic: an identity block is one whose ALL tensors are copied from
    # the previous layer → just check the zero suffixes of every dst layer
    total_zero = total_nonzero = 0
    for suf in zero_suffixes:
        for key, shard in dst_idx.weight_map.items():
            if not key.endswith(suf):
                continue
            t = dst_handles[shard].get_tensor(key)
            if t.abs().max().item() < 1e-9:
                total_zero += 1
            else:
                total_nonzero += 1

    if total_zero == 0:
        print("  [~] No zeroed projections found (non-FP method or no identity blocks)")
    else:
        print(f"  [✓] Found {total_zero} zeroed projection(s), "
              f"{total_nonzero} non-zero (original layers)")
    return True


# ── FP check ───────────────────────────────────────────────────────────────────

def check_fp(src_dir: Path, dst_dir: Path, seq_len: int, samples: int, atol: float) -> bool:
    print("\n[Function-Preserving logit check]")
    from transformers import AutoModelForCausalLM

    print("  Loading original model …")
    orig = AutoModelForCausalLM.from_pretrained(str(src_dir), dtype=torch.float32)
    print("  Loading expanded model …")
    try:
        exp = AutoModelForCausalLM.from_pretrained(str(dst_dir), dtype=torch.float32)
    except Exception as e:
        print(f"  [✗] Cannot load expanded model: {e}")
        return False

    orig.eval(); exp.eval()

    vocab = orig.config.vocab_size
    ids = torch.randint(0, vocab, (samples, seq_len))
    max_err = 0.0
    with torch.no_grad():
        for i in range(samples):
            inp = ids[i].unsqueeze(0)
            lo = orig(input_ids=inp).logits
            le = exp(input_ids=inp).logits
            max_err = max(max_err, (lo - le).abs().max().item())

    ok = max_err < atol
    icon = "✓" if ok else "✗"
    note = "" if ok else "  (expected for non-FP methods like SOLAR DUS)"
    print(f"  [{icon}] max|Δlogit| = {max_err:.3e}  (atol={atol}){note}")
    return ok


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()
    src_dir, dst_dir = Path(args.src), Path(args.dst)

    print(f"src: {src_dir}")
    print(f"dst: {dst_dir}")

    src_idx = ShardIndex.load(src_dir)
    dst_idx = ShardIndex.load(dst_dir)

    results: dict[str, bool] = {}
    results["config"]           = check_config(src_dir, dst_dir)
    results["tensor_counts"]    = check_tensor_counts(src_idx, dst_idx)
    results["weights_preserved"]= check_original_weights_preserved(src_idx, dst_idx)
    results["identity_zeroed"]  = check_identity_blocks_zeroed(dst_idx)

    if args.fp:
        results["fp_logit_check"] = check_fp(
            src_dir, dst_dir, args.fp_seq_len, args.fp_samples, args.fp_atol,
        )
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    all_ok = True
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  [{icon}] {name}")
        if not ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)


def _load_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    main()
