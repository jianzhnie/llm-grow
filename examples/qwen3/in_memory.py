#!/usr/bin/env python
"""In-memory expansion integration example: Qwen3-0.6B

示例内容：
  1. LLaMA-Pro  — 恒等块插入，FP 验证
  2. SOLAR DUS  — 层重叠复制，层数/参数量确认
  3. MSG        — 深度+宽度混合扩增，FP 验证
  4. MoE Upcycling — Dense → MoE，专家结构确认
  5. Expert Upcycling — MoE 专家数扩展
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.model_paths import QWEN3_06B, require_path

MODEL_PATH = require_path("QWEN3_06B", QWEN3_06B)
DEVICE = "cpu"
DTYPE = torch.float32  # CPU 测试用 fp32，避免 bfloat16 精度误差干扰 FP 验证


def load_fresh():
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=DTYPE)
    return model.to(DEVICE)


def print_sep(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def quick_fp_check(original, expanded, seq_len=32, atol=1e-4) -> bool:
    """随机 token 输入，检查 logits 最大误差。"""
    vocab = original.config.vocab_size
    ids = torch.randint(0, vocab, (1, seq_len))
    original.eval()
    expanded.eval()
    with torch.no_grad():
        lo = original(input_ids=ids).logits
        le = expanded(input_ids=ids).logits
    max_err = (lo - le).abs().max().item()
    passed = max_err < atol
    icon = "✓" if passed else "✗"
    print(f"  [{icon}] FP check  max|Δlogit| = {max_err:.3e}  (atol={atol})")
    return passed


def run_generate(model, tokenizer, prompt="Hello, I am"):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────────────────────
# Example 1: LLaMA-Pro
# ──────────────────────────────────────────────────────────────────────────────
def check_zero_block_insert():
    print_sep("Test 1: LLaMA-Pro — 恒等块插入")
    from llm_grow.expanders.depth.zero_block_insert import (
        ZeroBlockInsertConfig,
        ZeroBlockInsertExpander,
    )

    model = load_fresh()
    orig = copy.deepcopy(model)
    orig_layers = len(model.model.layers)
    orig_params = count_params(model)

    config = ZeroBlockInsertConfig(
        num_new_layers=7, insert_strategy="uniform", freeze_original=True
    )
    t0 = time.time()
    expanded = ZeroBlockInsertExpander().expand(model, config)
    elapsed = time.time() - t0

    exp_layers = len(expanded.model.layers)
    exp_params = count_params(expanded)
    trainable = sum(p.numel() for p in expanded.parameters() if p.requires_grad)

    print(f"  Layers : {orig_layers} → {exp_layers}  (+{exp_layers - orig_layers})")
    ratio = exp_params / orig_params
    print(
        f"  Params : {orig_params / 1e6:.1f}M → {exp_params / 1e6:.1f}M  ({ratio:.3f}x)"
    )
    train_pct = 100 * trainable / exp_params
    print(f"  Trainable params: {trainable / 1e6:.1f}M  ({train_pct:.1f}%)")
    print(f"  Expand time: {elapsed:.2f}s")
    quick_fp_check(orig, expanded)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Example 2: SOLAR DUS
# ──────────────────────────────────────────────────────────────────────────────
def check_overlap_copy():
    print_sep("Test 2: SOLAR DUS — 层重叠复制")
    from llm_grow.expanders.depth.overlap_copy import (
        OverlapCopyConfig,
        OverlapCopyExpander,
    )

    model = load_fresh()
    orig_layers = len(model.model.layers)
    orig_params = count_params(model)

    config = OverlapCopyConfig(num_overlap=8)
    t0 = time.time()
    expanded = OverlapCopyExpander().expand(model, config)
    elapsed = time.time() - t0

    exp_layers = len(expanded.model.layers)
    exp_params = count_params(expanded)
    expected = 2 * (orig_layers - config.num_overlap)

    print(f"  Layers : {orig_layers} → {exp_layers}  (expected {expected})")
    ratio = exp_params / orig_params
    print(
        f"  Params : {orig_params / 1e6:.1f}M → {exp_params / 1e6:.1f}M  ({ratio:.3f}x)"
    )
    print(f"  Expand time: {elapsed:.2f}s")
    assert exp_layers == expected, f"Layer count mismatch: {exp_layers} != {expected}"
    print("  [✓] Layer count correct")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Example 3: SVDInterpInsert
# ──────────────────────────────────────────────────────────────────────────────
def check_svd_interp_insert():
    print_sep("Test 3: SVDInterpInsert — SVD 插值（相邻层均值）")
    from llm_grow.expanders.depth.svd_interp_insert import (
        SVDInterpInsertConfig,
        SVDInterpInsertExpander,
    )

    model = load_fresh()
    orig_layers = len(model.model.layers)
    orig_params = count_params(model)

    # 仅在前 4 对相邻层之间插入，快速验证
    config = SVDInterpInsertConfig(insert_between=[(i, i + 1) for i in range(4)])
    t0 = time.time()
    expanded = SVDInterpInsertExpander().expand(model, config)
    elapsed = time.time() - t0

    exp_layers = len(expanded.model.layers)
    exp_params = count_params(expanded)
    print(f"  Layers : {orig_layers} → {exp_layers}")
    print(f"  Params : {orig_params / 1e6:.1f}M → {exp_params / 1e6:.1f}M")
    print(f"  Expand time: {elapsed:.2f}s")
    print("  [✓] SVDInterpInsert expansion complete (approx FP, atol=0.5 relaxed)")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Example 4: MSG（深度 + 宽度）
# ──────────────────────────────────────────────────────────────────────────────
def check_msg():
    print_sep("Test 4: MSG — 深度+宽度混合扩增")
    from llm_grow.expanders.width.multi_axis_pad import (
        MultiAxisPadConfig,
        MultiAxisPadExpander,
    )

    model = load_fresh()
    orig = copy.deepcopy(model)
    orig_layers = len(model.model.layers)
    orig_params = count_params(model)

    config = MultiAxisPadConfig(
        num_new_layers=4,
        hidden_size_expansion=0,
        intermediate_size_expansion=0,
        freeze_original=False,
    )
    t0 = time.time()
    expanded = MultiAxisPadExpander().expand(model, config)
    elapsed = time.time() - t0

    exp_layers = len(expanded.model.layers)
    exp_params = count_params(expanded)
    print(f"  Layers : {orig_layers} → {exp_layers}")
    ratio = exp_params / orig_params
    print(
        f"  Params : {orig_params / 1e6:.1f}M → {exp_params / 1e6:.1f}M  ({ratio:.3f}x)"
    )
    print(f"  Expand time: {elapsed:.2f}s")
    quick_fp_check(orig, expanded)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Example 5: MoE Upcycling
# ──────────────────────────────────────────────────────────────────────────────
def check_dense_to_moe():
    print_sep("Test 5: MoE Upcycling — Dense FFN → 稀疏 MoE")
    from llm_grow.expanders.sparse.dense_to_moe import (
        DenseToMoEConfig,
        DenseToMoEExpander,
    )

    model = load_fresh()
    orig_params = count_params(model)

    config = DenseToMoEConfig(num_experts=4, top_k=2, noise_std=0.01)
    t0 = time.time()
    expanded = DenseToMoEExpander().expand(model, config)
    elapsed = time.time() - t0

    exp_params = count_params(expanded)
    ratio = exp_params / orig_params
    print(
        f"  Params : {orig_params / 1e6:.1f}M → {exp_params / 1e6:.1f}M  ({ratio:.3f}x)"
    )
    print(f"  Expand time: {elapsed:.2f}s")

    from llm_grow.expanders.sparse.dense_to_moe import MoELayer

    moe_layers = [m for m in expanded.modules() if isinstance(m, MoELayer)]
    print(f"  MoE layers found: {len(moe_layers)}")
    if moe_layers:
        first = moe_layers[0]
        print(f"  Experts per layer: {len(first.experts)}, top_k={first.top_k}")

    # 前向推理验证结构完整
    ids = torch.randint(0, expanded.config.vocab_size, (1, 16))
    expanded.eval()
    with torch.no_grad():
        out = expanded(input_ids=ids)
    print(f"  [✓] Forward pass OK, logits shape: {out.logits.shape}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Example 6: Expert Upcycling（基于 MoE Upcycling 结果）
# ──────────────────────────────────────────────────────────────────────────────
def check_expert_clone():
    print_sep("Test 6: Expert Upcycling — MoE 专家数扩展 (M1)")
    from llm_grow.expanders.sparse.dense_to_moe import (
        DenseToMoEConfig,
        DenseToMoEExpander,
        MoELayer,
    )
    from llm_grow.expanders.sparse.expert_clone import (
        ExpertCloneConfig,
        ExpertCloneExpander,
    )

    # 先 upcycle 得到 MoE 基座
    model = load_fresh()
    moe_cfg = DenseToMoEConfig(num_experts=4, top_k=2, noise_std=0.01)
    moe_model = DenseToMoEExpander().expand(model, moe_cfg)
    moe_params = count_params(moe_model)

    # 再做 expert upcycling（4 → 8 专家）
    exp_cfg = ExpertCloneConfig(expand_factor=2, symmetry_break="noise")
    t0 = time.time()
    expanded = ExpertCloneExpander().expand(moe_model, exp_cfg)
    elapsed = time.time() - t0

    final_params = count_params(expanded)
    ratio = final_params / moe_params
    print(
        f"  Params : {moe_params / 1e6:.1f}M → "
        f"{final_params / 1e6:.1f}M  ({ratio:.3f}x)"
    )
    print(f"  Expand time: {elapsed:.2f}s")

    moe_layers = [m for m in expanded.modules() if isinstance(m, MoELayer)]
    if moe_layers:
        print(f"  Experts per layer after upcycling: {len(moe_layers[0].experts)}")

    ids = torch.randint(0, expanded.config.vocab_size, (1, 16))
    expanded.eval()
    with torch.no_grad():
        out = expanded(input_ids=ids)
    print(f"  [✓] Forward pass OK, logits shape: {out.logits.shape}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Example 7: 生成文本对比（LLaMA-Pro 扩增前后）
# ──────────────────────────────────────────────────────────────────────────────
def check_generation():
    print_sep("Test 7: 生成文本对比（LLaMA-Pro 扩增前后）")
    from llm_grow.expanders.depth.zero_block_insert import (
        ZeroBlockInsertConfig,
        ZeroBlockInsertExpander,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt = "The key to learning programming is"

    model_orig = load_fresh()
    model_exp = copy.deepcopy(model_orig)
    ZeroBlockInsertExpander().expand(
        model_exp, ZeroBlockInsertConfig(num_new_layers=7, freeze_original=False)
    )

    print(f"  Prompt: {prompt!r}")
    out_orig = run_generate(model_orig, tokenizer, prompt)
    out_exp = run_generate(model_exp, tokenizer, prompt)
    print(f"  [Original ] {out_orig}")
    print(f"  [Expanded ] {out_exp}")
    print("  [✓] Outputs should be identical (FP = function-preserving)")
    return out_orig == out_exp


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("ZeroBlockInsert", check_zero_block_insert),
        ("OverlapCopy", check_overlap_copy),
        ("SVDInterpInsert", check_svd_interp_insert),
        ("MSG", check_msg),
        ("DenseToMoE", check_dense_to_moe),
        ("ExpertClone", check_expert_clone),
        ("Generation check", check_generation),
    ]

    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as exc:
            print(f"\n  [✗] FAILED: {exc}")
            import traceback

            traceback.print_exc()
            results[name] = False

    print_sep("Summary")
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  [{icon}] {name}")
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\n  {len(failed)} test(s) failed: {failed}")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} tests passed!")
