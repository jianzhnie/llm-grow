#!/usr/bin/env python
"""Test auto_expand detection and dispatch across Dense + MoE models."""
import sys

MODELS = {
    "dense_qwen3_0.6b": "/Users/robin/hfhub/models/Qwen/Qwen3-0.6B",
    "moe_qwen3_30b":    "/Users/robin/hfhub/models/Qwen/Qwen3-30B-A3B",
    "moe_kimi_k2":      "/Users/robin/hfhub/models/moonshotai/Kimi-K2-Base",
    "moe_longcat":      "/Users/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat",
}


def test_detect(name, path):
    from llm_grow.safetensor.detect import detect_model
    p = detect_model(path)
    print(f"\n{'─'*60}")
    print(p.summary())
    return p


def test_auto_dry_run(name, path, method, **kwargs):
    from llm_grow.safetensor.auto import auto_expand
    print(f"\n  → auto_expand(method={method!r}, dry_run=True)")
    try:
        auto_expand(path, f"/tmp/auto_test/{name}", method=method,
                    verbose=False, dry_run=True, **kwargs)
        print(f"  [OK] {name} / {method}")
        return True
    except (ValueError, NotImplementedError) as e:
        print(f"  [expected-error] {e}")
        return "expected"
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    results = {}

    # ── 1. Detection ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Detection")
    print("="*60)
    profiles = {}
    for name, path in MODELS.items():
        try:
            profiles[name] = test_detect(name, path)
            results[f"detect/{name}"] = True
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            results[f"detect/{name}"] = False

    # ── 2. Verify detection is correct ───────────────────────────────────────
    print("\n" + "="*60)
    print("  Detection assertions")
    print("="*60)

    checks = {
        ("dense_qwen3_0.6b", "is_moe"):         (False,  lambda p: p.is_moe),
        ("dense_qwen3_0.6b", "family"):          ("dense",lambda p: p.family),
        ("moe_qwen3_30b",    "is_moe"):          (True,   lambda p: p.is_moe),
        ("moe_qwen3_30b",    "family"):          ("standard_moe", lambda p: p.family),
        ("moe_qwen3_30b",    "experts"):         (128,    lambda p: p.experts_per_moe_layer),
        ("moe_kimi_k2",      "is_moe"):          (True,   lambda p: p.is_moe),
        ("moe_kimi_k2",      "family"):          ("deepseek_moe", lambda p: p.family),
        ("moe_kimi_k2",      "experts"):         (384,    lambda p: p.experts_per_moe_layer),
        ("moe_kimi_k2",      "has_fp8"):         (True,   lambda p: p.has_fp8),
        ("moe_kimi_k2",      "dense_layer_0"):   (True,   lambda p: 0 in p.dense_only_layers),
        ("moe_longcat",      "family"):          ("longcat", lambda p: p.family),
        ("moe_longcat",      "has_dual_attn"):   (True,   lambda p: p.has_dual_attn),
    }

    for (model_name, check_name), (expected, getter) in checks.items():
        if model_name not in profiles:
            continue
        actual = getter(profiles[model_name])
        ok = actual == expected
        icon = "✓" if ok else "✗"
        print(f"  [{icon}] {model_name}.{check_name}: {actual!r} (expected {expected!r})")
        results[f"check/{model_name}.{check_name}"] = ok

    # ── 3. auto_expand dispatch ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  auto_expand dispatch (dry_run)")
    print("="*60)

    scenarios = [
        # Dense: depth OK, expert should error, width OK
        ("dense_qwen3_0.6b",  "depth",  {}, True),
        ("dense_qwen3_0.6b",  "expert", {}, "expected"),   # should raise ValueError
        ("dense_qwen3_0.6b",  "width",  {"ffn_size_expansion": 256}, True),

        # MoE Qwen3: depth OK, expert OK, width should error
        ("moe_qwen3_30b",     "depth",  {"num_new_layers": 4}, True),
        ("moe_qwen3_30b",     "expert", {"expand_factor": 2}, True),
        ("moe_qwen3_30b",     "width",  {}, "expected"),   # should raise NotImplementedError

        # MoE Kimi-K2: depth OK, expert OK
        ("moe_kimi_k2",       "depth",  {"num_new_layers": 4}, True),
        ("moe_kimi_k2",       "expert", {"expand_factor": 2}, True),

        # LongCat: depth OK, expert OK
        ("moe_longcat",       "depth",  {"num_new_layers": 4}, True),
        ("moe_longcat",       "expert", {"expand_factor": 2}, True),
    ]

    for model_name, method, kwargs, want in scenarios:
        if model_name not in MODELS:
            continue
        path = MODELS[model_name]
        result = test_auto_dry_run(model_name, path, method, **kwargs)
        key = f"auto/{model_name}/{method}"
        if want == "expected":
            ok = result == "expected"
        else:
            ok = result is True
        results[key] = ok

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    fails = []
    for k, v in results.items():
        icon = "✓" if v else "✗"
        print(f"  [{icon}] {k}")
        if not v:
            fails.append(k)

    if fails:
        print(f"\n  {len(fails)} failure(s): {fails}")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} checks passed!")


if __name__ == "__main__":
    main()
