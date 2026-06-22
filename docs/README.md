# llm-grow Documentation

## Tutorials

| Model | Architecture | Params | Layers | Experts | Guide |
|-------|:---:|:---:|:---:|:---:|-------|
| Qwen3-0.6B | Dense | 596M | 28 | — | [expand_qwen3_0.6b.md](expand_qwen3_0.6b.md) |
| Qwen3-30B-A3B | MoE Standard | ~30B | 48 | 128 | [expand_qwen3_30b_a3b.md](expand_qwen3_30b_a3b.md) |
| Kimi-K2 | DeepSeek MoE | ~1T | 61 | 384 | [expand_kimi_k2.md](expand_kimi_k2.md) |
| LongCat-Flash | LongCat MoE | ~0.5T | 28 | 512 | [expand_longcat_flash.md](expand_longcat_flash.md) |

## Quick Reference

### CLI Commands

```bash
# Expand
llm-grow expand --src <model> --dst <output> --method <depth|expert|width> [opts]

# Verify
llm-grow verify --src <original> --dst <expanded> [--fp]

# Inspect
llm-grow info --src <model>
```

### Expansion Methods at a Glance

| Method | CLI `--method` | FP | Best For |
|--------|:---:|:---:|------|
| ZeroBlockInsert | `depth` | ✓ | Dense depth expansion |
| MultiAxisPad | `width` | ✓ | Dense depth+width |
| OverlapCopy | (Python API) | ✗ | Simplest, data-abundant |
| SVDInterpInsert | (Python API) | ≈ | Faster convergence than OverlapCopy |
| ExpertClone | `expert` | ≈ | MoE expert doubling |
| DenseToMoE | (Python API) | ✗ | Dense→MoE conversion |

### Architecture Families

| Family | Detection | Models |
|--------|-----------|--------|
| `dense` | No `mlp.experts.*` keys | Qwen2.5/3.0 Dense, LLaMA, Mistral |
| `standard_moe` | `mlp.experts.*` + `mlp.gate.weight` | Qwen3-30B-A3B, Mixtral |
| `deepseek_moe` | MLA + fp8 + shared expert + dense layer 0 | Kimi-K2, DeepSeek-V2/V3 |
| `longcat` | Dual `self_attn.{0,1}` + dual `mlps.{0,1}` | LongCat-Flash |

### Typical CPT Budgets

| Method | Tokens | Phase 1 (frozen) | Phase 2 (full) |
|--------|:---:|:---:|:---:|
| ZeroBlockInsert | 8–16B | 12B, lr=2e-4 | 6B, lr=5e-5 |
| MultiAxisPad | 30–60B | 40B, lr=2e-4 | 20B, lr=5e-5 |
| ExpertClone | 30–50B | — (single phase) | lr=3e-5 |
| DenseToMoE | 50–100B | — (single phase) | lr=3e-5 + load balance |

### Model Paths (Local)

```bash
# Dense
/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-0.5B
/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B

# MoE
/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B
/home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K2-Thinking
/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite

# Cached expansions
/home/jianzhnie/llmtuner/hfhub/cache/
```

## Diagrams

| Diagram | File |
|---------|------|
| ZeroBlockInsert | [images/zero_block_insert.svg](images/zero_block_insert.svg) |
| OverlapCopy | [images/overlap_copy.svg](images/overlap_copy.svg) |
| SVDInterpInsert | [images/svd_interp_insert.svg](images/svd_interp_insert.svg) |
| MultiAxisPad | [images/multi_axis_pad.svg](images/multi_axis_pad.svg) |
| DenseToMoE | [images/dense_to_moe.svg](images/dense_to_moe.svg) |
| ExpertClone | [images/expert_clone.svg](images/expert_clone.svg) |
| Logo | [images/logo.svg](images/logo.svg) |
