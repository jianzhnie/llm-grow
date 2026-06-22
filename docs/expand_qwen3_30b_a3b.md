# Qwen3-30B-A3B Expansion Guide

**Architecture**: MoE Standard (`Qwen3MoeForCausalLM`)
**Parameters**: ~30B total, ~3B active (top-8 routing)
**Model**: `Qwen/Qwen3-30B-A3B` (HuggingFace) or local path

---

## Architecture

```
Layers:               48
Hidden size:          2048
Expert FFN width:     768 (moe_intermediate_size)
Experts per layer:    128
Active experts:       8 (num_experts_per_tok)
Attention heads:      32 / KV heads: 4 (GQA)
Vocab size:           151,936 (untied embedding)
```

Expert parameters account for ~96% of total (128 experts × 4.72M × 48 layers ≈ 29B).
Expert count expansion is the most efficient path to 2× scale.

| | Dense | Qwen3-30B-A3B (MoE) |
|---|---|---|
| Identity block zeros | 1 down_proj | **128** expert down_projs |
| Expansion axes | Depth / Width | Expert count ★ / Depth |
| Inference cost after expand | Linear with depth | **Nearly unchanged** for expert |

---

## Expansion Options

| Method | Config | Target | Ratio | Inference |
|--------|--------|:---:|:---:|:---:|
| **Expert 2×** ★ | 128→256 experts | ~59B | ~2.0× | top-8→16 (or unchanged) |
| Expert 2× (same cost) | 128→256, topk unchanged | ~59B | ~2.0× | **unchanged** |
| Depth | 48→60 layers | ~34B | ~1.13× | ↑ linear |

---

## Quickstart

```bash
# Install
pip install -e .

# Dry-run (plan only — no weights needed if index.json present)
llm-grow expand \
    --src /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B \
    --dst /tmp/qwen3_30b_2x \
    --method expert --expand-factor 2 --dry-run

# Execute expansion (parallel writing for speed)
llm-grow expand \
    --src /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B \
    --dst ./outputs/Qwen3-30B-A3B-2x \
    --method expert --expand-factor 2 \
    --workers 8 --validate-output

# Verify (structural only — model is 30B)
llm-grow verify \
    --src /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B \
    --dst ./outputs/Qwen3-30B-A3B-2x
```

Expected:

```
  source:  18,867 tensors, 16 shard(s)
  output:  37,299 tensors, num_hidden_layers → 48
  dup-rows: 48 (router.weight rows doubled per layer)
  new keys: 18,432 (128 experts × 48 layers × 3 tensors)
  config:  {num_experts: 256, num_experts_per_tok: 16}
```

---

## Keep Inference Cost Unchanged

Double expert count while keeping top-8 activation (same per-token FLOPs):

```python
from llm_grow.safetensor.models.moe_generic import (
    GenericDenseToMoEConfig, GenericMoEExpertCloneExpander,
)

expander = GenericMoEExpertCloneExpander(
    GenericDenseToMoEConfig(
        expand_factor=2,
        noise_scale=1e-6,
        router_weight_suffixes=["mlp.gate.weight"],
        config_expert_count_key="num_experts",
        config_topk_key="num_experts_per_tok",
        scale_topk=False,  # keep top-8, don't double
    )
)
expander.expand(
    src_dir="/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B",
    dst_dir="./outputs/Qwen3-30B-A3B-2x-top8",
    workers=8,
)
```

---

## Python API

**Factory (simplest)**:

```python
from llm_grow.safetensor.models.moe_generic import make_qwen3moe_expert_clone

make_qwen3moe_expert_clone(expand_factor=2, noise_scale=1e-6).expand(
    src_dir="/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B",
    dst_dir="./outputs/Qwen3-30B-A3B-2x",
    workers=8,
)
```

**auto_expand (architecture-agnostic)**:

```python
from llm_grow.safetensor.auto import auto_expand

auto_expand(
    src_dir="/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B",
    dst_dir="./outputs/Qwen3-30B-A3B-2x",
    method="expert",
    expand_factor=2,
    workers=8,
    validate_output=True,
)
```

---

## Continued Pre-training

Single-phase full-parameter CPT (no phase-1 freeze — all experts must specialize):

```
Tokens:               30–50B
Learning rate:        3e-5, cosine, warmup 2,000 steps
Batch size:           per_device=1, grad_accum=16 (~4M tokens global)
MoE balance loss:     balance_coeff=1e-2, z_coeff=1e-3
Data mix:             Web 60% + Code 20% + Math 10% + Science 10%
Evaluation:           Every 5B tokens — monitor expert activation frequency
```

```python
from llm_grow.training.load_balance import combined_moe_loss

loss = combined_moe_loss(
    lm_loss=lm_loss,
    router_logits_list=router_logits_list,
    num_experts=256,
    top_k=16,
    balance_coeff=1e-2,
    z_coeff=1e-3,
)
```
