# llm-grow

<p align="center">
  <img src="docs/images/logo.svg" width="280"/>
</p>

<p align="center">
  <em>Grow Larger Models from Existing LLM Checkpoints — Layer by Layer</em>
</p>

<p align="center">
  <a href="#installation">Install</a> &bull;
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#expansion-methods">Methods</a> &bull;
  <a href="#method-selection">Selection Guide</a> &bull;
  <a href="#api-reference">API</a> &bull;
  <a href="#training">Training</a> &bull;
  <a href="#benchmarks">Benchmarks</a> &bull;
  <a href="README_zh.md">中文文档</a>
</p>

<p align="center">
<img src="docs/images/llmgrow.png" alt="LLM-Grow overview">
</p>

---

A modular toolkit for **growing** larger models from existing LLM checkpoints.
No pre-training from scratch — expand depth, width, or expert count with
function-preserving guarantees where available.

**Key Features**

| | |
|---|---|
| **Two-tier expansion** | In-memory (`nn.Module`) and Safetensor-level (mmap streaming, peak ≤ 4 GB) |
| **Auto-detection** | Dense / MoE-Standard / DeepSeek-MoE / LongCat — detected from `config.json` alone |
| **6 algorithms** | ZeroBlockInsert, OverlapCopy, SVDInterpInsert, MultiAxisPad, DenseToMoE, ExpertClone |
| **Function-Preserving** | ZeroBlockInsert / MultiAxisPad → zero accuracy loss at expansion time |
| **Pluggable noise** | Gaussian / Uniform / ScaledGaussian strategies for symmetry breaking |
| **Registry** | Decorator-based `@register_expander` for in-memory expanders |
| **Training toolkit** | Frozen training, distillation, progressive mask growth, MoE load balancing |
| **Verified on** | Qwen2.5-0.5B, Qwen3-0.6B, Qwen3-30B-A3B, LongCat-Flash-Lite, Kimi-K2-Thinking |

---

## Installation

```bash
pip install -e .                # Core (expansion + CLI)
pip install -e ".[train]"       # + Training deps (DeepSpeed, Flash-Attn, Datasets)
pip install -e ".[eval]"        # + Evaluation deps (lm-eval-harness)
pip install -e ".[dev]"         # + Dev tools (pytest, ruff, mypy)
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.2, Transformers ≥ 4.40, safetensors ≥ 0.4

---

## Quickstart

### CLI

```bash
# Depth expansion (auto-detects Dense / MoE)
llm-grow expand --src /path/to/model --dst ./output \
    --method depth --num-new-layers 4

# MoE expert expansion
llm-grow expand --src /path/to/moe_model --dst ./output \
    --method expert --expand-factor 2

# FFN width expansion
llm-grow expand --src /path/to/model --dst ./output \
    --method width --ffn-size-expansion 512

# Dry-run (plan only, no file writes)
llm-grow expand --src /path/to/model --dst /tmp/out \
    --method depth --num-new-layers 4 --dry-run

# Parallel writing + validation + resume support
llm-grow expand --src /path/to/model --dst ./output \
    --method depth --num-new-layers 4 \
    --workers 8 --validate-output --resume

# Verify expansion (structural + FP logit comparison)
llm-grow verify --src /path/to/original --dst /path/to/expanded --fp

# Display model architecture info
llm-grow info --src /path/to/model
```

### Python API — Safetensor-Level (large models, no weight loading)

```python
from llm_grow.safetensor.auto import auto_expand

auto_expand(
    src_dir="/path/to/model",
    dst_dir="./expanded",
    method="depth",             # "depth" | "expert" | "width"
    num_new_layers=4,
    insert_strategy="uniform",  # "uniform" | "front" | "rear"
    target_shard_gb=4.0,
    workers=4,                  # parallel writer threads
    dry_run=False,
    validate_output=True,
    resume=False,
)
```

### Python API — In-Memory (small models / rapid prototyping)

```python
import copy
from transformers import AutoModelForCausalLM
from llm_grow.expanders.registry import get_expander
from llm_grow.expanders.depth.zero_block_insert import ZeroBlockInsertConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype="auto")
original = copy.deepcopy(model)

expander = get_expander("zero_block_insert")()
expanded = expander.expand(
    model,
    ZeroBlockInsertConfig(num_new_layers=9, freeze_original=True),
)
expander.verify(original, expanded)  # max|Δlogit| = 0.0000e+00
```

---

## Expansion Methods

### Comparison

| Method | FP | Axis | Accuracy | CPT Needed | Latency |
|--------|:---:|:---:|:---:|:---:|:---:|
| **ZeroBlockInsert** | ✓ | Depth | **100%** | 8–16B | +linear |
| **OverlapCopy** | ✗ | Depth | 50–80% | 100B+ | +linear |
| **SVDInterpInsert** | ≈ | Depth | 80–90% | <50B | +linear |
| **MultiAxisPad** | ✓ | Depth+Width | **100%** | 30–60B | ~1.4× |
| **DenseToMoE** | ✗ | Dense→MoE | 70–85% | 50–100B | ≈same |
| **ExpertClone** | ≈ | Expert count | — | saves 32–67% | ≈same |

> **FP** = Function-Preserving: output logits are identical to the original model after expansion.

### Visual Overview

<p align="center">
  <img src="docs/images/zero_block_insert.svg" width="48%"/>
  <img src="docs/images/overlap_copy.svg" width="48%"/>
  <img src="docs/images/svd_interp_insert.svg" width="48%"/>
  <img src="docs/images/multi_axis_pad.svg" width="48%"/>
  <img src="docs/images/dense_to_moe.svg" width="48%"/>
  <img src="docs/images/expert_clone.svg" width="48%"/>
</p>

---

## Method Selection

```
Need to expand parameters
│
├── Very large model (>30B, cannot load fully) → Safetensor-level
│   ├── Dense → depth:   llm-grow expand --method depth
│   │         → width:   llm-grow expand --method width
│   └── MoE   → expert:  llm-grow expand --method expert
│             → depth:   llm-grow expand --method depth
│
└── Small/medium model (fits in memory) → In-memory
    ├── Accuracy priority, limited data → ZeroBlockInsert (FP, 8–16B CPT)
    ├── Precise 2×, control latency     → MultiAxisPad (depth+width, ~1.4×)
    ├── Simplest, data abundant         → OverlapCopy
    ├── Cannot increase latency         → DenseToMoE (top-1 activation unchanged)
    └── Model is already MoE            → ExpertClone
```

---

## API Reference

### Architecture

| | In-Memory (`expanders/`) | Safetensor-Level (`safetensor/`) |
|---|---|---|
| **Input** | `nn.Module` | `.safetensors` directory |
| **Output** | `nn.Module` | `.safetensors` directory |
| **Peak memory** | Full model | ≤ 1 output shard (~4 GB) |
| **FP verify** | Direct logit comparison | Structural + optional logit compare |
| **Use case** | Small models / prototyping | 100B+ models |

### Auto-Detection

```python
from llm_grow.safetensor.detect import detect_model

profile = detect_model("/path/to/model")
print(profile.family)  # "dense" | "standard_moe" | "deepseek_moe" | "longcat"
```

| Family | Example Models | Identifiers |
|--------|---------------|-------------|
| `dense` | Qwen3-0.6B/8B/14B/32B | No `mlp.experts.*` keys |
| `standard_moe` | Qwen3-30B-A3B | `mlp.experts.*` + `mlp.gate.weight` |
| `deepseek_moe` | Kimi-K2-Thinking | MLA + fp8 `weight_scale_inv` + shared expert + dense layer 0 |
| `longcat` | LongCat-Flash-Lite | Dual `self_attn.{0,1}` + dual `mlps.{0,1}` + `mlp.router.classifier` |

### `auto_expand()` — Unified Entry Point

```python
from llm_grow.safetensor.auto import auto_expand

auto_expand(
    src_dir="/path/to/model",
    dst_dir="./expanded",
    method="depth",               # "depth" | "expert" | "width"
    num_new_layers=4,             # [depth] layers to insert
    insert_strategy="uniform",    # [depth] "uniform" | "front" | "rear"
    expand_factor=2,              # [expert] expert multiplier
    noise_scale=1e-6,             # [expert] router noise scale
    ffn_size_expansion=0,         # [width] intermediate_size increment
    target_shard_gb=4.0,          # output shard size limit
    workers=1,                    # parallel writer threads
    dry_run=False,                # True = plan only
    validate_output=False,        # True = post-write validation
    resume=False,                 # True = skip existing shards
)
```

### Factory Functions (Pre-configured)

| Function | Target | Notes |
|----------|--------|-------|
| `make_qwen3moe_expert_clone(factor)` | Qwen3-30B-A3B | Router: `mlp.gate.weight` |
| `make_qwen3moe_zero_block_insert(n)` | Qwen3-30B-A3B | Zeros: o_proj + all expert down_proj |
| `make_kimik2_expert_clone(factor)` | Kimi-K2 | fp8-aware, bias noise=0, shared expert preserve |
| `make_kimik2_zero_block_insert(n)` | Kimi-K2 | Dense layer-0 aware, shared-expert aware |

### In-Memory Expand via Registry

```python
from llm_grow.expanders.registry import get_expander, list_expanders

print(list_expanders())
# ['dense_to_moe', 'expert_clone', 'multi_axis_pad',
#  'overlap_copy', 'svd_interp_insert', 'zero_block_insert']

expander = get_expander("zero_block_insert")()
```

### Noise Strategies (Pluggable)

```python
from llm_grow.initializers.noise import GaussianNoise, UniformNoise, ScaledGaussianNoise
from llm_grow.expanders.sparse.dense_to_moe import DenseToMoEConfig, DenseToMoEExpander

# Default: Gaussian noise
config = DenseToMoEConfig(num_experts=8, noise_std=0.01)

# Use uniform noise instead
config = DenseToMoEConfig(num_experts=8, noise_std=0.01, noise=UniformNoise())

# Scaled Gaussian (std relative to tensor std, like safetensor dup_rows_noise_scale)
config = DenseToMoEConfig(num_experts=8, noise=ScaledGaussianNoise())
```

### ExpansionPlan Serialization

```python
from llm_grow.safetensor.recipe import ExpansionPlan

plan.save_json("plan.json")          # Save for offline review
plan = ExpansionPlan.load_json("plan.json")  # Reload

plan.to_dict()                       # Programmatic inspection
dict(plan.config_patches)            # {'num_experts': 256, ...}
```

---

## Training

### Two-Phase Frozen Training

```python
from llm_grow.training.freeze import (
    snapshot_param_ids, mark_new_params, freeze_original_layers, unfreeze_all,
)

original_ids = snapshot_param_ids(model)
expander.expand(model, config)
mark_new_params(model, original_ids)

freeze_original_layers(model)     # Phase 1: train only new parameters
train(model, phase1_data, lr=2e-4)

unfreeze_all(model)               # Phase 2: full fine-tune
train(model, phase2_data, lr=1e-5)
```

### Knowledge Distillation

```python
from llm_grow.training.distillation import DistillConfig, DistillationLoss

criterion = DistillationLoss(DistillConfig(temperature=2.0, alpha=0.5))
loss = criterion(student_logits, teacher_logits, labels=labels)
```

### MoE Load Balancing

```python
from llm_grow.training.load_balance import combined_moe_loss

loss = combined_moe_loss(
    lm_loss, router_logits_list,
    num_experts=8, top_k=2, balance_coeff=1e-2, z_coeff=1e-3,
)
```

### Progressive Masked Growth

```python
from llm_grow.training.growth_scheduler import GrowthScheduleConfig, GrowthScheduler

scheduler = GrowthScheduler(GrowthScheduleConfig(
    total_steps=100_000, warmup_ratio=0.3, strategy="linear",
))
scheduler.apply_masks(model, scheduler.get_unlock_ratio(step))
```

### Verification

```python
from llm_grow.eval import verify_fp, StructuralVerifier

verify_fp("path/to/original", "path/to/expanded", atol=1e-4)

verifier = StructuralVerifier(src_dir="/path/to/original", dst_dir="/path/to/expanded")
results = verifier.run_all()  # {'config': True, 'tensor_counts': True, ...}
```

---

## Benchmarks

### Function-Preserving Verification (Real Models)

| Model | Method | `max|Δlogit|` | Result |
|-------|--------|:---:|:---:|
| Qwen2.5-0.5B | depth +4 (24→28) | 0.0000e+00 | ✓ PASS |
| Qwen2.5-0.5B | width +512 (inter=5376) | 0.0000e+00 | ✓ PASS |
| Qwen3-0.6B | depth +4 (28→32) | 0.0000e+00 | ✓ PASS |
| Qwen3-0.6B | width +512 (inter=3584) | 0.0000e+00 | ✓ PASS |

### In-Memory Expansion (Qwen3-0.6B, 596M, 28 layers, CPU)

| Method | Layers | Params | Ratio | FP | Time |
|--------|:---:|:---:|:---:|:---:|:---:|
| ZeroBlockInsert (+7) | 28→35 | 596M→706M | 1.19× | max\|Δ\|=0.000 | 0.2s |
| OverlapCopy (overlap=8) | 28→40 | 596M→785M | 1.32× | Non-FP | 0.2s |
| SVDInterpInsert (+4) | 28→32 | 596M→659M | 1.11× | ≈FP | 0.05s |
| MultiAxisPad (+4) | 28→32 | 596M→659M | 1.11× | max\|Δ\|=0.000 | 0.05s |
| DenseToMoE (×4) | 28→MoE | 596M→1.39B | 2.33× | Non-FP | 6.1s |
| ExpertClone (4→8) | — | 1.39B→2.45B | 1.76× | post noise | 1.7s |

### Safetensor Expansion (Real Models, All Configs Verified)

| Model | Method | Source | Output | Config |
|-------|--------|:---:|:---:|--------|
| Qwen2.5-0.5B | depth +4 | 290 | 338 | layers=28 ✓ |
| Qwen2.5-0.5B | width +512 | 290 | 338 | inter=5376 ✓ |
| Qwen3-0.6B | depth +4 | 311 | 355 | layers=32 ✓ |
| Qwen3-0.6B | width +512 | 311 | 355 | inter=3584 ✓ |
| Qwen3-30B-A3B | depth +4 | 18,867 | 20,439 | layers=52 ✓ |
| Qwen3-30B-A3B | expert ×2 | 18,867 | 37,299 | experts=256 ✓ |
| LongCat-Flash-Lite | depth +2 | 11,160 | 12,748 | layers=16 ✓ |
| LongCat-Flash-Lite | expert ×2 | 11,160 | 21,912 | experts=512 ✓ |
| Kimi-K2-Thinking | depth +2 | 139,644 | 148,952 | layers=63 ✓ |
| Kimi-K2-Thinking | expert ×2 | 139,644 | 277,884 | experts=768 ✓ |

---

## Project Structure

```
llm-grow/
├── src/llm_grow/
│   ├── cli.py                    # CLI entry point
│   ├── configs/                  # Shared config dataclasses + constants
│   ├── core/                     # ModelInspector abstract + markers
│   ├── safetensor/               # Safetensor-level expansion (mmap streaming)
│   │   ├── auto.py               #   auto_expand() + @register_expander registry
│   │   ├── detect.py             #   ModelProfile architecture auto-detection
│   │   ├── base.py               #   SafetensorExpanderBase + dry_run
│   │   ├── recipe.py             #   TensorRecipe + ExpansionPlan
│   │   ├── shard_writer.py       #   Two-pass streaming ShardWriter
│   │   ├── writer.py             #   apply_recipe + worker_write_shard
│   │   ├── utils.py              #   ShardIndex + header-only utilities
│   │   ├── methods/              #   Expand by algorithm
│   │   └── models/               #   Expand by model architecture
│   ├── expanders/                # In-memory expansion
│   │   ├── base.py               #   AbstractExpander (ABC)
│   │   ├── registry.py           #   @register_expander decorator
│   │   ├── depth/                #   ZeroBlockInsert / OverlapCopy / SVDInterpInsert
│   │   ├── width/                #   MultiAxisPad
│   │   └── sparse/               #   DenseToMoE / ExpertClone
│   ├── initializers/             # Weight init + noise strategies
│   │   └── noise.py              #   GaussianNoise / UniformNoise / ScaledGaussian
│   ├── training/                 # Freeze / Distillation / Scheduling / Load balance
│   ├── eval/                     # FP verification / Structural checks / Recovery curves
│   └── utils/                    # Logging, model I/O, expansion rules, insertion
├── examples/                     # Example scripts (safetensor + in-memory)
├── tests/                        # 246 tests, 83% coverage
└── docs/                         # Tutorials + architecture diagrams
```

---

## Tutorials

| Model | Architecture | Parameters | Guide |
|-------|:---:|:---:|-------|
| Qwen3-0.6B | Dense | 596M | [docs/expand_qwen3_0.6b.md](docs/expand_qwen3_0.6b.md) |
| Qwen3-30B-A3B | MoE Standard | ~30B | [docs/expand_qwen3_30b_a3b.md](docs/expand_qwen3_30b_a3b.md) |
| Kimi-K2-Base | DeepSeek MoE | ~1T | [docs/expand_kimi_k2.md](docs/expand_kimi_k2.md) |
| LongCat-Flash-Chat | LongCat MoE | ~0.5T | [docs/expand_longcat_flash.md](docs/expand_longcat_flash.md) |

---

## References

1. **LLaMA-Pro** — Wu et al., [arXiv:2401.02415](https://arxiv.org/abs/2401.02415) (2024)
2. **SOLAR DUS** — Kim et al., [arXiv:2312.15166](https://arxiv.org/abs/2312.15166) (2023)
3. **LESA** — Yang et al., [arXiv:2502.13794](https://arxiv.org/abs/2502.13794) (2025)
4. **MSG** — Du et al., [arXiv:2305.02869](https://arxiv.org/abs/2305.02869) (2023)
5. **Net2Net** — Chen et al., [arXiv:1511.05641](https://arxiv.org/abs/1511.05641) (ICLR 2016)
6. **Sparse Upcycling** — Komatsuzaki et al., [arXiv:2212.05055](https://arxiv.org/abs/2212.05055) (ICLR 2023)
7. **ExpertClone** — Amazon AI, [arXiv:2604.19835](https://arxiv.org/abs/2604.19835) (2026)
8. **DeepSeek-V2 MLA** — DeepSeek AI, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) (2024)
