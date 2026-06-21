# llm-grow

<p align="center">
  <img src="docs/images/logo.svg" width="280"/>
</p>

<p align="center">
  <em>Expand Existing Models, Layer by Layer</em>
</p>

<p align="center">
  <a href="#installation">Installation</a> &bull;
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#expansion-methods">Methods</a> &bull;
  <a href="#api-reference">API Reference</a> &bull;
  <a href="#tutorials">Tutorials</a>
</p>

---

A modular toolkit for **growing** larger models from existing LLM checkpoints.

**Key Features**:

- **Two-tier expansion** — In-memory (`nn.Module` in-place) and Safetensor-level (mmap streaming, peak memory <= 4 GB)
- **Four architecture families** — Dense / MoE-Standard / DeepSeek-MoE / LongCat, auto-detected
- **Six expansion algorithms** — ZeroBlockInsert, OverlapCopy, SVDInterpInsert, MultiAxisPad, DenseToMoE, ExpertClone
- **Function-Preserving** — ZeroBlockInsert / MultiAxisPad produce zero accuracy loss at expansion time
- **Complete training toolkit** — Frozen training, knowledge distillation, progressive mask growth, MoE load balancing

---

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Expansion Methods](#expansion-methods)
- [Method Selection Guide](#method-selection-guide)
- [API Reference](#api-reference)
- [Training & Evaluation](#training--evaluation)
- [Tutorials](#tutorials)
- [Benchmarks](#benchmarks)
- [Project Structure](#project-structure)
- [References](#references)

---

## Installation

```bash
pip install -e .                # Core (expansion + CLI)
pip install -e ".[train]"       # + Training deps (DeepSpeed, Flash-Attn, Datasets)
pip install -e ".[eval]"        # + Evaluation deps (lm-eval-harness)
pip install -e ".[dev]"         # + Dev tools (pytest, ruff, mypy)
```

**Requirements**: Python >= 3.10, PyTorch >= 2.2, Transformers >= 4.40, safetensors >= 0.4

---

## Quickstart

### CLI (Recommended)

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
    --method depth --dry-run

# Verify expansion (structural + FP consistency)
llm-grow verify --src /path/to/original --dst /path/to/expanded --fp

# Display model architecture info
llm-grow info --src /path/to/model
```

### Python API

```python
# Safetensor expansion (large models, no weight loading)
from llm_grow.safetensor.auto import auto_expand

auto_expand(
    src_dir="/path/to/model",
    dst_dir="./expanded",
    method="depth",             # "depth" | "expert" | "width"
    num_new_layers=4,
)

# In-memory expansion (small models / rapid prototyping)
import copy
from transformers import AutoModelForCausalLM
from llm_grow.expanders.depth.zero_block_insert import (
    ZeroBlockInsertConfig, ZeroBlockInsertExpander,
)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="auto")
original = copy.deepcopy(model)
expanded = ZeroBlockInsertExpander().expand(
    model,
    ZeroBlockInsertConfig(num_new_layers=9, freeze_original=True),
)
ZeroBlockInsertExpander().verify(original, expanded)
```

---

## Expansion Methods

### Comparison

| Method | FP | Direction | Instant Accuracy | Recommended CPT | Inference Latency |
|--------|:---:|:---:|:---:|:---:|:---:|
| **ZeroBlockInsert** | yes | Depth | **100%** | 8-16B tokens | +linear |
| **OverlapCopy** | no | Depth | 50-80% | 100B+ tokens | +linear |
| **SVDInterpInsert** | ~yes | Depth | 80-90% | <50B tokens | +linear |
| **MultiAxisPad** | yes | Depth+Width | **100%** | 30-60B tokens | ~1.4x |
| **DenseToMoE** | no | Dense->MoE | 70-85% | 50-100B tokens | ~unchanged |
| **ExpertClone** | ~yes | MoE Experts | -- | saves 32-67% | ~unchanged |

> **FP** = Function-Preserving: output is identical to the original model after expansion (zero accuracy loss at zero-shot).

### How They Work

#### ZeroBlockInsert — Identity Block Grafting

Inserts identity blocks at uniform intervals (`o_proj` / `down_proj` zeroed). Residual connections guarantee `output = x + 0 = x`.

<p align="center"><img src="docs/images/zero_block_insert.svg" width="720"/></p>

#### OverlapCopy — Layer Overlap Splicing

Splits the model into upper/lower segments with an overlap zone for smoothness, then concatenates to double the layer count. Non-FP, requires extensive CPT.

<p align="center"><img src="docs/images/overlap_copy.svg" width="720"/></p>

#### SVDInterpInsert — SVD Interpolation Grafting

Interpolates adjacent layer weights via arithmetic mean. New layers start from meaningful initialization, converging faster than DUS.

<p align="center"><img src="docs/images/svd_interp_insert.svg" width="720"/></p>

#### MultiAxisPad — Multi-Axis Masked Growth

Simultaneously expands depth (identity blocks) and width (zero-pad hidden/FFN dimensions). All new parameters are zero-initialized — strictly FP.

<p align="center"><img src="docs/images/multi_axis_pad.svg" width="720"/></p>

#### DenseToMoE — Dense to Sparse MoE

Replicates the dense FFN into N experts + randomly initialized router. Top-K routing keeps inference cost nearly unchanged.

<p align="center"><img src="docs/images/dense_to_moe.svg" width="720"/></p>

#### ExpertClone — MoE Expert Cloning

Duplicates existing experts with symmetry breaking (noise/drop). Router weights are expanded accordingly. Inference cost unchanged.

<p align="center"><img src="docs/images/expert_clone.svg" width="720"/></p>

---

## Method Selection Guide

```
Need to expand parameters
|
+-- Very large model (cannot load fully) --> Safetensor-level expansion
|   +-- Dense model
|   |   +-- Depth expansion        llm-grow expand --method depth
|   |   +-- FFN width expansion    llm-grow expand --method width
|   +-- MoE model
|       +-- Expert expansion (unchanged inference cost)  llm-grow expand --method expert
|       +-- Depth expansion (more layers)                llm-grow expand --method depth
|
+-- Small/medium model (fits in memory) --> In-memory expansion
    +-- Accuracy priority, limited data    --> ZeroBlockInsert (FP, 8-16B tokens)
    +-- Precise 2x, control latency        --> MultiAxisPad (depth+width, ~1.4x latency)
    +-- Simplest impl, ample data          --> OverlapCopy
    +-- Cannot increase inference latency  --> DenseToMoE (top-1 activation unchanged)
    +-- Base model is already MoE          --> ExpertClone
```

---

## API Reference

### Two-Tier Expansion Architecture

| | In-Memory (`expanders/`) | Safetensor-Level (`safetensor/`) |
|---|---|---|
| **Input** | `nn.Module` | `.safetensors` directory |
| **Output** | `nn.Module` | `.safetensors` directory |
| **Peak Memory** | Full model | <= 1 output shard (~4 GB) |
| **FP Verification** | Direct logit comparison | Structural check + optional logit comparison |
| **Use Case** | Small models / rapid experiments | 100B+ models |

### Safetensor Expansion

#### Auto Architecture Detection

`detect_model()` infers architecture from `config.json` + weight index without loading weights:

```python
from llm_grow.safetensor.detect import detect_model

profile = detect_model("/path/to/model")
print(profile.family)   # "dense" | "standard_moe" | "deepseek_moe" | "longcat"
```

| Detection Result | Representative Models | Key Features |
|---------|---------|---------|
| `dense` | Qwen3-0.6B/8B/14B/32B | No `mlp.experts.*` |
| `standard_moe` | Qwen3-30B-A3B | `mlp.experts.*` + `mlp.gate.weight` |
| `deepseek_moe` | Kimi-K2-Base | MLA + fp8 + shared expert + dense first layer |
| `longcat` | LongCat-Flash-Chat | Dual attention + dual MLP + 512 experts |

#### auto_expand()

```python
from llm_grow.safetensor.auto import auto_expand

auto_expand(
    src_dir="/path/to/model",
    dst_dir="./expanded",
    method="depth",               # "depth" | "expert" | "width"
    num_new_layers=4,             # [depth] layers to insert
    insert_strategy="uniform",    # [depth] "uniform" | "front" | "rear"
    expand_factor=2,              # [expert] expert multiplier
    ffn_size_expansion=0,         # [width] intermediate_size increment
    target_shard_gb=4.0,          # output shard size
    dry_run=False,                # True = plan only, no file writes
)
```

#### Pre-configured Factory Functions

| Function | Target Model | Description |
|------|---------|------|
| `make_qwen3moe_expert_clone(factor)` | Qwen3-30B-A3B | Expert count expansion |
| `make_qwen3moe_zero_block_insert(n)` | Qwen3-30B-A3B | Depth expansion |
| `make_kimik2_expert_clone(factor)` | Kimi-K2-Base | Expert expansion (fp8-aware) |
| `make_kimik2_zero_block_insert(n)` | Kimi-K2-Base | Depth expansion (dense first-layer aware) |

#### MoE Width Expansion (M3/M4)

```python
from llm_grow.safetensor.models.moe_width import MoEWidthConfig, MoEWidthExpander

# M3: Widen expert FFN
MoEWidthExpander(MoEWidthConfig(ffn_size_expansion=1024)).expand(
    src_dir="/path/to/moe_model", dst_dir="./moe_wider",
)

# M4: Widen hidden_size (attention / router / embedding / lm_head)
MoEWidthExpander(MoEWidthConfig(hidden_size_expansion=256)).expand(
    src_dir="/path/to/moe_model", dst_dir="./moe_wider_hidden",
)
```

### In-Memory Expansion

#### ZeroBlockInsert

```python
from llm_grow.expanders.depth.zero_block_insert import (
    ZeroBlockInsertConfig, ZeroBlockInsertExpander,
)

expanded = ZeroBlockInsertExpander().expand(model, ZeroBlockInsertConfig(
    num_new_layers=9,           # recommended = original_layers // 4
    insert_strategy="uniform",  # "uniform" | "front" | "rear"
    freeze_original=True,       # Phase-1: only train new blocks
))
```

#### MultiAxisPad

```python
from llm_grow.expanders.width.multi_axis_pad import (
    MultiAxisPadConfig, MultiAxisPadExpander,
)

expanded = MultiAxisPadExpander().expand(model, MultiAxisPadConfig(
    num_new_layers=10,
    hidden_size_expansion=512,
    intermediate_size_expansion=3072,
    freeze_original=True,
))
```

#### DenseToMoE

```python
from llm_grow.expanders.sparse.dense_to_moe import DenseToMoEConfig, DenseToMoEExpander

expanded = DenseToMoEExpander().expand(
    model, DenseToMoEConfig(num_experts=8, top_k=2),
)
```

#### ExpertClone

```python
from llm_grow.expanders.sparse.expert_clone import (
    ExpertCloneConfig, ExpertCloneExpander, ExpertSelectionStrategy,
)

expanded = ExpertCloneExpander().expand(
    moe_model,
    ExpertCloneConfig(
        expand_factor=2,
        selection_strategy=ExpertSelectionStrategy.UTILITY,
    ),
)
```

#### SVDInterpInsert

```python
from llm_grow.expanders.depth.svd_interp_insert import (
    SVDInterpInsertConfig, SVDInterpInsertExpander,
)

expanded = SVDInterpInsertExpander().expand(
    model, SVDInterpInsertConfig(use_predictor=False),
)
```

---

## Training & Evaluation

### Two-Phase Frozen Training

```python
from llm_grow.training.freeze import (
    snapshot_param_ids, mark_new_params,
    freeze_original_layers, unfreeze_all,
)

original_ids = snapshot_param_ids(model)
expander.expand(model, config)
mark_new_params(model, original_ids)

freeze_original_layers(model)          # Phase-1: train only new parameters
train(model, phase1_data, lr=2e-4)

unfreeze_all(model)                    # Phase-2: full fine-tuning
train(model, phase2_data, lr=1e-5)
```

### Knowledge Distillation

```python
from llm_grow.training.distillation import (
    DistillConfig, DistillationLoss, run_teacher_inference,
)

criterion = DistillationLoss(DistillConfig(temperature=2.0, alpha=0.5))
teacher_logits = run_teacher_inference(teacher, input_ids)
loss = criterion(student_logits, teacher_logits, labels=labels)
```

### MoE Load Balancing

```python
from llm_grow.training.load_balance import combined_moe_loss

loss = combined_moe_loss(
    lm_loss, router_logits_list,
    num_experts=8, top_k=2,
    balance_coeff=1e-2, z_coeff=1e-3,
)
```

### Progressive Masked Growth (MSG)

```python
from llm_grow.training.growth_scheduler import GrowthScheduleConfig, GrowthScheduler

scheduler = GrowthScheduler(GrowthScheduleConfig(
    total_steps=100_000, warmup_ratio=0.3, strategy="linear",
))

ratio = scheduler.get_unlock_ratio(step)
scheduler.apply_masks(model, ratio)
```

### Function-Preserving Verification

```python
from llm_grow.eval import verify_fp, StructuralVerifier

verify_fp("path/to/original", "path/to/expanded", atol=1e-4)

verifier = StructuralVerifier(src_dir="/path/to/original", dst_dir="/path/to/expanded")
results = verifier.run_all()
```

### Recovery Curve Tracking

```python
from llm_grow.eval import RecoveryCurveTracker

tracker = RecoveryCurveTracker("recovery.jsonl")
tracker.set_baseline({"mmlu": 0.72, "gsm8k": 0.65})
tracker.log(step=1000, tokens_seen=2e9, scores=run_eval(model))
tracker.summary()
```

---

## Tutorials

| Model | Architecture | Parameters | Tutorial |
|------|:---:|:---:|------|
| Qwen3-0.6B | Dense | 596M | [docs/expand_qwen3_0.6b.md](docs/expand_qwen3_0.6b.md) |
| Qwen3-30B-A3B | MoE Standard | ~30B | [docs/expand_qwen3_30b_a3b.md](docs/expand_qwen3_30b_a3b.md) |
| Kimi-K2-Base | DeepSeek MoE | ~1T | [docs/expand_kimi_k2.md](docs/expand_kimi_k2.md) |
| LongCat-Flash-Chat | LongCat MoE | ~0.5T | [docs/expand_longcat_flash.md](docs/expand_longcat_flash.md) |

---

## Benchmarks

### In-Memory Expansion (Qwen3-0.6B, 596M, 28 layers, CPU)

| Method | Layers | Parameters | Ratio | FP Check | Time |
|--------|:---:|:---:|:---:|:---:|:---:|
| ZeroBlockInsert (+7) | 28->35 | 596M->706M | 1.19x | max\|d\|=**0.000** | 0.2s |
| OverlapCopy (overlap=8) | 28->40 | 596M->785M | 1.32x | Non-FP (expected) | 0.2s |
| SVDInterpInsert (+4) | 28->32 | 596M->659M | 1.11x | Near-FP | 0.05s |
| MultiAxisPad (+4) | 28->32 | 596M->659M | 1.11x | max\|d\|=**0.000** | 0.05s |
| DenseToMoE (x4) | 28->MoE | 596M->1.39B | 2.33x | Non-FP (expected) | 6.1s |
| ExpertClone (4->8) | -- | 1.39B->2.45B | 1.76x | Pass (after symmetry break) | 1.7s |

### Safetensor Expansion (dry-run verification)

| Model | Method | Source Tensors | Output Tensors | Added |
|------|------|:---:|:---:|:---:|
| Qwen3-0.6B | depth +4 layers | 311 | 355 | 44 |
| Qwen3-0.6B | ZeroBlockInsert +7 | 311 | 388 | 77 |
| Qwen3-30B-A3B | expert 128->256 | 18,867 | 37,299 | 18,432 |
| Qwen3-30B-A3B | depth 48->56 | 18,867 | 22,011 | 3,144 |
| Kimi-K2-Base | expert 384->768 | 139,644 | 277,884 | 138,240 |
| Kimi-K2-Base | depth 61->65 | 139,644 | 148,952 | 9,308 |

---

## Project Structure

```
llm-grow/
+-- src/llm_grow/
|   +-- cli.py                  # CLI entry point
|   +-- safetensor/             # Safetensor-level expansion (mmap streaming)
|   |   +-- auto.py             #   auto_expand() unified entry
|   |   +-- detect.py           #   ModelProfile architecture detection
|   |   +-- base.py             #   ExpansionPlan + two-pass shard writer
|   |   +-- methods/            #   Organized by expansion method
|   |   +-- models/             #   Organized by model architecture
|   +-- expanders/              # In-memory expansion
|   |   +-- depth/              #   ZeroBlockInsert / OverlapCopy / SVDInterpInsert
|   |   +-- width/              #   MultiAxisPad
|   |   +-- sparse/             #   DenseToMoE / ExpertClone
|   +-- initializers/           # Weight initialization
|   +-- training/               # Freeze / Distillation / Scheduling / Load balancing
|   +-- eval/                   # FP verification / Structural checks / Recovery curves
+-- examples/                   # Example scripts
+-- tests/                      # Unit tests
+-- docs/                       # Tutorials + architecture diagrams
```

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
