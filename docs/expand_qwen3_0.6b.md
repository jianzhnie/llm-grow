# Qwen3-0.6B Expansion Guide

**Architecture**: Dense Transformer Decoder (no MoE)
**Parameters**: 596M (28 layers, hidden=1024, intermediate=3072, tied embedding)
**Model**: `Qwen/Qwen3-0.6B` (HuggingFace) or local path

---

## Architecture

```
Layers:              28
Hidden size:         1024
Intermediate size:   3072
Attention heads:     16 / KV heads: 8 (GQA)
Head dim:            128
Vocab size:          151,936 (tied embedding)
Params breakdown:    Embedding 155M (26%) + 28 × 15.7M (74%)
```

The tied embedding accounts for 26% of total parameters. For a precise 2× scale,
38 identity blocks are needed (28→66 layers) rather than simply doubling layers.

---

## Expansion Options

| Method | Config | Target | Ratio | FP |
|--------|--------|:---:|:---:|:---:|
| **ZeroBlockInsert 2×** ★ | +38 layers | 1194M | 2.00× | ✓ |
| ZeroBlockInsert deep | +28 layers | 1037M | 1.74× | ✓ |
| MultiAxisPad | +14 layers, FFN+2048 | ~1081M | ~1.81× | ✓ |
| OverlapCopy | overlap=8 | 785M | 1.32× | ✗ |

---

## Quickstart

```bash
# Install
pip install -e .

# Dry-run (plan only)
llm-grow expand \
    --src /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B \
    --dst /tmp/qwen3_0.6b_2x \
    --method depth --num-new-layers 38 --dry-run

# Execute expansion
llm-grow expand \
    --src /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B \
    --dst ./outputs/Qwen3-0.6B-2x \
    --method depth --num-new-layers 38 --validate-output

# Verify
llm-grow verify \
    --src /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B \
    --dst ./outputs/Qwen3-0.6B-2x --fp
```

Expected output:

```
[FP Verify] PASSED  max|Δlogit|=0.0000e+00
  [pass] config          (num_hidden_layers: 28 → 66)
  [pass] tensor_counts   (311 → ~541)
  [pass] weights_preserved
  [pass] fp_logit_check
```

Output directory:
```
outputs/Qwen3-0.6B-2x/
├── config.json              # num_hidden_layers: 66
├── model.safetensors        # (or multi-shard)
├── tokenizer.json / tokenizer_config.json / ...
```

---

## MultiAxisPad (Depth + Width)

Lower inference latency than pure depth doubling:

```bash
llm-grow expand \
    --src /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B \
    --dst ./outputs/Qwen3-0.6B-msg \
    --method width --num-new-layers 14 --ffn-size-expansion 2048
# Result: 28→42 layers, intermediate 3072→5120, ~1081M (~1.81×)
# Latency: ~1.4× (vs 2× for pure depth doubling)
```

---

## Python API

```python
from llm_grow.safetensor.auto import auto_expand

auto_expand(
    src_dir="/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B",
    dst_dir="./outputs/Qwen3-0.6B-2x",
    method="depth",
    num_new_layers=38,
    workers=4,
    validate_output=True,
)
```

Or via the in-memory API with the registry:

```python
from llm_grow.expanders.registry import get_expander
from llm_grow.expanders.depth.zero_block_insert import ZeroBlockInsertConfig

expander = get_expander("zero_block_insert")()
expanded = expander.expand(model, ZeroBlockInsertConfig(num_new_layers=38))
expander.verify(original, expanded)  # max|Δlogit| = 0.0000e+00
```

---

## Continued Pre-training

**Two-phase training (recommended)**:

```python
from llm_grow.training.freeze import (
    snapshot_param_ids, mark_new_params, freeze_original_layers, unfree_all,
)

# Phase 1: freeze original 28 layers, train only new blocks
original_ids = snapshot_param_ids(model)
expander.expand(model, config)
mark_new_params(model, original_ids)
freeze_original_layers(model)
# Train ~12B tokens, lr=2e-4, cosine scheduler

# Phase 2: unfreeze all
unfreeze_all(model)
# Train ~6B tokens, lr=5e-5
```

**Data mix**: 90% general pretraining + 5% instruction + 5% replay (anti-forgetting).

**Evaluation**: MMLU / C-Eval / GSM8K / HumanEval every 2B tokens.

---

## Compute Requirements

| Phase | Tokens | Hardware | Time (est.) |
|-------|:---:|------|------|
| Phase 1 (frozen) | 12B | 4× A100-80G | ~30 hours |
| Phase 2 (full) | 6B | 4× A100-80G | ~40 hours |

---

## Parameter Count Verification

```python
from llm_grow.utils.arch_info import count_params, param_diff_report
from transformers import AutoModelForCausalLM

orig = AutoModelForCausalLM.from_pretrained("./models/Qwen3-0.6B")
expanded = AutoModelForCausalLM.from_pretrained("./outputs/Qwen3-0.6B-2x")
param_diff_report(orig, expanded)
# Expected: ratio = 2.000×
```
