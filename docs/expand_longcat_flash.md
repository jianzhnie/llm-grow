# LongCat-Flash Expansion Guide

**Architecture**: LongCat custom MoE (`LongcatFlashNgramForCausalLM`)
**Parameters**: ~0.5T (28 layers, 512 routed experts)
**Model**: `meituan-longcat/LongCat-Flash-Chat` or local path

---

## Architecture

```
Layers:               28
Hidden size:          6144
Routed experts:       512 per layer
Zero experts:         256 (identity-initialised, gradually learn)
Active experts:       12 (moe_topk)
Attention:            Dual MLA (self_attn.0 / self_attn.1)
Dense MLP:            Dual (mlps.0 / mlps.1), parallel to MoE
Router:               mlp.router.classifier.weight + e_score_correction_bias
Shards:               75 (model_00001-of-00075.safetensors …)
```

**Zero-expert design** (LongCat-unique):

```
Expert layout (pre-expansion):
  [expert_0 … expert_255]    → Real experts (trained weights)
  [expert_256 … expert_511]  → Zero experts (identity init, all-zero weights)

Router weight shape: [768, 6144]
  rows [0:512]   → Real expert scores
  rows [512:768] → Zero expert scores
```

Zero experts implement ZeroBlockInsert at the MoE level — they start as identity
functions and specialise during training, giving free capacity with no inference
cost increase.

---

## Expansion Options

| Method | Config | Target | Top-K |
|--------|--------|:---:|:---:|
| **Expert 2×** ★ | 512→1024 (256+256 zero → 512+512 zero) | ~1T | 12→24 (or 12) |
| Depth | 28→32 layers (+4 identity blocks) | ~0.57T | unchanged |

> `moe_topk` does **not** auto-scale by default (`scale_moe_topk=False`).
> Set `scale_moe_topk=True` to double it from 12→24.

---

## Quickstart

```bash
# Dry-run
llm-grow expand \
    --src /home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite \
    --dst /tmp/longcat_2x \
    --method expert --expand-factor 2 --dry-run

# Execute
llm-grow expand \
    --src /home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite \
    --dst ./outputs/LongCat-Flash-2x \
    --method expert --expand-factor 2 \
    --workers 4 --validate-output

# Verify
llm-grow verify \
    --src /home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite \
    --dst ./outputs/LongCat-Flash-2x
```

Expected:

```
  source:  11,160 tensors, 26 shard(s)
  output:  21,912 tensors, num_hidden_layers → 14
  dup-rows: 28 (router classifier + bias per layer)
  new keys: 10,752 (256 experts × 14 layers × 3 tensors)
  config:  {n_routed_experts: 512, zero_expert_num: 256}
```

---

## Zero-Expert Handling

`LongcatExpertCloneExpander` uses `router_split=256` to distinguish zero-expert rows:

```
router.classifier.weight expansion:
  rows [0:256]    → Real experts → clone WITH noise (noise_scale=1e-6)
  rows [256:512]  → Zero experts → clone WITHOUT noise (preserve identity)
  result: [768, 6144] → [1536, 6144]
```

---

## Depth Expansion

Insert identity blocks into LongCat (each identity block zeros **516 tensors**):

```bash
llm-grow expand \
    --src /home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite \
    --dst ./outputs/LongCat-Flash-deeper \
    --method depth --num-new-layers 4
# 14→18 layers, ~14% param increase
```

Zeroed per identity block:
- `self_attn.0.o_proj.weight` + `self_attn.1.o_proj.weight` (2)
- `mlps.0.down_proj.weight` + `mlps.1.down_proj.weight` (2)
- `mlp.experts.{0..511}.down_proj.weight` (512)

---

## Python API

```python
from llm_grow.safetensor.models.longcat import (
    LongcatExpertCloneConfig, LongcatExpertCloneExpander,
)

cfg = LongcatExpertCloneConfig(expand_factor=2, noise_scale=1e-6)
LongcatExpertCloneExpander(cfg).expand(
    src_dir="/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite",
    dst_dir="./outputs/LongCat-Flash-2x",
    workers=4,
)
```

---

## Verify Output

```bash
# Confirm auto_map Python files copied
ls ./outputs/LongCat-Flash-2x/*.py
# Expected: configuration_longcat_ngram.py  modeling_longcat_ngram.py

# Confirm config
python -c "
import json
cfg = json.load(open('./outputs/LongCat-Flash-2x/config.json'))
print('n_routed_experts:', cfg['n_routed_experts'])  # 512
print('zero_expert_num:', cfg['zero_expert_num'])      # 256
"
```

---

## Continued Pre-training

```
Focus:          Real/zero expert specialisation
Tokens:         40–80B
Learning rate:  2e-5, cosine, warmup 2,000 steps
Batch size:     per_device=1, grad_accum=16
Seq length:     8192 (LongCat supports long context)
MoE loss:       balance_coeff=1e-2, z_loss_coeff=1e-3

Monitor (every 2B tokens):
  - Expert activation frequency (uniform = good load balance)
  - Zero-expert activation rate (should gradually increase)
  - Router entropy (higher = more balanced)
  - MMLU / LongBench / Needle-in-Haystack
```
