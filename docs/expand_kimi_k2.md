# Kimi-K2 Expansion Guide

**Architecture**: DeepSeek-V3 variant (`DeepseekV3ForCausalLM`)
**Parameters**: ~1T (fp8 quantized, ~500 GB on disk)
**Model**: `moonshotai/Kimi-K2-Base` or local path

---

## Architecture

```
Layers:               61 (layer 0 = Dense, layers 1–60 = MoE)
Hidden size:          7168
Routed experts:       384 per MoE layer
Shared experts:       1 per MoE layer (always activated)
Active experts:       8 (num_experts_per_tok)
MLA attention:        kv_lora_rank=512, q_lora_rank=1536
FP8 quantization:     weight_scale_inv per tensor
Router:               mlp.gate.weight + mlp.gate.e_score_correction_bias
Shards:               61 (model-1-of-61.safetensors … model-61-of-61.safetensors)
```

**Key architecture notes**:

1. **Layer 0 is Dense** (no experts) — layers 1–60 are MoE
2. **FP8**: every `*.weight` has a `*.weight_scale_inv` — both are cloned during expansion
3. **Shared expert** `mlp.shared_experts.*`: always activated, preserved unchanged
4. **MLA attention**: `q_a_proj`/`kv_a_proj` low-rank decompositions — no special handling needed
5. **auto_map**: `config.json` references `configuration_deepseek.py` / `modeling_deepseek.py` — auto-copied

---

## Expansion Options

| Method | Config | Target | Ratio | Inference |
|--------|--------|:---:|:---:|:---:|
| **Expert 2×** ★ | 384→768 experts | ~2T | ~2.0× | top-8→16 (or unchanged) |
| Expert 2× (same cost) | 384→768, topk unchanged | ~2T | ~2.0× | **unchanged** |
| Depth | 61→65 layers | ~1.07T | ~1.07× | ↑ ~7% |

---

## Quickstart

```bash
# Dry-run (only index.json needed — no weight files)
llm-grow expand \
    --src /home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K2-Thinking \
    --dst /tmp/kimi_k2_2x \
    --method expert --expand-factor 2 --dry-run

# Execute expansion
llm-grow expand \
    --src /home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K2-Thinking \
    --dst ./outputs/Kimi-K2-2x \
    --method expert --expand-factor 2 \
    --workers 8 --validate-output

# Memory-constrained
llm-grow expand \
    --src /home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K2-Thinking \
    --dst ./outputs/Kimi-K2-2x \
    --method expert --expand-factor 2 \
    --workers 2 --target-shard-gb 2

# Verify (structural only — 1T model)
llm-grow verify \
    --src /home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K2-Thinking \
    --dst ./outputs/Kimi-K2-2x
```

Expected:

```
  source:  139,644 tensors, 61 shard(s)
  output:  277,884 tensors, num_hidden_layers → 61
  dup-rows: 120 (60 MoE layers × 2 router tensors)
  new keys: 138,240 (384 experts × 60 layers × 6 tensors incl. weight_scale_inv)
  config:  {n_routed_experts: 768, num_experts_per_tok: 16}
```

---

## Memory Requirements

Peak memory ≈ largest input shard × 2 per worker. Kimi-K2 shards are ~8 GB each:

```
  Workers=1:  ~16 GB peak
  Workers=4:  ~64 GB peak
  Workers=8:  ~128 GB peak
```

Reduce `--workers` or `--target-shard-gb` if memory is constrained.

---

## FP8 Expansion Details

Kimi-K2 uses fp8 (E4M3). Each weight has a scale tensor:

```
mlp.experts.0.gate_proj.weight           # fp8 [2048, 7168]
mlp.experts.0.gate_proj.weight_scale_inv # scale [16, 56]
```

During expert cloning, both `weight` and `weight_scale_inv` are copied together.
Cloned experts use identical quantization parameters — inference accuracy is preserved.

**Training with fp8**:
- Use fp8-aware frameworks (TransformerEngine / Nanotron / Megatron-LM)
- `weight_scale_inv` is recomputed during training — copied values are fine as initialisation
- Recommend `activation_scheme: dynamic` (consistent with source model)

---

## Python API

```python
from llm_grow.safetensor.models.moe_generic import make_kimik2_expert_clone

make_kimik2_expert_clone(expand_factor=2, noise_scale=1e-6).expand(
    src_dir="/home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K2-Thinking",
    dst_dir="./outputs/Kimi-K2-2x",
    workers=8,
)
```

---

## Verify Output

```bash
# Confirm auto_map Python files were copied
ls ./outputs/Kimi-K2-2x/*.py
# Expected: configuration_deepseek.py  modeling_deepseek.py  tokenization_kimi.py

# Confirm config patches
python -c "
import json
cfg = json.load(open('./outputs/Kimi-K2-2x/config.json'))
print('n_routed_experts:', cfg['n_routed_experts'])       # 768
print('num_experts_per_tok:', cfg['num_experts_per_tok'])  # 16
"
```

---

## Continued Pre-training

```
Scale:          Cluster-grade (1T+ model)
Tokens:         50–100B
Learning rate:  1e-5 (conservative), cosine, warmup 3,000 steps
Batch size:     per_device=1, grad_accum=32
FP8 training:   TransformerEngine or Nanotron
MoE loss:       aux_loss_alpha=0.001, seq_aux=True
Seq length:     4096 initial, ramp to 8192+
Monitor:        Expert activation distribution, fp8 overflow rate, router entropy
```
