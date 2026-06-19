# Qwen3-0.6B 扩增教程

**架构**：Dense（纯 Transformer Decoder，无 MoE）
**参数量**：596M（28 层，hidden=1024，intermediate=3072，tied embedding）
**模型路径**：`Qwen/Qwen3-0.6B`（HuggingFace）或本地目录

---

## 架构分析

```
层数：28
hidden_size：1024
intermediate_size：3072
num_attention_heads：16  num_key_value_heads：8（GQA）
head_dim：128
vocab_size：151936  tie_word_embeddings：True
参数分布：embedding 155M（26%）+ 28 层 × 15.7M（74%）
```

**参数占比说明**：tied embedding 占全模型 26%，因此单纯层数翻倍只能达到 ~1.74x；
要精确 2x 需要插入 38 个恒等块（28→66 层）。

---

## 可用扩增方案

| 方案 | 方法 | 目标参数量 | 倍率 | FP | 参考配置 |
|------|------|:---:|:---:|:---:|------|
| **IdentityGraft 2x** | 深度 +38 块 | 1194M | 2.00x | ✓ | `configs/Qwen3-0.6B/identity_graft.yaml` |
| IdentityGraft 层翻倍 | 深度 +28 块 | 1037M | 1.74x | ✓ | — |
| MultiAxisGrow 深度+宽度 | 深度+14, FFN+2048 | ~1081M | ~1.81x | ✓ | `configs/Qwen3-0.6B/multi_axis_grow.yaml` |
| OverlapSplit | 层重叠复制 | 785M | 1.32x | ✗ | `configs/Qwen3-0.6B/overlap_split.yaml` |

---

## 前置条件

```bash
# 1. 克隆 llm-grow
git clone https://github.com/jianzhnie/llm-grow
cd llm-grow && pip install -e .

# 2. 下载模型权重
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
```

---

## 方案 A：IdentityGraft 深度扩增（推荐，精确 2x）

### Step 1  Dry-run 验证方案（无需权重）

```bash
python scripts/safetensor_expand.py auto \
    --src ./models/Qwen3-0.6B \
    --dst /tmp/qwen3_0.6b_2x \
    --method depth \
    --num-new-layers 38 \
    --dry-run
```

预期输出：
```
[SafetensorExpand] [dry_run] LlamaProSafetensorExpander ...
  source:  311 tensors, 1 shard(s)
  output:  541 tensors, num_hidden_layers → 66
  zero-out tensors : 38      ← 38 个恒等块的 o_proj + down_proj
  brand-new keys   : 418
```

### Step 2  执行扩增

```bash
python scripts/safetensor_expand.py auto \
    --src ./models/Qwen3-0.6B \
    --dst ./outputs/Qwen3-0.6B-2x \
    --method depth \
    --num-new-layers 38
```

预期输出目录：
```
outputs/Qwen3-0.6B-2x/
├── config.json               # num_hidden_layers: 66
├── model-00001-of-XXXXX.safetensors
├── model-00002-of-XXXXX.safetensors
├── model.safetensors.index.json
├── tokenizer.json / tokenizer_config.json / ...
```

### Step 3  验证

```bash
python scripts/verify_safetensor.py \
    --src ./models/Qwen3-0.6B \
    --dst ./outputs/Qwen3-0.6B-2x \
    --fp
```

预期：
```
[Config diff]  num_hidden_layers: 28 → 66
[Tensor counts]  [✓] expected 541, got 541
[Identity blocks zeroed]  [✓] Found 38 zeroed projection(s)
[FP logit check]  [✓] max|Δlogit| = 0.000e+00
```

---

## 方案 B：MultiAxisGrow 深度+宽度（推理延迟更小）

```bash
python scripts/safetensor_expand.py msg \
    --src ./models/Qwen3-0.6B \
    --dst ./outputs/Qwen3-0.6B-msg \
    --num-new-layers 14 \
    --ffn-size-expansion 2048
# 结果：28→42 层，intermediate 3072→5120，~1081M（~1.81x）
# 推理延迟约 1.4x（优于纯深度翻倍的 2x）
```

---

## Continued Pre-training 建议

扩增完成后需要 CPT 恢复/超越原始精度。

**两阶段训练（推荐）**：

```python
from llm_grow.training.freeze import freeze_original_layers, unfreeze_all, report_trainable

# Phase 1：冻结原始 28 层，仅训练 38 个新增块
model = AutoModelForCausalLM.from_pretrained("./outputs/Qwen3-0.6B-2x")
freeze_original_layers(model)
report_trainable(model)
# 训练约 12B tokens，lr=2e-4，cosine scheduler

# Phase 2：解冻全部
unfreeze_all(model)
# 训练约 6B tokens，lr=5e-5
```

**数据建议**：
- 90% 通用预训练语料（与原始分布一致）
- 5% 高质量指令数据（防遗忘）
- 5% 原始预训练数据 replay（防灾难性遗忘）

**评估**：MMLU / C-Eval / GSM8K / HumanEval，每 2B tokens 评估一次。

---

## 参数量精确计算

```python
from llm_grow.utils.arch_info import count_params, param_diff_report
import copy
from transformers import AutoModelForCausalLM

orig = AutoModelForCausalLM.from_pretrained("./models/Qwen3-0.6B")
expanded = AutoModelForCausalLM.from_pretrained("./outputs/Qwen3-0.6B-2x")
param_diff_report(orig, expanded)
# Expansion ratio: 2.000x
```
