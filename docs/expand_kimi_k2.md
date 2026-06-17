# Kimi-K2-Base 扩增教程

**架构**：DeepSeek-V3 变体（`DeepseekV3ForCausalLM`）
**参数量**：~1T（fp8 量化存储，实际约 500GB 磁盘）
**模型路径**：`moonshotai/Kimi-K2-Base`（HuggingFace）

---

## 架构分析

```
层数：61（layer 0 = Dense，layer 1–60 = MoE）
hidden_size：7168
n_routed_experts：384（每 MoE 层路由专家数）
n_shared_experts：1（每 MoE 层共享专家，所有 token 都激活）
num_experts_per_tok：8（top-k 路由专家数）
MLA 注意力：kv_lora_rank=512，q_lora_rank=1536
fp8 量化：每个权重张量有对应的 weight_scale_inv 张量
Router：mlp.gate.weight + mlp.gate.e_score_correction_bias
分片：61 个 shard（model-1-of-61.safetensors ... model-61-of-61.safetensors）
```

**特殊架构注意事项**：

1. **Layer 0 是 Dense 层**（无专家），layers 1–60 是 MoE 层
2. **fp8 量化**：每个 `*.weight` 张量有对应的 `*.weight_scale_inv`，扩增时同步复制
3. **共享专家** `mlp.shared_experts.*`：所有 token 常激活，不参与路由，扩增时保持不变
4. **MLA 注意力**：`q_a_proj`/`kv_a_proj` 是 low-rank 分解，不需要特殊处理
5. **`auto_map`**：`config.json` 引用 `configuration_deepseek.py` 和 `modeling_deepseek.py`，
   扩增时自动复制到输出目录

---

## 可用扩增方案

| 方案 | 方法 | 目标参数量 | 倍率 | 推理成本 | 参考配置 |
|------|------|:---:|:---:|:---:|------|
| **专家扩增 2x** ★ | 384→768 experts | ~2T | ~2.0x | 激活↑（top-8→16） | `configs/Kimi-K2-Base/expert_upcycling.yaml` |
| 专家扩增（推理不变） | 384→768，topk 不变 | ~2T | ~2.0x | **不变** | 修改 YAML |
| 深度扩增 | 61→65 层 | ~1.07T | ~1.07x | ↑ ~7% | `configs/Kimi-K2-Base/depth.yaml` |

> **深度扩增增量有限**（每层包含 384 专家，层参数占比很小），主要用于精度微调场景。

---

## 前置条件

```bash
# 下载完整模型（~500GB，建议高速存储）
huggingface-cli download moonshotai/Kimi-K2-Base \
    --local-dir ./models/Kimi-K2-Base

# 验证下载完整性
python -c "
from llm_grow.safetensor.utils import ShardIndex
idx = ShardIndex.load('./models/Kimi-K2-Base')
import os
missing = [s for s in idx.shard_files
           if not (idx.model_dir / s).exists()]
print(f'Missing shards: {missing or \"None\"}')
print(f'Total: {len(idx.shard_files)} shards, {idx.total_size_bytes()/1e9:.1f} GB')
"
```

---

## Step 1  Dry-run（无权重验证方案）

```bash
python scripts/safetensor_expand.py auto \
    --src ./models/Kimi-K2-Base \
    --dst /tmp/kimi_k2_2x \
    --method expert \
    --expand-factor 2 \
    --dry-run
```

预期输出：
```
[SafetensorExpand] [dry_run] GenericMoEExpertUpcyclingExpander ...
  source:  139644 tensors, 61 shard(s)
  output:  277884 tensors, num_hidden_layers → 61
  dup-rows tensors : 120     ← 60 层 × 2（router weight + bias）
  brand-new keys   : 138240  ← 384 专家/层 × 60 层 × 6 张量（含 weight_scale_inv）
  config patches: {n_routed_experts: 768, num_experts_per_tok: 16}
```

## Step 2  执行扩增

```bash
python scripts/safetensor_expand.py auto \
    --src ./models/Kimi-K2-Base \
    --dst ./outputs/Kimi-K2-Base-2x \
    --method expert \
    --expand-factor 2 \
    --workers 8            # 建议 4–8 workers，输出约 120 个分片
```

**内存需求**：峰值约等于最大单个输入 shard × 2。Kimi-K2 每个 shard 约 8GB，
并行写出时每个 worker 占用约 16GB，8 workers 共约 128GB 内存。
若内存不足，减少 `--workers` 或改用 `--target-shard-gb 2`。

```bash
# 内存受限时
python scripts/safetensor_expand.py auto \
    --src ./models/Kimi-K2-Base \
    --dst ./outputs/Kimi-K2-Base-2x \
    --method expert \
    --expand-factor 2 \
    --workers 2 \
    --target-shard-gb 2    # 更小分片，减少单次内存占用
```

## Step 3  验证

```bash
python scripts/verify_safetensor.py \
    --src ./models/Kimi-K2-Base \
    --dst ./outputs/Kimi-K2-Base-2x
# 注：不加 --fp，加载 1T 模型需要极大内存
```

预期：
```
[Config diff]  n_routed_experts: 384 → 768
[Tensor counts]  [✓] expected 277884, got 277884
[Identity blocks zeroed]  [~] No zeroed projections (expert_upcycling，非 FP）
```

---

## fp8 扩增细节

Kimi-K2 使用 fp8 (E4M3) 量化，每个权重有对应 scale 张量：

```
mlp.experts.0.gate_proj.weight          # fp8 权重 [2048, 7168]
mlp.experts.0.gate_proj.weight_scale_inv  # 量化 scale [16, 56]
```

扩增时，新专家的 `weight` 和 `weight_scale_inv` **均被同步复制**（`TensorRecipe` 不区分主权重和 scale 张量，均视为普通张量）。这是正确行为：复制后的专家与源专家使用相同的量化参数，确保推理精度一致。

**训练时的 fp8 注意事项**：
- 需要 fp8 感知训练框架（TransformerEngine / Nanotron / Megatron-LM）
- 训练过程中 `weight_scale_inv` 会被重新计算，无需保留复制值
- 建议使用 `activation_scheme: dynamic`（与原始模型一致）

---

## Python API

```python
from llm_grow.safetensor.moe_generic import make_kimik2_upcycling

make_kimik2_upcycling(expand_factor=2, noise_scale=1e-6).expand(
    src_dir="./models/Kimi-K2-Base",
    dst_dir="./outputs/Kimi-K2-Base-2x",
    workers=8,
)
```

---

## Continued Pre-training 建议

```
规模：超大模型，建议在专用集群运行
数据量：50–100B tokens
学习率：1e-5（超大模型保守设置），cosine，warmup 3000 步
批次大小：per_device=1，gradient_accumulation=32
fp8 训练：使用 TransformerEngine 或 Nanotron
MoE 损失：aux_loss_alpha=0.001（与原始模型一致），seq_aux=True
序列长度：4096 起步，逐步增加到 8192+
监控：expert 激活分布、fp8 溢出率、router entropy
```

**输出目录检查**：

```bash
# 确认 auto_map 引用的 Python 文件已复制
ls ./outputs/Kimi-K2-Base-2x/*.py
# 预期：configuration_deepseek.py  modeling_deepseek.py  tokenization_kimi.py

# 确认 config.json 正确更新
python -c "
import json
cfg = json.load(open('./outputs/Kimi-K2-Base-2x/config.json'))
print('n_routed_experts:', cfg['n_routed_experts'])    # 768
print('num_experts_per_tok:', cfg['num_experts_per_tok'])  # 16
"
```
