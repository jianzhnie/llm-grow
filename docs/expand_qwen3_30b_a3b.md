# Qwen3-30B-A3B 扩增教程

**架构**：MoE Standard（`Qwen3MoeForCausalLM`）
**参数量**：~30B 总参数，激活约 3B（top-8 路由）
**模型路径**：`Qwen/Qwen3-30B-A3B`（HuggingFace）

---

## 架构分析

```
层数：48
hidden_size：2048
moe_intermediate_size：768（每专家 FFN 宽度）
num_experts：128（每层路由专家数）
num_experts_per_tok：8（top-k，激活专家数）
num_attention_heads：32  num_key_value_heads：4（GQA）
vocab_size：151936  tie_word_embeddings：False
```

**参数分布**：专家参数约占总参数 96%（128 专家 × 4.72M/专家 × 48 层 ≈ 29B），
因此专家数扩增是最高效的 2x 路径。

**MoE 扩增与 Dense 扩增的关键区别**：

| | Dense | Qwen3-30B-A3B (MoE) |
|---|---|---|
| Identity block 需置零 | 1 个 down_proj | **128 个** expert down_proj |
| 可选扩增轴 | 深度 / 宽度 | 专家数 ★ / 深度 |
| 推理成本 | 深度扩增后线性增加 | 专家扩增后**几乎不变** |

---

## 可用扩增方案

| 方案 | 方法 | 目标参数量 | 倍率 | 推理成本 | 参考配置 |
|------|------|:---:|:---:|:---:|------|
| **专家扩增 2x** ★ | 128→256 experts | ~59B | ~2.0x | 激活↑（top-8→16） | `configs/Qwen3-30B-A3B/expert_clone.yaml` |
| 专家扩增（推理成本不变） | 128→256，topk 不变 | ~59B | ~2.0x | **不变** | 修改 YAML：`num_experts_per_tok: 8` |
| 深度扩增 | 48→60 层 | ~34B | ~1.13x | ↑ 线性 | `configs/Qwen3-30B-A3B/depth.yaml` |

> **注**：深度扩增对 MoE 模型增量有限（专家参数占 96%），优先选择专家扩增。

---

## 前置条件

```bash
# 下载模型（仅 index，无实际权重时可先 dry-run）
huggingface-cli download Qwen/Qwen3-30B-A3B \
    --local-dir ./models/Qwen3-30B-A3B
```

---

## Step 1  Dry-run（无权重验证方案）

```bash
python scripts/safetensor_expand.py auto \
    --src ./models/Qwen3-30B-A3B \
    --dst /tmp/qwen3_30b_2x \
    --method expert \
    --expand-factor 2 \
    --dry-run
```

预期输出：
```
[SafetensorExpand] [dry_run] GenericMoEExpertUpcyclingExpander ...
  source:  18867 tensors, 16 shard(s)
  output:  37299 tensors, num_hidden_layers → 48
  dup-rows tensors : 48      ← 48 层 router.weight 行翻倍
  brand-new keys   : 18432   ← 新增 128 个专家/层 × 48 层 × 3 个张量
  config patches: {num_experts: 256, num_experts_per_tok: 16}
```

## Step 2  执行扩增

```bash
python scripts/safetensor_expand.py auto \
    --src ./models/Qwen3-30B-A3B \
    --dst ./outputs/Qwen3-30B-A3B-2x \
    --method expert \
    --expand-factor 2 \
    --workers 4            # 并行写出，加速大模型分片写入
```

预期输出目录：
```
outputs/Qwen3-30B-A3B-2x/
├── config.json               # num_experts: 256, num_experts_per_tok: 16
├── model-00001-of-XXXXX.safetensors  ...（约 32 个分片）
├── model.safetensors.index.json
├── tokenizer* / vocab.json / merges.txt
```

## Step 3  验证

```bash
python scripts/verify_safetensor.py \
    --src ./models/Qwen3-30B-A3B \
    --dst ./outputs/Qwen3-30B-A3B-2x
# --fp 需要加载完整模型（~30B），内存充足时可加
```

---

## 保持推理成本不变的专家扩增

若要专家数翻倍但**推理激活成本保持不变**（仍只激活 8 个专家），
修改 `configs/Qwen3-30B-A3B/expert_clone.yaml`：

```yaml
expansion:
  config_patches:
    num_experts: 256
    num_experts_per_tok: 8    # 保持 top-8，不翻倍
```

此配置下：总参数 ~2x，推理每 token 激活的 expert FLOPs 不变，仅 Router 计算量微增。

---

## Python API

```python
from llm_grow.safetensor.moe_generic import make_qwen3dense_to_moe

make_qwen3dense_to_moe(expand_factor=2, noise_scale=1e-6).expand(
    src_dir="./models/Qwen3-30B-A3B",
    dst_dir="./outputs/Qwen3-30B-A3B-2x",
    workers=4,
)
```

---

## Continued Pre-training 建议

```
训练阶段：单阶段全参数 CPT（不适合 Phase-1 冻结，需所有专家参与分化）
数据量：30–50B tokens
学习率：3e-5，cosine scheduler，warmup 2000 步
MoE 负载均衡损失：balance_coeff=1e-2，z_coeff=1e-3
数据混合：通用 Web 60% + 代码 20% + 数学 10% + 科学 10%
批次大小：per_device=1，gradient_accumulation=16（等效全局 batch ≈ 4M tokens）
评估：每 5B tokens，监控各专家激活频率（防 expert collapse）
```

```python
from llm_grow.training.load_balance import combined_moe_loss

total_loss = combined_moe_loss(
    lm_loss=lm_loss,
    router_logits_list=router_logits_list,
    num_experts=256,
    top_k=16,
    balance_coeff=1e-2,
    z_coeff=1e-3,
)
```
