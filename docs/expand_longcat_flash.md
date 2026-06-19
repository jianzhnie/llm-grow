# LongCat-Flash-Chat 扩增教程

**架构**：LongCat 自定义 MoE（`LongcatFlashForCausalLM`）
**模型路径**：`meituan-longcat/LongCat-Flash-Chat`（HuggingFace）

---

## 架构分析

```
层数：28
hidden_size：6144
n_routed_experts：512（每层路由专家数）
zero_expert_num：256（预初始化为恒等映射的零专家）
moe_topk：12（top-k 激活专家数）
注意力：双路 MLA（self_attn.0 / self_attn.1）
Dense MLP：双路（mlps.0 / mlps.1），与 MoE 并列
Router：mlp.router.classifier.weight + mlp.router.e_score_correction_bias
auto_map：引用 configuration_longcat_flash.py / modeling_longcat_flash.py
分片：75 个 shard（model_00001-of-00075.safetensors ... ）
```

**核心特性——零专家（zero_expert）**：

LongCat 独有设计：512 路由专家中，256 个是真实专家，256 个是"零专家"（预初始化为恒等映射）。
这是 ZeroBlockInsert 思路在 MoE 上的直接实现——零专家在训练过程中逐步学习，无训练代价的容量扩展。

```
专家布局（扩增前）：
  [expert_0 .. expert_255]   → 真实专家（有训练权重）
  [expert_256 .. expert_511] → 零专家（identity 初始化，全零权重）

Router weight 形状：[512+256, 6144] = [768, 6144]
  rows [0:512]   → 真实路由分数
  rows [512:768] → 零专家路由分数
```

---

## 可用扩增方案

| 方案 | 方法 | 专家变化 | moe_topk | 参考配置 |
|------|------|:---:|:---:|------|
| **专家扩增 2x** ★ | expert_clone | 512→1024 (256+256 zero → 512+512 zero) | 12→24 | `configs/LongCat-Flash-Chat/expert_clone.yaml` |

> **注**：`moe_topk` 默认**不会**自动翻倍（`scale_moe_topk=False` 是默认值）。
> 上表中 `moe_topk: 12→24` 需要在配置中显式设置 `scale_moe_topk=True`，
> 否则扩增后 `moe_topk` 仍保持为 12。
| 深度扩增 | depth | 28→32 层 | 不变 | `configs/LongCat-Flash-Chat/depth.yaml` |

---

## 前置条件

```bash
# 下载完整模型（权重文件大，约几百GB）
huggingface-cli download meituan-longcat/LongCat-Flash-Chat \
    --local-dir ./models/LongCat-Flash-Chat

# 验证
python -c "
from llm_grow.safetensor.utils import ShardIndex
idx = ShardIndex.load('./models/LongCat-Flash-Chat')
present = sum(1 for s in idx.shard_files if (idx.model_dir/s).exists())
print(f'{present}/{len(idx.shard_files)} shards present')
"
```

---

## Step 1  Dry-run（无需权重文件）

```bash
python examples/common/safetensor_expand.py auto \
    --src ./models/LongCat-Flash-Chat \
    --dst /tmp/longcat_2x \
    --method expert \
    --expand-factor 2 \
    --dry-run
```

预期输出：
```
[SafetensorExpand] [dry_run] LongcatExpertCloneExpander ...
  source:  43756 tensors, 75 shard(s)
  output:  86764 tensors, num_hidden_layers → 28
  dup-rows tensors : 56      ← 28 层 × 2（classifier.weight + e_score_correction_bias）
  brand-new keys   : 43008   ← 新增 512 专家/层 × 28 层 × 3 张量
  config patches: {n_routed_experts: 1024, zero_expert_num: 512, moe_topk: 24}
```

> 上述 `moe_topk: 24` 仅在 `scale_moe_topk=True` 时出现；默认 `scale_moe_topk=False` 时 `moe_topk` 保持为 12。

**零专家处理说明**：

`LongcatExpertCloneExpander` 通过 `router_split=512` 正确区分零专家行：

```
router.classifier.weight 扩增逻辑：
  rows [0:512]   → 真实专家 → 复制时加小噪声（noise_scale=1e-6）
  rows [512:768] → 零专家   → 精确复制，不加噪（保持 identity 语义）
  结果：[512+512, 768+768, 6144] = [1536, 6144]
```

## Step 2  执行扩增

```bash
python examples/common/safetensor_expand.py auto \
    --src ./models/LongCat-Flash-Chat \
    --dst ./outputs/LongCat-Flash-Chat-2x \
    --method expert \
    --expand-factor 2 \
    --workers 4
```

## Step 3  验证

```bash
python examples/common/verify_safetensor.py \
    --src ./models/LongCat-Flash-Chat \
    --dst ./outputs/LongCat-Flash-Chat-2x
```

预期：
```
[Config diff]
  moe_topk: 12 → 24
  n_routed_experts: 512 → 1024
  zero_expert_num: 256 → 512

[Tensor counts]  [✓] expected 86764, got 86764
```

**检查 auto_map 引用的 Python 文件已复制**：

```bash
ls ./outputs/LongCat-Flash-Chat-2x/*.py
# 预期：configuration_longcat_flash.py  modeling_longcat_flash.py
#       expand_experts.py（随模型目录复制）
```

**加载验证**（需要权重）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./outputs/LongCat-Flash-Chat-2x",
    trust_remote_code=True,   # 使用 modeling_longcat_flash.py
    dtype="auto",
)
print(model.config.n_routed_experts)    # 1024
print(model.config.zero_expert_num)     # 512
print(model.config.moe_topk)           # 24
```

---

## Python API

```python
from llm_grow.safetensor.models.longcat import (
    LongcatExpertCloneConfig, LongcatExpertCloneExpander,
)

cfg = LongcatExpertCloneConfig(
    expand_factor=2,
    noise_scale=1e-6,
    double_zero_experts=True,  # zero_expert_num 256 → 512
)
LongcatExpertCloneExpander(cfg).expand(
    src_dir="./models/LongCat-Flash-Chat",
    dst_dir="./outputs/LongCat-Flash-Chat-2x",
    workers=4,
)
```

---

## 深度扩增方案（ZeroBlockInsert 风格）

若希望增加层数而非专家数：

```bash
python examples/common/safetensor_expand.py auto \
    --src ./models/LongCat-Flash-Chat \
    --dst ./outputs/LongCat-Flash-Chat-deeper \
    --method depth \
    --num-new-layers 4     # 28 → 32 层，参数增量 ~14%
```

每个 identity 层需置零 **516 个张量**（由 `LongcatDepthExpander` 自动处理）：

```
self_attn.0.o_proj.weight        (2 个)
self_attn.1.o_proj.weight
mlps.0.down_proj.weight          (2 个)
mlps.1.down_proj.weight
mlp.experts.{0..511}.down_proj.weight  (512 个)
```

---

## Continued Pre-training 建议

```
重点：专家分化（真实专家与零专家的行为分离）
数据量：40–80B tokens
学习率：2e-5，cosine，warmup 2000 步
批次大小：per_device=1，gradient_accumulation=16
序列长度：8192（LongCat 支持长上下文，充分利用）
MoE 损失：balance_coeff=1e-2，z_loss_coeff=1e-3

监控指标（每 2B tokens）：
  - expert 激活频率分布（均匀则负载均衡良好）
  - zero_expert 激活率（扩增后零专家应逐步获得非零激活权重）
  - Router entropy（越高越均衡）
  - MMLU / LongBench / Needle-in-Haystack
```

**使用自定义 modeling 代码训练**：

```python
# LongCat 需要 trust_remote_code=True
from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained(
    "./outputs/LongCat-Flash-Chat-2x",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
```
