# llm-grow

> 从已有 LLM checkpoint **生长**出更大模型的模块化工具库。
> 同时支持 **内存级扩增**（加载模型后修改）和 **Safetensor 直接扩增**（无需加载权重，适合超大模型）。
> 覆盖 Dense / MoE-Standard / DeepSeek-MoE / LongCat 四类架构，全部通过实测验证。

## 目录

- [支持的扩增方法](#支持的扩增方法)
- [安装](#安装)
- [快速开始](#快速开始)
- [Safetensor 直接扩增](#safetensor-直接扩增)
- [内存级扩增 API](#内存级扩增-api)
- [方法选择指南](#方法选择指南)
- [训练工具链](#训练工具链)
- [配置文件参考](#配置文件参考)
- [扩增教程（按模型）](#扩增教程按模型)
- [项目结构](#项目结构)
- [实测结果](#实测结果)
- [参考文献](#参考文献)

---

## 支持的扩增方法

### 内存级扩增（加载模型后操作）

| 方法 | FP | 扩展方向 | 即时精度 | 推荐 CPT | 推理延迟 |
|------|:---:|:---:|:---:|:---:|:---:|
| **LLaMA-Pro** | ✓ | 深度 | **100%** | 8–16B tokens | ↑ 线性 |
| **SOLAR DUS** | ✗ | 深度 | 50–80% | 100B+ tokens | ↑ 线性 |
| **LESA** | ≈ | 深度 | 80–90% | <50B tokens | ↑ 线性 |
| **MSG** | ✓ | 深度+宽度 | **100%** | 30–60B tokens | ↑ ~1.4x |
| **MoE Upcycling** | ✗ | Dense→MoE | 70–85% | 50–100B tokens | **≈不变** |
| **Expert Upcycling** | ≈ | MoE专家扩展 | — | 节省 32–67% | **≈不变** |

### Safetensor 直接扩增（无需加载权重）

| 方法 | 适用架构 | 扩展轴 | 内存峰值 |
|------|:---:|:---:|:---:|
| `auto depth` | Dense / MoE / LongCat | 深度（自动选 expander） | ≤ 1 shard |
| `auto expert` | 任意 MoE | 专家数 | ≤ 1 shard × 2 |
| `auto width` | Dense only | FFN 宽度 | ≤ 1 shard |
| `llama_pro` | Dense | 深度 | ≤ 1 shard |
| `solar_dus` | Dense | 深度 | ≤ 1 shard |
| `msg` | Dense | 深度+FFN宽度 | ≤ 1 shard |

> **FP**（Function-Preserving）= 扩增后模型输出与原始模型完全一致（zero-shot 精度零损失）。

---

## 安装

```bash
pip install -e .                # 基础（扩增工具）
pip install -e ".[train]"       # 含训练依赖（DeepSpeed、Flash-Attn、Datasets）
pip install -e ".[eval]"        # 含评估依赖（lm-eval-harness）
pip install -e ".[dev]"         # 开发环境（pytest、ruff、mypy）
```

**环境要求**：Python ≥ 3.10，PyTorch ≥ 2.2，Transformers ≥ 4.40，safetensors ≥ 0.4

---

## 快速开始

### Safetensor 直接扩增（推荐用于大模型）

```bash
# 自动检测架构，一行搞定
python scripts/safetensor_expand.py auto \
    --src /path/to/any_model \
    --dst ./output \
    --method depth \
    --num-new-layers 4

# 先 dry-run 确认方案（无需权重文件）
python scripts/safetensor_expand.py auto \
    --src /path/to/model --dst /tmp/out \
    --method depth --dry-run

# MoE 专家扩增
python scripts/safetensor_expand.py auto \
    --src /path/to/moe_model \
    --dst ./output \
    --method expert --expand-factor 2
```

### 内存级扩增（小模型 / 快速实验）

```python
import copy
from transformers import AutoModelForCausalLM
from llm_grow.expanders.depth.llama_pro import LlamaProConfig, LlamaProExpander

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="auto")
original = copy.deepcopy(model)

expanded = LlamaProExpander().expand(
    model, LlamaProConfig(num_new_blocks=9, freeze_original=True)
)
LlamaProExpander().verify(original, expanded)  # max|Δlogit| 应为 0
```

---

## Safetensor 直接扩增

不加载完整模型权重，通过 **mmap 流式读写** safetensor 文件实现扩增：
- 内存峰值 ≤ 单个输出 shard（默认 4 GB）
- 支持任意分片数（75 分片的 LongCat-Flash 亦可处理）
- 先用 `dry_run()` 验证方案，再下载权重执行

### 自动检测：Dense vs MoE

`detect_model()` 从 `config.json` + 权重索引推断完整架构画像，无需加载权重：

```python
from llm_grow.safetensor.detect import detect_model

profile = detect_model("/path/to/model")
print(profile.summary())
```

**四类架构自动识别**：

| 检测结果 | 代表模型 | 关键特征 |
|---------|---------|---------|
| `dense` | Qwen3-0.6B/8B/14B/32B | 无 `mlp.experts.*`，有 `mlp.down_proj` |
| `standard_moe` | Qwen3-30B-A3B | `mlp.experts.*` + `mlp.gate.weight`，无 fp8 |
| `deepseek_moe` | Kimi-K2-Base | MLA 注意力 + fp8 `weight_scale_inv` + 共享专家 + dense 首层 |
| `longcat` | LongCat-Flash-Chat | 双路注意力 `self_attn.0/1` + 双 MLP `mlps.0/1` + 512 专家 |

**已验证模型**：

```
Qwen3-0.6B      → dense           ✓
Qwen3-30B-A3B   → standard_moe    ✓  (128 experts/layer, 48 layers)
Kimi-K2-Base    → deepseek_moe    ✓  (384 experts/layer, fp8, dense layer-0)
LongCat-Flash   → longcat         ✓  (512 experts/layer, dual-attn)
```

### 核心区别：identity block 的差异

**这是 Dense 与 MoE 扩增最关键的不同点。**

恒等块要求 `Block(x) = 0`，使残差连接保证 `output = x + 0 = x`。
但 Dense 和 MoE 的 FFN 结构不同，需要零化的张量也不同：

| 架构 | 必须置零的输出投影 | 数量 |
|------|-----------------|:---:|
| Dense | `mlp.down_proj.weight` | 1 |
| Qwen3-MoE (128 experts) | `mlp.experts.{0..127}.down_proj.weight` | **128** |
| Kimi-K2 (384 experts + shared) | `mlp.experts.{0..383}.down_proj.weight` + `mlp.shared_experts.down_proj.weight` | **385** |
| LongCat (512 experts + 2 dense MLP) | 所有 expert down_proj + `mlps.{0,1}.down_proj.weight` | **514** |

> ⚠️ **用错的后果**：若对 MoE 模型使用 Dense 的 `LlamaProSafetensorExpander`，
> 它只会零化 `mlp.down_proj.weight`（在 MoE 层根本不存在），
> 导致所有专家 FFN 输出仍为非零值，identity block 失效，zero-shot 精度立刻崩塌。

`auto_expand()` 通过 `detect_model()` 自动规避这个问题。

**不同架构的扩展轴对比**：

```
Dense 模型扩增轴：
  ├── 深度（depth）    → 插入 identity 层（LLaMA-Pro / SOLAR DUS / LESA）
  └── 宽度（width）    → 零填充 FFN 维度（MSG）

MoE 模型扩增轴：
  ├── 深度（depth）    → 插入 identity 层（与 Dense 相似，但需零化所有专家）
  └── 专家（expert）   → 复制已有专家（MoE 独有，推理成本不变）✅
```

### CLI 用法

```bash
# ── auto 模式（推荐，自动检测架构）────────────────────────────────────────
# Dense 深度扩增（Qwen3-0.6B 28→32 层）
python scripts/safetensor_expand.py auto \
    --src /path/to/Qwen3-0.6B --dst ./out \
    --method depth --num-new-layers 4

# MoE 专家扩增（Qwen3-30B-A3B 128→256 experts）
python scripts/safetensor_expand.py auto \
    --src /path/to/Qwen3-30B-A3B --dst ./out \
    --method expert --expand-factor 2

# Dense 宽度扩增（FFN +512）
python scripts/safetensor_expand.py auto \
    --src /path/to/Qwen3-0.6B --dst ./out \
    --method width --ffn-size-expansion 512

# Dense 调用 expert 会明确报错：
# ValueError: method='expert' requires MoE model, detected family='dense'

# ── 显式指定 expander（细粒度控制）──────────────────────────────────────
python scripts/safetensor_expand.py llama_pro \
    --src /path/to/model --dst ./out --num-new-layers 7

python scripts/safetensor_expand.py solar_dus \
    --src /path/to/model --dst ./out --num-overlap 8

python scripts/safetensor_expand.py msg \
    --src /path/to/model --dst ./out \
    --num-new-layers 4 --ffn-size-expansion 1024

python scripts/safetensor_expand.py moe_expert \
    --src /path/to/moe_model --dst ./out --expand-factor 2

# ── dry-run（无需权重文件，只验证方案）───────────────────────────────────
python scripts/safetensor_expand.py auto \
    --src /path/to/model --dst /tmp/x --method depth --dry-run
```

### Python API（Safetensor）

```python
from llm_grow.safetensor.auto import auto_expand
from llm_grow.safetensor.detect import detect_model

# 查看模型画像
profile = detect_model("/path/to/model")
print(profile.family)             # "dense" | "standard_moe" | "deepseek_moe" | "longcat"
print(profile.is_moe)             # True/False
print(profile.experts_per_moe_layer)  # 0 for dense
print(profile.has_fp8)            # True for Kimi-K2

# 自动扩增
auto_expand(
    src_dir="/path/to/model",
    dst_dir="./expanded",
    method="depth",               # "depth" | "expert" | "width"
    num_new_layers=4,
    insert_strategy="uniform",
    target_shard_gb=4.0,
    dry_run=False,
)

# 手动选择 expander
from llm_grow.safetensor.moe_generic import make_qwen3moe_upcycling
make_qwen3moe_upcycling(expand_factor=2).expand(
    src_dir="/path/to/Qwen3-30B-A3B",
    dst_dir="./Qwen3-30B-A3B-256experts",
)
```

**预配置工厂函数**：

| 函数 | 适用模型 | 说明 |
|------|---------|------|
| `make_qwen3moe_upcycling(factor)` | Qwen3-30B-A3B | 专家数扩增 |
| `make_qwen3moe_depth(n)` | Qwen3-30B-A3B | 深度扩增 |
| `make_kimik2_upcycling(factor)` | Kimi-K2-Base | 专家数扩增（含fp8处理） |
| `make_kimik2_depth(n)` | Kimi-K2-Base | 深度扩增（含dense首层处理） |

### Dry-run 与验证

```bash
# 1. Dry-run：无需权重文件，在几秒内验证扩增方案
python scripts/safetensor_expand.py auto \
    --src /path/to/model --dst /tmp/x --method depth --dry-run
# 输出：plan 张量数、zero-out 数、brand-new keys 等统计

# 2. 扩增后验证（结构 + FP）
python scripts/verify_safetensor.py \
    --src /path/to/original \
    --dst /path/to/expanded \
    --fp   # 可选：加载模型并做 logit 一致性检查
```

验证脚本执行 5 项检查：
- **Config diff**：num_hidden_layers、intermediate_size 等变化确认
- **Tensor counts**：输出张量数与理论预期一致
- **Original weights preserved**：原始层权重未被修改（扫描最近匹配）
- **Identity blocks zeroed**：o_proj / down_proj 已置零
- **FP logit check**：加载两个模型，随机输入对比 logits（`--fp`）

---

## 内存级扩增 API

适用于小/中等规模模型（可完整加载进内存的场景）。

### LLaMA-Pro — 恒等块插入

```python
from llm_grow.expanders.depth.llama_pro import LlamaProConfig, LlamaProExpander

config = LlamaProConfig(
    num_new_blocks=9,           # 插入块数，建议 = 原层数 // 4
    insert_strategy="uniform",  # "uniform" | "front" | "rear"
    freeze_original=True,       # Phase-1 仅训练新块
)
expanded = LlamaProExpander().expand(model, config)
```

**推荐场景**：精度最优先、训练数据有限（< 20B tokens）。

### MSG — 多维度掩码生长

```python
from llm_grow.expanders.width.msg import MSGConfig, MSGExpander

config = MSGConfig(
    depth_expansion=10,
    hidden_size_expansion=512,
    intermediate_size_expansion=3072,
    freeze_original=True,
    growth_schedule="linear",
)
expanded = MSGExpander().expand(model, config)
```

### MoE Upcycling — Dense → 稀疏 MoE

```python
from llm_grow.expanders.sparse.moe_upcycling import MoEUpcyclingConfig, MoEUpcyclingExpander
from llm_grow.training.load_balance import combined_moe_loss

expanded = MoEUpcyclingExpander().expand(
    model, MoEUpcyclingConfig(num_experts=8, top_k=2)
)
# 训练时加入负载均衡损失
loss = combined_moe_loss(lm_loss, router_logits_list, num_experts=8, top_k=2)
```

### Expert Upcycling — MoE 专家数扩展

```python
from llm_grow.expanders.sparse.expert_upcycling import (
    ExpertUpcyclingConfig, ExpertUpcyclingExpander, ExpertSelectionStrategy
)

expanded = ExpertUpcyclingExpander().expand(moe_model, ExpertUpcyclingConfig(
    expand_factor=2,
    selection_strategy=ExpertSelectionStrategy.UTILITY,
))
```

---

## 方法选择指南

```
需要扩增参数
├── 超大模型（无法完整加载）→ 用 Safetensor 直接扩增
│   ├── Dense 模型
│   │   ├── 深度扩增        python safetensor_expand.py auto --method depth
│   │   └── FFN 宽度扩增    python safetensor_expand.py auto --method width
│   └── MoE 模型
│       ├── 专家扩增（推理成本不变）  python safetensor_expand.py auto --method expert
│       └── 深度扩增（层数增加）      python safetensor_expand.py auto --method depth
│
└── 中小模型（可加载进内存）→ 用内存级扩增
    ├── 精度最优先，数据有限   → ✅ LLaMA-Pro（FP，8-16B tokens）
    ├── 精确 2x，控制延迟      → ✅ MSG（深度+宽度，延迟 ~1.4x）
    ├── 最简实现，数据充足     → SOLAR DUS（10 行代码）
    ├── 推理延迟不能增加       → MoE Upcycling（top-1 激活量不变）
    ├── 有大 Teacher 模型      → 扩展 + 蒸馏（效果上限最高）
    └── 基座已是 MoE           → Expert Upcycling M1
```

---

## 训练工具链

### 两阶段冻结训练

```python
from llm_grow.training.freeze import freeze_original_layers, unfreeze_all, report_trainable

freeze_original_layers(model)          # Phase-1：仅训练新增参数
train(model, phase1_data, lr=2e-4)

unfreeze_all(model)                    # Phase-2：全量微调
train(model, phase2_data, lr=1e-5)
```

### 知识蒸馏

```python
from llm_grow.training.distillation import DistillConfig, DistillationLoss, run_teacher_inference

criterion = DistillationLoss(DistillConfig(temperature=2.0, alpha=0.5))
teacher_logits = run_teacher_inference(teacher_model, input_ids)
loss = criterion(student_logits, teacher_logits, labels=batch["labels"])
```

### MoE 负载均衡

```python
from llm_grow.training.load_balance import combined_moe_loss

loss = combined_moe_loss(
    lm_loss, router_logits_list,
    num_experts=8, top_k=2,
    balance_coeff=1e-2, z_coeff=1e-3,
)
```

---

## 评估

### Function-Preserving 验证

```python
from llm_grow.eval.fp_verifier import verify_fp

verify_fp("path/to/original", "path/to/expanded", num_samples=8, atol=1e-4)
```

### 精度恢复曲线追踪

```python
from llm_grow.eval.recovery_curve import RecoveryCurveTracker

tracker = RecoveryCurveTracker("recovery.jsonl")
tracker.set_baseline({"mmlu": 0.72, "gsm8k": 0.65})
tracker.log(step=1000, tokens_seen=2e9, scores=run_eval(model))
tracker.summary()
```

---

## 配置文件参考

按模型分类，每种方法一个 YAML（仅含 `model` / `expansion` / `output`）：

```
configs/
├── Qwen3-0.6B/      llama_pro.yaml  solar_dus.yaml  msg.yaml
├── Qwen3-8B/        llama_pro.yaml  msg.yaml  moe_upcycling.yaml
├── Qwen3-30B-A3B/   expert_upcycling.yaml  depth.yaml
├── Kimi-K2-Base/    expert_upcycling.yaml  depth.yaml
└── LongCat-Flash-Chat/  expert_upcycling.yaml  depth.yaml
```

---

## 扩增教程（按模型）

| 模型 | 架构 | 参数量 | 教程 |
|------|:---:|:---:|------|
| Qwen3-0.6B | Dense | 596M | [docs/expand_qwen3_0.6b.md](docs/expand_qwen3_0.6b.md) |
| Qwen3-30B-A3B | MoE Standard | ~30B | [docs/expand_qwen3_30b_a3b.md](docs/expand_qwen3_30b_a3b.md) |
| Kimi-K2-Base | DeepSeek MoE + fp8 | ~1T | [docs/expand_kimi_k2.md](docs/expand_kimi_k2.md) |
| LongCat-Flash-Chat | LongCat MoE | ~0.5T | [docs/expand_longcat_flash.md](docs/expand_longcat_flash.md) |

---

## 项目结构

```
llm-grow/
├── configs/
│   ├── Qwen3-0.6B/    Qwen3-8B/    Qwen3-30B-A3B/
│   ├── Kimi-K2-Base/  LongCat-Flash-Chat/
├── docs/
│   ├── expand_qwen3_0.6b.md         # Dense 扩增教程
│   ├── expand_qwen3_30b_a3b.md      # MoE Standard 扩增教程
│   ├── expand_kimi_k2.md            # DeepSeek MoE + fp8 教程
│   └── expand_longcat_flash.md      # LongCat MoE 教程
├── scripts/
│   ├── safetensor_expand.py   # ★ 统一 Safetensor CLI（auto / llama_pro / ...）
│   ├── verify_safetensor.py   # ★ 扩增验证（结构 + FP）
│   ├── expand_llama_pro.py    # 内存级 LLaMA-Pro
│   ├── expand_msg.py          # 内存级 MSG
│   └── test_*.py              # 集成测试
├── src/llm_grow/
│   ├── safetensor/
│   │   ├── detect.py          # ModelProfile 自动检测（Dense/MoE 区分）
│   │   ├── auto.py            # auto_expand() 统一入口
│   │   ├── base.py            # 两阶段写出 + 并行 + router_split
│   │   ├── utils.py           # ShardIndex + 头部扫描 + auto_detect_shard_size
│   │   ├── llama_pro.py / solar_dus.py / msg.py
│   │   ├── moe_generic.py     # Qwen3MoE / KimiK2 通用 MoE 扩增
│   │   └── longcat.py         # LongCat 专用（router_split 零专家处理）
│   ├── expanders/             # 内存级扩增
│   ├── initializers/          # 权重初始化策略
│   ├── training/              # CPT / 蒸馏 / 冻结 / 负载均衡
│   └── eval/                  # FP 验证 / 精度恢复曲线
└── tests/  test_expanders.py  test_initializers.py
```

---

## 实测结果

### 内存级扩增（Qwen3-0.6B，596M，28层，CPU）

| 方法 | 层数 | 参数量 | 倍率 | FP 验证 | 耗时 |
|------|:---:|:---:|:---:|:---:|:---:|
| LLaMA-Pro (+7块) | 28→35 | 596M→706M | 1.19x | max\|Δ\|=**0.000** ✓ | 0.2s |
| SOLAR DUS (overlap=8) | 28→40 | 596M→785M | 1.32x | 非FP（预期） | 0.2s |
| LESA (+4层) | 28→32 | 596M→659M | 1.11x | 近似FP ✓ | 0.05s |
| MSG (+4层) | 28→32 | 596M→659M | 1.11x | max\|Δ\|=**0.000** ✓ | 0.05s |
| MoE Upcycling (×4 experts) | 28→MoE | 596M→1.39B | 2.33x | 非FP（预期） | 6.1s |
| Expert Upcycling (4→8) | — | 1.39B→2.45B | 1.76x | 对称破坏后 ✓ | 1.7s |

生成文本一致性（LLaMA-Pro FP 验证）：
```
Prompt  : "The key to learning programming is"
原始模型 : "...to understand the concept of variables and data types..."
扩增模型 : "...to understand the concept of variables and data types..."  ← 逐字符相同 ✓
```

### Safetensor 直接扩增（无权重，dry_run 验证）

| 模型 | 方法 | 原始张量 | 输出张量 | 新增 | 耗时 |
|------|------|:---:|:---:|:---:|:---:|
| Qwen3-0.6B (Dense) | depth +4层 | 311 | 355 | 44 | 2.9s |
| Qwen3-0.6B (Dense) | LLaMA-Pro +7块 | 311 | 388 | 77 | 2.9s |
| Qwen3-0.6B (Dense) | SOLAR DUS | 311 | 443 | 132 | 4.0s |
| Qwen3-0.6B (Dense) | MSG depth+FFN | 311 | 355 | 44 | 2.9s |
| Qwen3-30B-A3B (MoE) | expert 128→256 | 18,867 | 37,299 | 18,432 | dry_run |
| Qwen3-30B-A3B (MoE) | depth 48→56层 | 18,867 | 22,011 | 3,144 | dry_run |
| Kimi-K2-Base (MoE+fp8) | expert 384→768 | 139,644 | 277,884 | 138,240 | dry_run |
| Kimi-K2-Base (MoE+fp8) | depth 61→65层 | 139,644 | 148,952 | 9,308 | dry_run |

FP 验证（Qwen3-0.6B safetensor，有权重）：
```
LLaMA-Pro: max|Δlogit| = 0.000e+00  ✓
MSG:        max|Δlogit| = 0.000e+00  ✓
SOLAR DUS:  max|Δlogit| = 1.211e+01  (预期，非FP方法)
```

运行所有测试：
```bash
python -m pytest tests/ -q                  # 12 unit tests
python scripts/test_real_model.py           # 7 integration tests (Qwen3-0.6B)
python scripts/test_auto_detect.py          # 26 auto-detect + dispatch tests
python scripts/test_longcat_dryrun.py       # LongCat dry_run (2 cases)
python scripts/test_qwen3_kimi_dryrun.py    # Qwen3-30B + Kimi-K2 dry_run (4 cases)
```

---

## 参考文献

1. **LLaMA-Pro** — Wu et al., [arXiv:2401.02415](https://arxiv.org/abs/2401.02415) (2024)
2. **SOLAR DUS** — Kim et al., [arXiv:2312.15166](https://arxiv.org/abs/2312.15166) (2023)
3. **LESA** — Yang et al., [arXiv:2502.13794](https://arxiv.org/abs/2502.13794) (2025)
4. **MSG** — Du et al., [arXiv:2305.02869](https://arxiv.org/abs/2305.02869) (2023)
5. **Net2Net** — Chen et al., [arXiv:1511.05641](https://arxiv.org/abs/1511.05641) (ICLR 2016)
6. **Sparse Upcycling** — Komatsuzaki et al., [arXiv:2212.05055](https://arxiv.org/abs/2212.05055) (ICLR 2023)
7. **Expert Upcycling** — Amazon AI, [arXiv:2604.19835](https://arxiv.org/abs/2604.19835) (2026)
8. **Cluster-Aware Upcycling** — [arXiv:2604.13508](https://arxiv.org/abs/2604.13508) (2026)
9. **DeepSeek-V2 MLA** — DeepSeek AI, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) (2024)
10. **Skywork-MoE** — [Hugging Face](https://huggingface.co/Skywork/Skywork-MoE) (2024)
