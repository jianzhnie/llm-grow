# llm-grow

<p align="center">
  <img src="docs/images/logo.svg" width="280"/>
</p>

<p align="center">
  <em>从已有 LLM Checkpoint 生长出更大模型 — 逐层扩展</em>
</p>

<p align="center">
  <a href="#安装">安装</a> &bull;
  <a href="#快速开始">快速开始</a> &bull;
  <a href="#扩增方法">方法</a> &bull;
  <a href="#方法选择">选择指南</a> &bull;
  <a href="#api-参考">API</a> &bull;
  <a href="#训练">训练</a> &bull;
  <a href="#实测结果">Benchmark</a> &bull;
  <a href="README.md">English</a>
</p>

<p align="center">
  <img src="docs/images/llmgrow.png" alt="LLM-Grow 概览">
</p>

---

从已有 LLM checkpoint **生长**出更大模型的模块化工具库。
无需从零预训练 — 按需扩展深度、宽度或专家数量，Function-Preserving 方法保证零精度损失。

**核心特性**

| | |
|---|---|
| **两层扩增体系** | 内存级 (`nn.Module`) + Safetensor 级 (mmap 流式, 峰值 ≤ 4 GB) |
| **架构自动检测** | Dense / MoE-Standard / DeepSeek-MoE / LongCat — 仅需 `config.json` |
| **6 种算法** | ZeroBlockInsert, OverlapCopy, SVDInterpInsert, MultiAxisPad, DenseToMoE, ExpertClone |
| **Function-Preserving** | ZeroBlockInsert / MultiAxisPad → 扩增后 zero-shot 精度零损失 |
| **可插拔噪声策略** | Gaussian / Uniform / ScaledGaussian 用于对称性破坏 |
| **装饰器注册机制** | `@register_expander` 一键注册内存级扩增器 |
| **完整训练工具链** | 冻结训练、知识蒸馏、渐进式掩码生长、MoE 负载均衡 |
| **真实模型验证** | Qwen2.5-0.5B, Qwen3-0.6B, Qwen3-30B-A3B, LongCat-Flash-Lite, Kimi-K2-Thinking |

---

## 安装

```bash
pip install -e .                # 基础 (扩增 + CLI)
pip install -e ".[train]"       # + 训练依赖 (DeepSpeed, Flash-Attn, Datasets)
pip install -e ".[eval]"        # + 评估依赖 (lm-eval-harness)
pip install -e ".[dev]"         # + 开发环境 (pytest, ruff, mypy)
```

**环境要求**: Python ≥ 3.10, PyTorch ≥ 2.2, Transformers ≥ 4.40, safetensors ≥ 0.4

---

## 快速开始

### CLI

```bash
# 深度扩增 (自动检测 Dense / MoE)
llm-grow expand --src /path/to/model --dst ./output \
    --method depth --num-new-layers 4

# MoE 专家扩增
llm-grow expand --src /path/to/moe_model --dst ./output \
    --method expert --expand-factor 2

# FFN 宽度扩增
llm-grow expand --src /path/to/model --dst ./output \
    --method width --ffn-size-expansion 512

# Dry-run 确认方案 (不写文件)
llm-grow expand --src /path/to/model --dst /tmp/out \
    --method depth --num-new-layers 4 --dry-run

# 并行写入 + 输出验证 + 断点续传
llm-grow expand --src /path/to/model --dst ./output \
    --method depth --num-new-layers 4 \
    --workers 8 --validate-output --resume

# 验证扩增结果 (结构检查 + FP logit 对比)
llm-grow verify --src /path/to/original --dst /path/to/expanded --fp

# 查看模型架构信息
llm-grow info --src /path/to/model
```

### Python API — Safetensor 级 (大模型, 不加载权重)

```python
from llm_grow.safetensor.auto import auto_expand

auto_expand(
    src_dir="/path/to/model",
    dst_dir="./expanded",
    method="depth",             # "depth" | "expert" | "width"
    num_new_layers=4,
    insert_strategy="uniform",  # "uniform" | "front" | "rear"
    target_shard_gb=4.0,
    workers=4,                  # 并行写入线程数
    dry_run=False,
    validate_output=True,
    resume=False,
)
```

### Python API — 内存级 (小模型 / 快速实验)

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

## 扩增方法

### 方法对比

| 方法 | FP | 方向 | 即时精度 | 推荐 CPT | 推理延迟 |
|------|:---:|:---:|:---:|:---:|:---:|
| **ZeroBlockInsert** | ✓ | 深度 | **100%** | 8–16B | +线性 |
| **OverlapCopy** | ✗ | 深度 | 50–80% | 100B+ | +线性 |
| **SVDInterpInsert** | ≈ | 深度 | 80–90% | <50B | +线性 |
| **MultiAxisPad** | ✓ | 深度+宽度 | **100%** | 30–60B | ~1.4× |
| **DenseToMoE** | ✗ | Dense→MoE | 70–85% | 50–100B | ≈不变 |
| **ExpertClone** | ≈ | 专家数 | — | 节省 32–67% | ≈不变 |

> **FP** = Function-Preserving: 扩增后输出 logits 与原始模型完全一致。

### 原理图

<p align="center">
  <img src="docs/images/zero_block_insert.svg" width="48%"/>
  <img src="docs/images/overlap_copy.svg" width="48%"/>
  <img src="docs/images/svd_interp_insert.svg" width="48%"/>
  <img src="docs/images/multi_axis_pad.svg" width="48%"/>
  <img src="docs/images/dense_to_moe.svg" width="48%"/>
  <img src="docs/images/expert_clone.svg" width="48%"/>
</p>

---

## 方法选择

```
需要扩增参数量
│
├── 超大模型 (>30B, 无法完整加载) → Safetensor 级
│   ├── Dense → 深度:   llm-grow expand --method depth
│   │         → 宽度:   llm-grow expand --method width
│   └── MoE   → 专家:   llm-grow expand --method expert
│             → 深度:   llm-grow expand --method depth
│
└── 中小模型 (可加载进内存) → 内存级
    ├── 精度优先, 数据有限     → ZeroBlockInsert (FP, 8–16B CPT)
    ├── 精确倍数, 控制延迟     → MultiAxisPad (深度+宽度, ~1.4×)
    ├── 最简实现, 数据充足     → OverlapCopy
    ├── 延迟不能增加           → DenseToMoE (top-1 激活量不变)
    └── 基座已是 MoE           → ExpertClone
```

---

## API 参考

### 两层扩增体系

| | 内存级 (`expanders/`) | Safetensor 级 (`safetensor/`) |
|---|---|---|
| **输入** | `nn.Module` | `.safetensors` 目录 |
| **输出** | `nn.Module` | `.safetensors` 目录 |
| **内存峰值** | 完整模型 | ≤ 1 个输出 shard (~4 GB) |
| **FP 验证** | 直接对比 logits | 结构检查 + 可选 logit 对比 |
| **适用场景** | 小模型 / 快速实验 | 100B+ 超大模型 |

### 自动检测架构

```python
from llm_grow.safetensor.detect import detect_model

profile = detect_model("/path/to/model")
print(profile.family)  # "dense" | "standard_moe" | "deepseek_moe" | "longcat"
```

| 检测结果 | 代表模型 | 关键特征 |
|---------|---------|---------|
| `dense` | Qwen3-0.6B/8B/14B/32B | 无 `mlp.experts.*` 键 |
| `standard_moe` | Qwen3-30B-A3B | `mlp.experts.*` + `mlp.gate.weight` |
| `deepseek_moe` | Kimi-K2-Thinking | MLA + fp8 `weight_scale_inv` + 共享专家 + dense 第 0 层 |
| `longcat` | LongCat-Flash-Lite | 双路 `self_attn.{0,1}` + 双 MLP `mlps.{0,1}` + `mlp.router.classifier` |

### `auto_expand()` — 统一入口

```python
from llm_grow.safetensor.auto import auto_expand

auto_expand(
    src_dir="/path/to/model",
    dst_dir="./expanded",
    method="depth",               # "depth" | "expert" | "width"
    num_new_layers=4,             # [depth] 新增层数
    insert_strategy="uniform",    # [depth] "uniform" | "front" | "rear"
    expand_factor=2,              # [expert] 专家倍数
    noise_scale=1e-6,             # [expert] router 噪声强度
    ffn_size_expansion=0,         # [width] intermediate_size 增量
    target_shard_gb=4.0,          # 输出 shard 大小上限
    workers=1,                    # 并行写入线程
    dry_run=False,                # True = 仅打印方案不写文件
    validate_output=False,        # True = 写入后校验
    resume=False,                 # True = 跳过已存在的 shard
)
```

### 预配置工厂函数

| 函数 | 适用模型 | 说明 |
|------|---------|------|
| `make_qwen3moe_expert_clone(factor)` | Qwen3-30B-A3B | Router: `mlp.gate.weight` |
| `make_qwen3moe_zero_block_insert(n)` | Qwen3-30B-A3B | 置零: o_proj + 全部 expert down_proj |
| `make_kimik2_expert_clone(factor)` | Kimi-K2 | fp8 感知, bias noise=0, 共享专家保留 |
| `make_kimik2_zero_block_insert(n)` | Kimi-K2 | dense 第 0 层感知, 共享专家感知 |

### 内存级扩增 — 通过 Registry 调用

```python
from llm_grow.expanders.registry import get_expander, list_expanders

print(list_expanders())
# ['dense_to_moe', 'expert_clone', 'multi_axis_pad',
#  'overlap_copy', 'svd_interp_insert', 'zero_block_insert']

expander = get_expander("zero_block_insert")()
```

### 噪声策略 (可插拔)

```python
from llm_grow.initializers.noise import GaussianNoise, UniformNoise, ScaledGaussianNoise
from llm_grow.expanders.sparse.dense_to_moe import DenseToMoEConfig

# 默认: 高斯噪声
config = DenseToMoEConfig(num_experts=8, noise_std=0.01)

# 切换为均匀分布噪声
config = DenseToMoEConfig(num_experts=8, noise_std=0.01, noise=UniformNoise())

# 缩放高斯 (std 相对于 tensor std, 与 safetensor dup_rows_noise_scale 语义一致)
config = DenseToMoEConfig(num_experts=8, noise=ScaledGaussianNoise())
```

### ExpansionPlan 序列化

```python
from llm_grow.safetensor.recipe import ExpansionPlan

plan.save_json("plan.json")          # 保存方案供离线检查
plan = ExpansionPlan.load_json("plan.json")  # 重新加载

plan.to_dict()                       # 编程式检查
dict(plan.config_patches)            # {'num_experts': 256, ...}
```

---

## 训练

### 两阶段冻结训练

```python
from llm_grow.training.freeze import (
    snapshot_param_ids, mark_new_params, freeze_original_layers, unfreeze_all,
)

original_ids = snapshot_param_ids(model)
expander.expand(model, config)
mark_new_params(model, original_ids)

freeze_original_layers(model)     # Phase 1: 仅训练新增参数
train(model, phase1_data, lr=2e-4)

unfreeze_all(model)               # Phase 2: 全量微调
train(model, phase2_data, lr=1e-5)
```

### 知识蒸馏

```python
from llm_grow.training.distillation import DistillConfig, DistillationLoss

criterion = DistillationLoss(DistillConfig(temperature=2.0, alpha=0.5))
loss = criterion(student_logits, teacher_logits, labels=labels)
```

### MoE 负载均衡

```python
from llm_grow.training.load_balance import combined_moe_loss

loss = combined_moe_loss(
    lm_loss, router_logits_list,
    num_experts=8, top_k=2, balance_coeff=1e-2, z_coeff=1e-3,
)
```

### 渐进式掩码生长

```python
from llm_grow.training.growth_scheduler import GrowthScheduleConfig, GrowthScheduler

scheduler = GrowthScheduler(GrowthScheduleConfig(
    total_steps=100_000, warmup_ratio=0.3, strategy="linear",
))
scheduler.apply_masks(model, scheduler.get_unlock_ratio(step))
```

### 验证

```python
from llm_grow.eval import verify_fp, StructuralVerifier

verify_fp("path/to/original", "path/to/expanded", atol=1e-4)

verifier = StructuralVerifier(src_dir="/path/to/original", dst_dir="/path/to/expanded")
results = verifier.run_all()  # {'config': True, 'tensor_counts': True, ...}
```

---

## 实测结果

### Function-Preserving 验证 (真实模型)

| 模型 | 方法 | `max|Δlogit|` | 结果 |
|-------|--------|:---:|:---:|
| Qwen2.5-0.5B | depth +4 (24→28) | 0.0000e+00 | ✓ PASS |
| Qwen2.5-0.5B | width +512 (inter=5376) | 0.0000e+00 | ✓ PASS |
| Qwen3-0.6B | depth +4 (28→32) | 0.0000e+00 | ✓ PASS |
| Qwen3-0.6B | width +512 (inter=3584) | 0.0000e+00 | ✓ PASS |

### 内存级扩增 (Qwen3-0.6B, 596M, 28 层, CPU)

| 方法 | 层数 | 参数量 | 倍率 | FP | 耗时 |
|------|:---:|:---:|:---:|:---:|:---:|
| ZeroBlockInsert (+7) | 28→35 | 596M→706M | 1.19× | max\|Δ\|=0.000 | 0.2s |
| OverlapCopy (overlap=8) | 28→40 | 596M→785M | 1.32× | 非FP | 0.2s |
| SVDInterpInsert (+4) | 28→32 | 596M→659M | 1.11× | ≈FP | 0.05s |
| MultiAxisPad (+4) | 28→32 | 596M→659M | 1.11× | max\|Δ\|=0.000 | 0.05s |
| DenseToMoE (×4) | 28→MoE | 596M→1.39B | 2.33× | 非FP | 6.1s |
| ExpertClone (4→8) | — | 1.39B→2.45B | 1.76× | 对称破坏后 | 1.7s |

### Safetensor 扩增 (真实模型, 全部 Config 已验证)

| 模型 | 方法 | 原始张量 | 输出张量 | Config |
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

## 项目结构

```
llm-grow/
├── src/llm_grow/
│   ├── cli.py                    # CLI 入口
│   ├── configs/                  # 共享配置 dataclass + 常量
│   ├── core/                     # ModelInspector 抽象 + 标记
│   ├── safetensor/               # Safetensor 级扩增 (mmap 流式)
│   │   ├── auto.py               #   auto_expand() + @register_expander 注册
│   │   ├── detect.py             #   ModelProfile 架构自动检测
│   │   ├── base.py               #   SafetensorExpanderBase + dry_run
│   │   ├── recipe.py             #   TensorRecipe + ExpansionPlan
│   │   ├── shard_writer.py       #   两阶段流式 ShardWriter
│   │   ├── writer.py             #   apply_recipe + worker_write_shard
│   │   ├── utils.py              #   ShardIndex + 仅 header 工具
│   │   ├── methods/              #   按扩增方法组织
│   │   └── models/               #   按模型架构组织
│   ├── expanders/                # 内存级扩增
│   │   ├── base.py               #   AbstractExpander (ABC)
│   │   ├── registry.py           #   @register_expander 装饰器
│   │   ├── depth/                #   ZeroBlockInsert / OverlapCopy / SVDInterpInsert
│   │   ├── width/                #   MultiAxisPad
│   │   └── sparse/               #   DenseToMoE / ExpertClone
│   ├── initializers/             # 权重初始化 + 噪声策略
│   │   └── noise.py              #   GaussianNoise / UniformNoise / ScaledGaussian
│   ├── training/                 # 冻结 / 蒸馏 / 调度 / 负载均衡
│   ├── eval/                     # FP 验证 / 结构检查 / 恢复曲线
│   └── utils/                    # 日志 / 模型 I/O / 扩增规则 / 插入策略
├── examples/                     # 示例脚本 (safetensor + 内存级)
├── tests/                        # 246 测试, 83% 覆盖率
└── docs/                         # 教程 + 架构图
```

---

## 扩增教程

| 模型 | 架构 | 参数量 | 教程 |
|------|:---:|:---:|------|
| Qwen3-0.6B | Dense | 596M | [docs/expand_qwen3_0.6b.md](docs/expand_qwen3_0.6b.md) |
| Qwen3-30B-A3B | MoE Standard | ~30B | [docs/expand_qwen3_30b_a3b.md](docs/expand_qwen3_30b_a3b.md) |
| Kimi-K2-Base | DeepSeek MoE | ~1T | [docs/expand_kimi_k2.md](docs/expand_kimi_k2.md) |
| LongCat-Flash-Chat | LongCat MoE | ~0.5T | [docs/expand_longcat_flash.md](docs/expand_longcat_flash.md) |

---

## 参考文献

1. **LLaMA-Pro** — Wu et al., [arXiv:2401.02415](https://arxiv.org/abs/2401.02415) (2024)
2. **SOLAR DUS** — Kim et al., [arXiv:2312.15166](https://arxiv.org/abs/2312.15166) (2023)
3. **LESA** — Yang et al., [arXiv:2502.13794](https://arxiv.org/abs/2502.13794) (2025)
4. **MSG** — Du et al., [arXiv:2305.02869](https://arxiv.org/abs/2305.02869) (2023)
5. **Net2Net** — Chen et al., [arXiv:1511.05641](https://arxiv.org/abs/1511.05641) (ICLR 2016)
6. **Sparse Upcycling** — Komatsuzaki et al., [arXiv:2212.05055](https://arxiv.org/abs/2212.05055) (ICLR 2023)
7. **ExpertClone** — Amazon AI, [arXiv:2604.19835](https://arxiv.org/abs/2604.19835) (2026)
8. **DeepSeek-V2 MLA** — DeepSeek AI, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) (2024)
