# llm-grow

<p align="center">
  <img src="docs/images/logo.svg" width="280"/>
</p>

<p align="center">
  <em>Expand Existing Models, Layer by Layer</em>
</p>

<p align="center">
  <a href="#安装">安装</a> &bull;
  <a href="#快速开始">快速开始</a> &bull;
  <a href="#扩增方法总览">方法总览</a> &bull;
  <a href="#api-参考">API 参考</a> &bull;
  <a href="#扩增教程">教程</a>
</p>

从已有 LLM checkpoint **生长**出更大模型的模块化工具库。

**核心特性**：

- **两层扩增体系** — 内存级（`nn.Module` 原地修改）和 Safetensor 级（mmap 流式，峰值内存 ≤ 4 GB）
- **四类架构全覆盖** — Dense / MoE-Standard / DeepSeek-MoE / LongCat，自动检测自动选择
- **六种扩增算法** — LLaMA-Pro、SOLAR DUS、LESA、MSG、MoE Upcycling、Expert Upcycling
- **Function-Preserving** — LLaMA-Pro / MSG 扩增后 zero-shot 精度零损失
- **完整训练工具链** — 冻结训练、知识蒸馏、渐进式掩码生长、MoE 负载均衡

---

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [扩增方法总览](#扩增方法总览)
- [方法选择指南](#方法选择指南)
- [API 参考](#api-参考)
- [训练与评估](#训练与评估)
- [扩增教程](#扩增教程)
- [实测结果](#实测结果)
- [项目结构](#项目结构)
- [参考文献](#参考文献)

---

## 安装

```bash
pip install -e .                # 基础（扩增 + CLI）
pip install -e ".[train]"       # + 训练依赖（DeepSpeed, Flash-Attn, Datasets）
pip install -e ".[eval]"        # + 评估依赖（lm-eval-harness）
pip install -e ".[dev]"         # + 开发环境（pytest, ruff, mypy）
```

**环境要求**：Python >= 3.10, PyTorch >= 2.2, Transformers >= 4.40, safetensors >= 0.4

---

## 快速开始

### CLI（推荐）

安装后提供 `llm-grow` 命令：

```bash
# 深度扩增（自动检测 Dense / MoE）
llm-grow expand --src /path/to/model --dst ./output \
    --method depth --num-new-layers 4

# MoE 专家扩增
llm-grow expand --src /path/to/moe_model --dst ./output \
    --method expert --expand-factor 2

# FFN 宽度扩增（Dense only）
llm-grow expand --src /path/to/model --dst ./output \
    --method width --ffn-size-expansion 512

# 先 dry-run 确认方案（不写文件）
llm-grow expand --src /path/to/model --dst /tmp/out \
    --method depth --dry-run

# 验证扩增结果（结构 + FP 一致性）
llm-grow verify --src /path/to/original --dst /path/to/expanded --fp

# 查看模型架构信息
llm-grow info --src /path/to/model
```

### Python API

```python
# ── Safetensor 扩增（适合大模型，不加载权重）──────────────────
from llm_grow.safetensor.auto import auto_expand

auto_expand(
    src_dir="/path/to/model",
    dst_dir="./expanded",
    method="depth",             # "depth" | "expert" | "width"
    num_new_layers=4,
)

# ── 内存级扩增（适合小模型 / 快速实验）────────────────────────
import copy
from transformers import AutoModelForCausalLM
from llm_grow.expanders.depth.llama_pro import (
    LlamaProConfig, LlamaProExpander,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", torch_dtype="auto",
)
original = copy.deepcopy(model)
expanded = LlamaProExpander().expand(
    model,
    LlamaProConfig(num_new_layers=9, freeze_original=True),
)
LlamaProExpander().verify(original, expanded)
```

---

## 扩增方法总览

### 方法对比

| 方法 | FP | 扩展方向 | 即时精度 | 推荐 CPT | 推理延迟 |
|------|:---:|:---:|:---:|:---:|:---:|
| **LLaMA-Pro** | yes | 深度 | **100%** | 8-16B tokens | +线性 |
| **SOLAR DUS** | no | 深度 | 50-80% | 100B+ tokens | +线性 |
| **LESA** | ~yes | 深度 | 80-90% | <50B tokens | +线性 |
| **MSG** | yes | 深度+宽度 | **100%** | 30-60B tokens | ~1.4x |
| **MoE Upcycling** | no | Dense->MoE | 70-85% | 50-100B tokens | ~不变 |
| **Expert Upcycling** | ~yes | MoE 专家 | -- | 节省 32-67% | ~不变 |

> **FP** = Function-Preserving：扩增后输出与原始模型完全一致（zero-shot 精度零损失）。

### 原理图

#### LLaMA-Pro -- 恒等块插入

在均匀间隔处插入 identity block（`o_proj` / `down_proj` 置零），残差连接保证 `output = x + 0 = x`。

<p align="center"><img src="docs/images/llama_pro.svg" width="720"/></p>

#### SOLAR DUS -- 层重叠拼接

将原模型分为 upper / lower 两段（重叠区保证平滑），拼接后层数倍增。非 FP，需大量 CPT。

<p align="center"><img src="docs/images/solar_dus.svg" width="720"/></p>

#### LESA -- SVD 插值扩层

对相邻层权重做算术平均插值，新层从有意义的初始化出发，收敛速度优于 DUS。

<p align="center"><img src="docs/images/lesa.svg" width="720"/></p>

#### MSG -- 多维掩码生长

同时扩增深度（identity block）和宽度（零填充 hidden / FFN 维度），所有新参数初始化为零，严格 FP。

<p align="center"><img src="docs/images/msg.svg" width="720"/></p>

#### MoE Upcycling -- Dense 转稀疏 MoE

将 Dense FFN 复制为 N 个专家 + 随机初始化 Router。Top-K 路由使推理成本近似不变。

<p align="center"><img src="docs/images/moe_upcycling.svg" width="720"/></p>

#### Expert Upcycling -- MoE 专家数扩展

复制已有专家并施加对称性破坏（noise / drop），Router 权重对应扩展。推理成本不变。

<p align="center"><img src="docs/images/expert_upcycling.svg" width="720"/></p>

---

## 方法选择指南

```
需要扩增参数量
|
+-- 超大模型（无法完整加载）--> Safetensor 直接扩增
|   +-- Dense 模型
|   |   +-- 深度扩增        llm-grow expand --method depth
|   |   +-- FFN 宽度扩增    llm-grow expand --method width
|   +-- MoE 模型
|       +-- 专家扩增（推理成本不变）  llm-grow expand --method expert
|       +-- 深度扩增（层数增加）      llm-grow expand --method depth
|
+-- 中小模型（可加载进内存）--> 内存级扩增
    +-- 精度最优先，数据有限   --> LLaMA-Pro（FP, 8-16B tokens）
    +-- 精确 2x，控制延迟     --> MSG（深度+宽度, ~1.4x 延迟）
    +-- 最简实现，数据充足     --> SOLAR DUS
    +-- 推理延迟不能增加       --> MoE Upcycling（top-1 激活量不变）
    +-- 基座已是 MoE           --> Expert Upcycling
```

---

## API 参考

### 两层扩增体系

| | 内存级 (`expanders/`) | Safetensor 级 (`safetensor/`) |
|---|---|---|
| **输入** | `nn.Module` | `.safetensors` 目录 |
| **输出** | `nn.Module` | `.safetensors` 目录 |
| **内存峰值** | 完整模型 | <= 1 个输出 shard (~4 GB) |
| **FP 验证** | 直接对比 logits | 结构检查 + 可选 logit 对比 |
| **适用场景** | 小模型 / 快速实验 | 100B+ 超大模型 |

### Safetensor 扩增

#### 自动检测架构

`detect_model()` 从 `config.json` + 权重索引推断架构，无需加载权重：

```python
from llm_grow.safetensor.detect import detect_model

profile = detect_model("/path/to/model")
print(profile.family)   # "dense" | "standard_moe" | "deepseek_moe" | "longcat"
```

| 检测结果 | 代表模型 | 关键特征 |
|---------|---------|---------|
| `dense` | Qwen3-0.6B/8B/14B/32B | 无 `mlp.experts.*` |
| `standard_moe` | Qwen3-30B-A3B | `mlp.experts.*` + `mlp.gate.weight` |
| `deepseek_moe` | Kimi-K2-Base | MLA + fp8 + 共享专家 + dense 首层 |
| `longcat` | LongCat-Flash-Chat | 双路注意力 + 双 MLP + 512 专家 |

#### auto_expand()

```python
from llm_grow.safetensor.auto import auto_expand

auto_expand(
    src_dir="/path/to/model",
    dst_dir="./expanded",
    method="depth",               # "depth" | "expert" | "width"
    num_new_layers=4,             # [depth] 新增层数
    insert_strategy="uniform",    # [depth] "uniform" | "front" | "rear"
    expand_factor=2,              # [expert] 专家倍数
    ffn_size_expansion=0,         # [width] intermediate_size 增量
    target_shard_gb=4.0,          # 输出 shard 大小
    dry_run=False,                # True = 只打印方案不写文件
)
```

#### 预配置工厂函数

| 函数 | 适用模型 | 说明 |
|------|---------|------|
| `make_qwen3moe_upcycling(factor)` | Qwen3-30B-A3B | 专家数扩增 |
| `make_qwen3moe_depth(n)` | Qwen3-30B-A3B | 深度扩增 |
| `make_kimik2_upcycling(factor)` | Kimi-K2-Base | 专家数扩增（含 fp8 处理） |
| `make_kimik2_depth(n)` | Kimi-K2-Base | 深度扩增（含 dense 首层处理） |

#### ExpansionPlan 序列化

```python
from llm_grow.safetensor.base import ExpansionPlan

plan.save_json("my_plan.json")
plan = ExpansionPlan.load_json("my_plan.json")
```

#### Dense 与 MoE 的关键差异

恒等块要求 `Block(x) = 0`（残差连接保证输出不变），但不同架构需要置零的张量不同：

| 架构 | 必须置零的投影 | 数量 |
|------|-------------|:---:|
| Dense | `mlp.down_proj.weight` | 1 |
| Qwen3-MoE (128 experts) | 128 个 expert `down_proj` | **128** |
| Kimi-K2 (384 experts + shared) | 384 expert + shared `down_proj` | **385** |
| LongCat (512 experts + 2 MLP) | 512 expert + 2 dense `down_proj` | **514** |

> `auto_expand()` 通过 `detect_model()` 自动选择正确的 expander，避免用错。

### 内存级扩增

#### LLaMA-Pro

```python
from llm_grow.expanders.depth.llama_pro import (
    LlamaProConfig, LlamaProExpander,
)

expanded = LlamaProExpander().expand(model, LlamaProConfig(
    num_new_layers=9,           # 建议 = 原层数 // 4
    insert_strategy="uniform",  # "uniform" | "front" | "rear"
    freeze_original=True,       # Phase-1 仅训练新块
))
```

#### MSG

```python
from llm_grow.expanders.width.msg import MSGConfig, MSGExpander

expanded = MSGExpander().expand(model, MSGConfig(
    num_new_layers=10,
    hidden_size_expansion=512,
    intermediate_size_expansion=3072,
    freeze_original=True,
))
```

#### MoE Upcycling

```python
from llm_grow.expanders.sparse.moe_upcycling import (
    MoEUpcyclingConfig, MoEUpcyclingExpander,
)

expanded = MoEUpcyclingExpander().expand(
    model, MoEUpcyclingConfig(num_experts=8, top_k=2),
)
```

#### Expert Upcycling

```python
from llm_grow.expanders.sparse.expert_upcycling import (
    ExpertUpcyclingConfig, ExpertUpcyclingExpander,
    ExpertSelectionStrategy,
)

expanded = ExpertUpcyclingExpander().expand(
    moe_model,
    ExpertUpcyclingConfig(
        expand_factor=2,
        selection_strategy=ExpertSelectionStrategy.UTILITY,
    ),
)
```

---

## 训练与评估

### 两阶段冻结训练

```python
from llm_grow.training.freeze import (
    snapshot_param_ids, mark_new_params,
    freeze_original_layers, unfreeze_all,
)

# 扩增前快照 -> 扩增 -> 精确标记新增参数
original_ids = snapshot_param_ids(model)
expander.expand(model, config)
mark_new_params(model, original_ids)

freeze_original_layers(model)          # Phase-1: 仅训练新增参数
train(model, phase1_data, lr=2e-4)

unfreeze_all(model)                    # Phase-2: 全量微调
train(model, phase2_data, lr=1e-5)
```

### 知识蒸馏

```python
from llm_grow.training.distillation import (
    DistillConfig, DistillationLoss, run_teacher_inference,
)

criterion = DistillationLoss(
    DistillConfig(temperature=2.0, alpha=0.5),
)
teacher_logits = run_teacher_inference(teacher, input_ids)
loss = criterion(student_logits, teacher_logits, labels=labels)
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

### Function-Preserving 验证

```python
from llm_grow.eval import verify_fp, StructuralVerifier

# 内存级 FP 验证
verify_fp("path/to/original", "path/to/expanded", atol=1e-4)

# Safetensor 结构验证
verifier = StructuralVerifier(
    src_dir="/path/to/original",
    dst_dir="/path/to/expanded",
)
results = verifier.run_all()  # config / tensor_counts / weights / identity
```

### 精度恢复曲线追踪

```python
from llm_grow.eval import RecoveryCurveTracker

tracker = RecoveryCurveTracker("recovery.jsonl")
tracker.set_baseline({"mmlu": 0.72, "gsm8k": 0.65})
tracker.log(step=1000, tokens_seen=2e9, scores=run_eval(model))
tracker.summary()
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

## 实测结果

### 内存级扩增（Qwen3-0.6B, 596M, 28 层, CPU）

| 方法 | 层数 | 参数量 | 倍率 | FP 验证 | 耗时 |
|------|:---:|:---:|:---:|:---:|:---:|
| LLaMA-Pro (+7) | 28->35 | 596M->706M | 1.19x | max\|d\|=**0.000** | 0.2s |
| SOLAR DUS (overlap=8) | 28->40 | 596M->785M | 1.32x | 非FP（预期） | 0.2s |
| LESA (+4) | 28->32 | 596M->659M | 1.11x | 近似FP | 0.05s |
| MSG (+4) | 28->32 | 596M->659M | 1.11x | max\|d\|=**0.000** | 0.05s |
| MoE Upcycling (x4) | 28->MoE | 596M->1.39B | 2.33x | 非FP（预期） | 6.1s |
| Expert Upcycling (4->8) | -- | 1.39B->2.45B | 1.76x | 对称破坏后通过 | 1.7s |

### Safetensor 扩增（dry-run 验证）

| 模型 | 方法 | 原始张量 | 输出张量 | 新增 |
|------|------|:---:|:---:|:---:|
| Qwen3-0.6B | depth +4 层 | 311 | 355 | 44 |
| Qwen3-0.6B | LLaMA-Pro +7 | 311 | 388 | 77 |
| Qwen3-30B-A3B | expert 128->256 | 18,867 | 37,299 | 18,432 |
| Qwen3-30B-A3B | depth 48->56 | 18,867 | 22,011 | 3,144 |
| Kimi-K2-Base | expert 384->768 | 139,644 | 277,884 | 138,240 |
| Kimi-K2-Base | depth 61->65 | 139,644 | 148,952 | 9,308 |

### 运行测试

```bash
python -m pytest tests/ -q                  # 27 unit tests
python scripts/test_real_model.py           # 集成测试（Qwen3-0.6B）
python scripts/test_auto_detect.py          # 自动检测 + 分发测试
python scripts/test_longcat_dryrun.py       # LongCat dry_run
python scripts/test_qwen3_kimi_dryrun.py    # Qwen3-30B + Kimi-K2 dry_run
```

---

## 项目结构

```
llm-grow/
+-- src/llm_grow/
|   +-- cli.py                  # llm-grow 命令行入口
|   +-- safetensor/             # Safetensor 级扩增（mmap 流式）
|   |   +-- auto.py             #   auto_expand() 统一入口
|   |   +-- detect.py           #   ModelProfile 架构自动检测
|   |   +-- base.py             #   ExpansionPlan + 两阶段写出
|   |   +-- llama_pro.py        #   Dense 深度扩增
|   |   +-- solar_dus.py        #   Dense 深度扩增（DUS）
|   |   +-- msg.py              #   Dense 深度+宽度扩增
|   |   +-- moe_generic.py      #   Qwen3MoE / KimiK2 通用 MoE
|   |   +-- longcat.py          #   LongCat 专用
|   |   +-- utils.py            #   ShardIndex / header 扫描
|   +-- expanders/              # 内存级扩增
|   |   +-- base.py             #   AbstractExpander 基类
|   |   +-- depth/              #   LLaMA-Pro / SOLAR DUS / LESA
|   |   +-- width/              #   MSG / Net2Net
|   |   +-- sparse/             #   MoE Upcycling / Expert Upcycling
|   +-- initializers/           # 权重初始化（identity, SVD, 对称破坏）
|   +-- training/               # 冻结 / 蒸馏 / 调度 / 负载均衡
|   +-- eval/                   # FP 验证 / 结构验证 / 恢复曲线
+-- scripts/                    # CLI 脚本 + 集成测试
+-- tests/                      # 单元测试 (27 tests)
+-- configs/                    # 按模型分类的 YAML 配置
+-- docs/                       # 教程 + 架构图
```

---

## 配置文件

```
configs/
+-- Qwen3-0.6B/        llama_pro.yaml  solar_dus.yaml  msg.yaml
+-- Qwen3-8B/          llama_pro.yaml  msg.yaml  moe_upcycling.yaml
+-- Qwen3-30B-A3B/     expert_upcycling.yaml  depth.yaml
+-- Kimi-K2-Base/      expert_upcycling.yaml  depth.yaml
+-- LongCat-Flash-Chat/  expert_upcycling.yaml  depth.yaml
```

---

## 参考文献

1. **LLaMA-Pro** -- Wu et al., [arXiv:2401.02415](https://arxiv.org/abs/2401.02415) (2024)
2. **SOLAR DUS** -- Kim et al., [arXiv:2312.15166](https://arxiv.org/abs/2312.15166) (2023)
3. **LESA** -- Yang et al., [arXiv:2502.13794](https://arxiv.org/abs/2502.13794) (2025)
4. **MSG** -- Du et al., [arXiv:2305.02869](https://arxiv.org/abs/2305.02869) (2023)
5. **Net2Net** -- Chen et al., [arXiv:1511.05641](https://arxiv.org/abs/1511.05641) (ICLR 2016)
6. **Sparse Upcycling** -- Komatsuzaki et al., [arXiv:2212.05055](https://arxiv.org/abs/2212.05055) (ICLR 2023)
7. **Expert Upcycling** -- Amazon AI, [arXiv:2604.19835](https://arxiv.org/abs/2604.19835) (2026)
8. **DeepSeek-V2 MLA** -- DeepSeek AI, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) (2024)
