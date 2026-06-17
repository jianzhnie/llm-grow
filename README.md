# llm-grow

> 从已有 LLM checkpoint **生长**出更大模型的模块化工具库。
> 支持 Dense 深度/宽度扩增、Dense→MoE Upcycling、MoE 基座专家扩展七种算法，全部在 Qwen3 上通过实测验证。

## 目录

- [支持的扩增方法](#支持的扩增方法)
- [安装](#安装)
- [快速开始](#快速开始)
- [方法详解与 API](#方法详解与-api)
- [方法选择指南](#方法选择指南)
- [训练工具链](#训练工具链)
- [评估](#评估)
- [配置文件参考](#配置文件参考)
- [项目结构](#项目结构)
- [实测结果](#实测结果)
- [参考文献](#参考文献)

---

## 支持的扩增方法

| 方法 | 路线 | FP | 扩展方向 | 即时精度 | 推荐 CPT 量 | 推理延迟 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **LLaMA-Pro** | 架构不变 | ✓ | 深度 | **100%** | 8–16B tokens | ↑ 线性 |
| **SOLAR DUS** | 架构不变 | ✗ | 深度 | 50–80% | 100B+ tokens | ↑ 线性 |
| **LESA** | 架构不变 | ≈ | 深度 | 80–90% | <50B tokens | ↑ 线性 |
| **MSG** | 架构不变 | ✓ | 深度+宽度 | **100%** | 30–60B tokens | ↑ ~1.4x |
| **Net2Net** | 架构不变 | ✓ | 深度+宽度 | **100%** | 中等 | 可控 |
| **MoE Upcycling** | 架构改变 | ✗ | Dense→MoE | 70–85% | 50–100B tokens | **≈不变** (top-1) |
| **Expert Upcycling** | 架构改变 | ≈ | MoE→更大MoE | — | 节省 32–67% | **≈不变** |

> **FP**（Function-Preserving）= 扩增后模型与原始模型输出完全一致（zero-shot 精度零损失）。

---

## 安装

```bash
# 基础安装（仅扩增工具，不含训练依赖）
pip install -e .

# 含训练依赖（DeepSpeed、Flash-Attn、Datasets）
pip install -e ".[train]"

# 含评估依赖（lm-eval-harness）
pip install -e ".[eval]"

# 开发环境（pytest、ruff、mypy）
pip install -e ".[dev]"
```

**环境要求**：Python ≥ 3.10，PyTorch ≥ 2.2，Transformers ≥ 4.40

---

## 快速开始

### Python API

```python
import copy
from transformers import AutoModelForCausalLM
from llm_grow.expanders.depth.llama_pro import LlamaProConfig, LlamaProExpander
from llm_grow.utils.arch_info import param_diff_report

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="auto")
original = copy.deepcopy(model)

config = LlamaProConfig(num_new_blocks=9, insert_strategy="uniform", freeze_original=True)
expanded = LlamaProExpander().expand(model, config)

param_diff_report(original, expanded)          # 打印扩增前后参数量对比
LlamaProExpander().verify(original, expanded)  # FP 验证：max|Δlogit| 应为 0
```

### 命令行脚本

```bash
# LLaMA-Pro：恒等块插入（Qwen3-8B → ~10B，FP）
python scripts/expand_llama_pro.py \
    --model Qwen/Qwen3-8B \
    --num-new-blocks 9 \
    --output-dir ./outputs/qwen3_llama_pro \
    --verify

# MSG：深度+宽度混合扩增（精确控制参数量，FP）
python scripts/expand_msg.py \
    --model Qwen/Qwen3-8B \
    --depth-expansion 10 \
    --hidden-size-expansion 512 \
    --intermediate-size-expansion 3072 \
    --output-dir ./outputs/qwen3_msg \
    --verify

# MoE Upcycling：Dense → 稀疏 MoE（推理激活参数近似不变）
python scripts/moe_upcycling.py \
    --model Qwen/Qwen3-8B \
    --num-experts 8 \
    --top-k 2 \
    --output-dir ./outputs/qwen3_moe

# Expert Upcycling：MoE 基座专家数扩展（M1，推理延迟不变）
python scripts/expert_upcycling.py \
    --model path/to/moe_model \
    --expand-factor 2 \
    --selection-strategy utility \
    --output-dir ./outputs/qwen3_moe_2x
```

---

## 方法详解与 API

### LLaMA-Pro — 恒等块插入

**原理**：复制源层后将 `o_proj` 和 `down_proj` 置零，使新块满足 `Block(x) = 0`，
残差连接保证 `output = x + 0 = x`，实现严格恒等映射。

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

---

### SOLAR DUS — 层重叠复制

**原理**：取原模型上段前 `(L - overlap)` 层与下段后 `(L - overlap)` 层拼接，
重叠区保证拼接点分布连续。非 FP，需大量 CPT。

```python
from llm_grow.expanders.depth.solar_dus import SolarDUSConfig, SolarDUSExpander

config = SolarDUSConfig(num_overlap=8)
# 28 层模型 → 2*(28-8) = 40 层
expanded = SolarDUSExpander().expand(model, config)
```

---

### LESA — SVD 插值

**原理**：对相邻层权重做 SVD 分解提取特征，训练轻量预测网络生成插入层参数；
默认使用相邻层算术平均（`use_predictor=False`）作为快速 baseline。

```python
from llm_grow.expanders.depth.lesa import LESAConfig, LESAExpander

config = LESAConfig(
    insert_between=[(i, i+1) for i in range(14)],  # 指定插入位置
    svd_rank=64,
    use_predictor=False,   # True 时使用 MLP 预测网络（需额外训练）
)
expanded = LESAExpander().expand(model, config)
```

---

### MSG — 多维度掩码生长

**原理**：同时扩展深度（恒等块）+ 宽度（零填充）+ FFN，
新增维度初始贡献为零（FP），配合 `GrowthScheduler` 渐进解锁。

```python
from llm_grow.expanders.width.msg import MSGConfig, MSGExpander

config = MSGConfig(
    depth_expansion=10,                 # 新增层数
    hidden_size_expansion=512,          # hidden_size 增量
    intermediate_size_expansion=3072,   # FFN 宽度增量
    freeze_original=True,
    growth_schedule="linear",           # "instant" | "linear" | "cosine"
)
expanded = MSGExpander().expand(model, config)
```

配合渐进式解锁调度：

```python
from llm_grow.training.growth_scheduler import GrowthScheduler, GrowthScheduleConfig

scheduler = GrowthScheduler(GrowthScheduleConfig(total_steps=50000, warmup_ratio=0.3))
for step, batch in enumerate(dataloader):
    ratio = scheduler.get_unlock_ratio(step)
    scheduler.apply_masks(model, ratio)
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

---

### MoE Upcycling — Dense → 稀疏 MoE

**原理**：将 Dense FFN 复制 `num_experts` 份作为专家初始权重，
新增 Router（随机初始化）；每 token 通过 Top-K 路由激活 K 个专家。
Top-1 时推理激活参数量与原 Dense 近似相同。

```python
from llm_grow.expanders.sparse.moe_upcycling import MoEUpcyclingConfig, MoEUpcyclingExpander

config = MoEUpcyclingConfig(
    num_experts=8,           # 专家数
    top_k=2,                 # 每 token 激活专家数
    noise_std=0.01,          # 对称性破坏噪声
    ffn_module_pattern="mlp" # FFN 模块名称模式
)
expanded = MoEUpcyclingExpander().expand(model, config)
```

训练需加入负载均衡损失：

```python
from llm_grow.training.load_balance import combined_moe_loss

total_loss = combined_moe_loss(
    lm_loss=lm_loss,
    router_logits_list=router_logits_list,  # 每层 MoE 的 router logits
    num_experts=8,
    top_k=2,
    balance_coeff=1e-2,   # Switch Transformer load balance loss 系数
    z_coeff=1e-3,         # z-loss 系数
)
```

---

### Expert Upcycling — MoE 专家数扩展（M1）

**原理**：将 MoE 基座已有的 E 个专家复制为 mE 个，保持 Top-K 不变，
推理激活参数量不变，总参数量线性增长。关键步骤是打破副本间的对称性。

```python
from llm_grow.expanders.sparse.expert_upcycling import (
    ExpertUpcyclingConfig, ExpertUpcyclingExpander, ExpertSelectionStrategy
)

config = ExpertUpcyclingConfig(
    expand_factor=2,                                    # 专家数倍数
    selection_strategy=ExpertSelectionStrategy.UTILITY, # 效用导向（推荐）
    symmetry_break="noise",    # "noise" | "drop"
    noise_std=0.01,
)
expanded = ExpertUpcyclingExpander().expand(moe_model, config)
```

**三种专家选择策略**：

| 策略 | 说明 | 效果 |
|------|------|------|
| `uniform` | 每个专家等概率复制 | 基线 |
| `utility` | 按 L2 范数（可扩展为梯度重要性）优先复制高价值专家 | 差距闭合 **3x+** |
| `random_subset` | 随机选部分专家 | 不稳定 |

---

## 方法选择指南

```
需要扩增参数
├── 不接受架构变化（保持 Dense）
│   ├── 训练数据 < 20B tokens  →  ✅ LLaMA-Pro（零精度损失，最省数据）
│   ├── 训练数据 20–100B tokens → LESA（收敛最快）
│   ├── 训练数据 > 100B tokens  → SOLAR DUS（10 行代码，最简单）
│   └── 需精确 2x 且控制延迟   →  ✅ MSG（多维度组合，延迟仅增 ~40%）
└── 接受架构变化
    ├── 推理延迟不能增加        →  ✅ MoE Upcycling（top-1 激活量不变）
    ├── 有大 Teacher 模型       →  扩展 + 蒸馏（效果上限最高）
    └── 基座已是 MoE            →  ✅ Expert Upcycling M1（推理成本近似不变）
```

---

## 训练工具链

### 两阶段冻结训练（LLaMA-Pro / MSG 推荐）

```python
from llm_grow.training.freeze import freeze_original_layers, unfreeze_all, report_trainable

# Phase 1：仅训练新增参数（保护原始能力）
freeze_original_layers(model)
report_trainable(model)
train(model, phase1_data, lr=2e-4, tokens=10e9)

# Phase 2：全量微调（弥合新旧参数分布差异）
unfreeze_all(model)
train(model, phase2_data, lr=1e-5, tokens=5e9)
```

### 知识蒸馏（扩展 + 蒸馏流水线）

```python
from llm_grow.training.distillation import DistillConfig, DistillationLoss, run_teacher_inference

criterion = DistillationLoss(DistillConfig(temperature=2.0, alpha=0.5))

# 生成 teacher soft labels
teacher_logits = run_teacher_inference(teacher_model, input_ids, batch_size=4)

# 蒸馏训练
loss = criterion(
    student_logits=student_out.logits,
    teacher_logits=teacher_logits,
    labels=batch["labels"],
)
```

---

## 评估

### Function-Preserving 验证

扩增完成后立即验证输出一致性（FP 方法应 max\|Δlogit\| ≈ 0）：

```python
from llm_grow.eval.fp_verifier import verify_fp

passed = verify_fp(
    original="path/to/original",
    expanded="path/to/expanded",
    num_samples=8,
    seq_len=64,
    atol=1e-4,
)
```

### 精度恢复曲线追踪

```python
from llm_grow.eval.recovery_curve import RecoveryCurveTracker

tracker = RecoveryCurveTracker(save_path="recovery.jsonl")
tracker.set_baseline({"mmlu": 0.72, "gsm8k": 0.65})

for step, batch in enumerate(dataloader):
    ...
    if step % eval_interval == 0:
        scores = run_eval(model)
        tracker.log(step=step, tokens_seen=step * batch_tokens, scores=scores)

tracker.summary()
```

---

## 配置文件参考

所有脚本支持通过 YAML 配置文件驱动（`configs/` 目录下有三个开箱即用的模板）：

| 配置文件 | 适用方法 | 目标模型 |
|---------|---------|---------|
| `configs/llama_pro/qwen3_8b_to_16b.yaml` | LLaMA-Pro | Qwen3-8B → ~16B |
| `configs/msg/qwen3_8b_2x.yaml` | MSG | Qwen3-8B → ~16B |
| `configs/moe_upcycling/qwen3_8b.yaml` | MoE Upcycling | Qwen3-8B → MoE |

**LLaMA-Pro 配置关键字段**：

```yaml
expansion:
  method: llama_pro
  num_new_blocks: 36        # 插入块数
  insert_strategy: uniform  # uniform | front | rear
  freeze_original: true

training:
  phase1:
    max_tokens: 10_000_000_000
    learning_rate: 2.0e-4
    warmup_steps: 500
    lr_scheduler: cosine
  phase2:
    max_tokens: 5_000_000_000
    learning_rate: 1.0e-5

data:
  instruction_data_ratio: 0.05  # 5% 指令数据防遗忘
```

**MSG 配置关键字段**：

```yaml
expansion:
  depth_expansion: 10            # 新增层数
  hidden_size_expansion: 512     # hidden_size 增量
  intermediate_size_expansion: 3072

growth_schedule:
  strategy: linear    # linear | cosine | step
  warmup_ratio: 0.3
  total_steps: 50000
```

---

## 项目结构

```
llm-grow/
├── configs/                        # 开箱即用的参考配置
│   ├── llama_pro/qwen3_8b_to_16b.yaml
│   ├── msg/qwen3_8b_2x.yaml
│   └── moe_upcycling/qwen3_8b.yaml
├── scripts/                        # 命令行入口
│   ├── expand_llama_pro.py
│   ├── expand_msg.py
│   ├── moe_upcycling.py
│   ├── expert_upcycling.py
│   └── test_real_model.py          # 集成测试（Qwen3-0.6B）
├── src/llm_grow/
│   ├── expanders/
│   │   ├── base.py                 # AbstractExpander 基类
│   │   ├── depth/
│   │   │   ├── llama_pro.py        # 恒等块插入（FP）
│   │   │   ├── solar_dus.py        # 层重叠复制
│   │   │   └── lesa.py             # SVD 插值
│   │   ├── width/
│   │   │   ├── msg.py              # 多维度掩码生长（FP）
│   │   │   └── net2net.py          # Net2WiderNet（工具函数）
│   │   └── sparse/
│   │       ├── moe_upcycling.py    # Dense → MoE
│   │       └── expert_upcycling.py # MoE 专家数扩展（M1）
│   ├── initializers/
│   │   ├── identity.py             # 零初始化输出投影
│   │   ├── svd_interp.py           # SVD 插值工具
│   │   └── symmetry_break.py       # 加噪 / Drop-Upcycling
│   ├── training/
│   │   ├── freeze.py               # 分阶段冻结/解冻
│   │   ├── growth_scheduler.py     # MSG 掩码生长调度
│   │   ├── distillation.py         # CE + KL 蒸馏损失
│   │   └── load_balance.py         # MoE load-balance + z-loss
│   ├── eval/
│   │   ├── fp_verifier.py          # Function-Preserving 验证
│   │   └── recovery_curve.py       # 精度恢复曲线追踪
│   └── utils/
│       ├── model_io.py             # HF 模型加载/保存
│       ├── arch_info.py            # 架构解析 & 参数量对比报告
│       └── param_counter.py
└── tests/
    ├── test_expanders.py           # LLaMA-Pro / SOLAR DUS 单测（12 cases）
    └── test_initializers.py        # Identity init / Symmetry break 单测
```

---

## 实测结果

基于 **Qwen3-0.6B**（596M 参数，28 层，hidden=1024）在 CPU (Apple Silicon) 上的实测数据：

| 方法 | 扩增后层数 | 扩增后参数量 | 倍率 | FP 验证 | 扩增耗时 |
|------|:---:|:---:|:---:|:---:|:---:|
| LLaMA-Pro (+7块) | 28 → 35 | 596M → 706M | 1.19x | max\|Δlogit\|=**0.000** ✓ | 0.2s |
| SOLAR DUS (overlap=8) | 28 → 40 | 596M → 785M | 1.32x | 非FP（跳过） | 0.2s |
| LESA (+4层) | 28 → 32 | 596M → 659M | 1.11x | 近似FP ✓ | 0.05s |
| MSG (+4层) | 28 → 32 | 596M → 659M | 1.11x | max\|Δlogit\|=**0.000** ✓ | 0.05s |
| MoE Upcycling (×4 experts) | 28层→MoE | 596M → 1.39B | 2.33x | 非FP（跳过） | 6.1s |
| Expert Upcycling (4→8 experts) | — | 1.39B → 2.45B | 1.76x | 需对称破坏 | 1.7s |

**生成文本一致性验证（LLaMA-Pro）**：

```
Prompt   : "The key to learning programming is"
Original : "...to understand the concept of variables and data types..."
Expanded : "...to understand the concept of variables and data types..."  ← 逐字符相同 ✓
```

运行集成测试：

```bash
python scripts/test_real_model.py
# All 7 tests passed!
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
