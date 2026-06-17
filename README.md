# llm-grow

> 从已有 LLM checkpoint 生长出更大模型的模块化工具库。

## 支持的扩增方法

| 方法 | 路线 | Function-Preserving | 扩展方向 |
|------|:---:|:---:|:---:|
| LLaMA-Pro | 架构不变 | ✓ | 深度 |
| SOLAR DUS | 架构不变 | ✗ | 深度 |
| LESA | 架构不变 | ≈ | 深度 |
| MSG | 架构不变 | ✓ | 深度+宽度 |
| Net2Net | 架构不变 | ✓ | 深度+宽度 |
| MoE Upcycling | 架构改变 | ✗ | Dense→MoE |
| Expert Upcycling | 架构改变 | ≈ | MoE→更大MoE |

## 快速开始

```bash
pip install -e ".[train]"

# LLaMA-Pro 恒等块插入（Qwen3-8B → ~16B）
python scripts/expand_llama_pro.py \
    --config configs/llama_pro/qwen3_8b_to_16b.yaml

# MoE Upcycling（Dense → MoE）
python scripts/moe_upcycling.py \
    --config configs/moe_upcycling/qwen3_8b.yaml

# Function-Preserving 验证
python -c "
from llm_grow.eval.fp_verifier import verify_fp
verify_fp('path/to/original', 'path/to/expanded')
"
```

## 项目结构

```
src/llm_grow/
├── expanders/          # 扩增算法
│   ├── depth/          # LLaMA-Pro, SOLAR DUS, LESA
│   ├── width/          # Net2Net, MSG
│   └── sparse/         # MoE Upcycling, Expert Upcycling
├── initializers/       # 权重初始化策略
├── training/           # CPT / 蒸馏 / 冻结 / 生长调度
├── eval/               # FP 验证 / 精度恢复曲线 / Benchmark
└── utils/              # 模型 IO / 架构解析 / 参数统计
```

## 参考文献

- LLaMA-Pro: [arXiv:2401.02415](https://arxiv.org/abs/2401.02415)
- MSG: [arXiv:2305.02869](https://arxiv.org/abs/2305.02869)
- SOLAR DUS: [arXiv:2312.15166](https://arxiv.org/abs/2312.15166)
- LESA: [arXiv:2502.13794](https://arxiv.org/abs/2502.13794)
- Sparse Upcycling: [arXiv:2212.05055](https://arxiv.org/abs/2212.05055)
- Expert Upcycling: [arXiv:2604.19835](https://arxiv.org/abs/2604.19835)
