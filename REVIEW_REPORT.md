# llm-grow 代码库 Review 报告

> Review 日期：2026-06-19  
> Review 范围：`src/llm_grow/`、`tests/`、`examples/`、`docs/`、`README.md`、`pyproject.toml`

## 总体评估

`llm-grow` 是一个设计思路清晰的 LLM 参数扩增工具库，**内存级（`expanders/`）+ Safetensor 级（`safetensor/`）双轨架构**是其核心亮点，能够同时覆盖小模型实验和超大模型（100B+）mmap 流式扩增场景。代码结构合理、模块职责分离清楚，`TensorRecipe` + `ExpansionPlan` 的声明式设计在 `safetensor/` 层表现尤为出色。

当前主要风险集中在：**3 处已确认的运行时 bug**、Safetensor 层测试覆盖严重不足、文档与代码不同步、以及部分类型安全和跨层一致性问题。

---

## 确认的关键 Bug（建议立即修复）

| # | 问题 | 位置 | 验证结果 |
|---|------|------|---------|
| 1 | **ImportError**：`auto.py` 从 `llm_grow.safetensor.moe_generic` 导入，实际模块在 `llm_grow.safetensor.models.moe_generic` | `src/llm_grow/safetensor/auto.py:249` | 已复现 `ModuleNotFoundError` |
| 2 | **README 示例 import 路径错误**：`llm_grow.expanders.width.msg` 不存在，应为 `multi_axis_pad` | `README.md:300` | 已复现 `ModuleNotFoundError` |
| 3 | **教程引用不存在的 `scripts/` 目录**：多篇教程使用 `python scripts/safetensor_expand.py` / `scripts/verify_safetensor.py` | `docs/expand_*.md` 多处 | `scripts/` 目录不存在，实际在 `examples/common/` |
| 4 | **`verify_fp` 签名不匹配**：README 把 `atol` 当位置参数传，但函数签名为 keyword-only | `README.md:397` | 阅读源码确认 |

---

## 一、架构与代码结构

### 优点

- **双轨体系设计合理**：`expanders/` 负责内存中 `nn.Module` 修改，`safetensor/` 负责磁盘级 mmap 流式改写，二者共享 `insert_positions()` 等公共逻辑。
- **抽象接口清晰**：`AbstractExpander`（`expanders/base.py`）定义了统一的 `expand()` + `verify()`；`SafetensorExpanderBase`（`safetensor/base.py`）通过 `_build_plan()` 让子类只负责生成 `ExpansionPlan`。
- **`TensorRecipe` / `ExpansionPlan` 设计优秀**：可序列化、可 dry-run、可离线审阅，是 Safetensor 层的核心资产。
- **职责分离好**：detect / expand / verify / train / eval / initializers 六大子包边界清楚。

### 问题与建议

1. **`expanders/__init__.py` 为空**
   - 顶层 `llm_grow/__init__.py` 只导入了 `expanders` 下的类，但 `expanders/__init__.py` 没有任何导出。
   - 建议：在 `expanders/__init__.py` 统一导出 public API，与 `safetensor/__init__.py` 保持一致。

2. **跨层依赖方向不当**
   - `expanders/depth/zero_block_insert.py` 从 `safetensor/utils.py` 导入 `insert_positions()`。
   - 建议：将 `insert_positions()` 移到 `llm_grow/utils/` 或 `llm_grow/common/`，避免内存层依赖 Safetensor 层。

3. **配置类重复**
   - 每种算法都有内存级 `*Config` 和 Safetensor 级 `*SafetensorConfig`（如 `ZeroBlockInsertConfig` vs `ZeroBlockInsertSafetensorConfig`），字段高度重合但无继承关系。
   - 建议：抽取公共 base config 或 mixin，减少重复和维护成本。

4. **`safetensor/auto.py` 硬编码分发器难以扩展**
   - `_build_expander()` 和 `_build_depth_expander()` 包含大量条件分支和硬编码 import。
   - 建议：引入注册表模式，例如 `@register_expander("qwen3_moe", "depth")`，新增架构时不需要修改核心分发逻辑。

5. **`TensorRecipe` 正在变成"神级 dataclass"**
   - 16 个字段覆盖 zero/pad/dup/interp/noise/create 等多种操作，多数 recipe 只用到 2-3 个字段。
   - 建议：长期来看可考虑拆分为 discriminated union 或组合模式；短期至少用 `dataclasses.astuple()` 替换 `_worker_write_shard()` 里脆弱的位置解包。

6. **`eval/` 存在循环依赖风险**
   - `eval/fp_verifier.py` 从 `expanders/base.py` 导入私有函数 `_get_vocab_size`。
   - 建议：把 `_get_vocab_size` 移到 `utils/`，避免 `eval` 层依赖 `expanders` 层。

---

## 二、代码质量

### 静态检查

- **Ruff**：全部通过。
- **Mypy**：**10 个错误**，分布在 6 个文件，其中多个是运行时 monkey-patch 导致的真实类型安全问题。

### 主要问题

| 文件 | 行号 | 问题 | 建议 |
|------|------|------|------|
| `utils/logger_utils.py` | 78 | 动态给 `LogRecord` 加 `rank` 属性 | 使用自定义 `LogRecord` 子类或 TypedDict 包装 |
| `utils/logger_utils.py` | 167 | `handlers` 列表类型被推断为 `StreamHandler`，但追加了 `FileHandler` | 显式声明 `handlers: list[logging.Handler] = []` |
| `training/freeze.py` | 40 | 动态给 `nn.Parameter` 加 `_is_new_growth` | 自定义 `Parameter` 子类或加 `# type: ignore` 并注释 |
| `expanders/width/multi_axis_pad.py` | 131, 141 | 同上 | 同上 |
| `expanders/depth/zero_block_insert.py` | 116, 124, 137 | `_is_new_growth` monkey-patch + `Module \| None` 赋值 | 修复类型注解或重构遍历逻辑 |
| `expanders/sparse/expert_clone.py` | 178 | `result = []` 缺少类型注解 | 加 `result: list[int] = []` |
| `utils/arch_info.py` | 25 | `None` 赋值给声明为 `dict` 的变量 | 使用 `dict[str, Any] \| None` 类型 |

### 其他代码质量问题

1. **`safetensor/base.py:303-305` — 静默跳过错误**
   - `_write_shards()` 中如果 `src_meta is None` 直接 `continue`。recipe 中缺失源张量属于 plan 损坏，应该抛出清晰的异常。

2. **`safetensor/base.py:576` — 不必要的 `.clone()`**
   - 对 passthrough 张量也执行 `clone()`，浪费内存。应仅在需要 zero_out/pad/noise 等变换时才克隆。

3. **`expanders/sparse/dense_to_moe.py:164` — 魔法数 fallback**
   - `_get_hidden_size()` 找不到配置时直接返回硬编码 `4096`，且无警告。建议至少打 warning。

4. **`expanders/width/multi_axis_pad.py:98-106` — 子串匹配不精确**
   - `_classify_linear()` 用 `"gate_proj" in name` 判断，可能误匹配自定义模块名。建议匹配最后的 leaf name。

5. **`training/growth_scheduler.py:55` — 边界 bug**
   - `step` 策略在 `progress` 接近 1.0 但不到 1.0 时返回 `0.75`，无法在最后一步完全解锁。建议用 `math.ceil` 或调整阶梯逻辑。

6. **中英文 docstring 混用**
   - 公开 API 建议统一用英文 docstring，提升开源可访问性；内部实现可保留中文。

7. **缺少 mypy 配置**
   - `pyproject.toml` 只把 mypy 列为 dev 依赖，没有 `[tool.mypy]` 配置。建议加上基础配置并修复现有错误。

---

## 三、测试覆盖

### 现状

- 10 个测试文件，约 1200 行测试代码，主要覆盖 `training/`、`utils/`、`initializers/`、`safetensor/base.py`。
- **严重偏单元测试，缺乏集成测试**。

### 关键缺口

| 模块 | 测试状态 | 风险 |
|------|---------|------|
| `safetensor/detect.py` | 完全无测试 | 核心自动检测逻辑未验证 |
| `safetensor/auto.py` | 完全无测试 | `auto_expand()` 入口未验证 |
| `safetensor/methods/` | 完全无测试 | 所有 Safetensor 算法实现未测试 |
| `safetensor/models/` | 完全无测试 | MoE/LongCat 专用扩增未测试 |
| `eval/structural.py`, `eval/fp_verifier.py`, `eval/recovery_curve.py` | 基本无测试 | 验证工具自身未验证 |
| `cli.py` | 无测试 | 命令行入口未测试 |
| `expanders/depth/overlap_copy.py` | 仅层数测试 | 无前向传播测试 |
| `expanders/depth/svd_interp_insert.py` | 仅层数测试 | 无 `use_predictor=True`、无 FP 测试 |
| `expanders/sparse/dense_to_moe.py` | 仅结构测试 | 无路由逻辑、无负载均衡集成 |
| `expanders/sparse/expert_clone.py` | 仅结构测试 | 无 `RANDOM_SUBSET`、无 `drop` 策略 |

### 测试质量问题

- **缺少 `@pytest.mark.parametrize`**：大量测试用硬编码值，未系统覆盖不同模型尺寸、策略、边界条件。
- **边界条件和错误路径测试不足**：`num_new_layers=0`、`expand_factor=1`、`num_overlap >= num_layers` 等未覆盖。
- **FP 保证验证不完整**：
  - `ZeroBlockInsert`、`MultiAxisPad`（单独 depth/width）有 FP 测试。
  - `MultiAxisPad` 的 **depth+width 组合** 无 FP 测试。
  - `SVDInterpInsert`、`OverlapCopy`、`DenseToMoE`、`ExpertClone` 的输出变化行为未测试。
- **示例目录 `examples/` 的 dry-run 脚本不在 pytest 测试套件中**，无法通过 `pytest` 自动运行。

### 建议

1. **最高优先级**：补充 Safetensor 层集成测试——至少覆盖 `detect_model()`、`auto_expand()` 的 dry-run，以及一个端到端的文件 I/O 扩增流程。
2. 为所有 expander 补充边界条件测试和错误路径测试。
3. 增加 CLI 测试（可用 `subprocess` 或 `argparse` 模拟）。
4. 将 `examples/*/test_*.py` 纳入主测试套件或明确标记为集成测试。
5. 对声称 FP 的方法，增加组合场景和更严格的 logits 对比。

---

## 四、文档与示例

### 已确认的问题

1. **`README.md:300` 错误 import**
   ```python
   from llm_grow.expanders.width.msg import ...  # msg 不存在
   ```
   应为 `llm_grow.expanders.width.multi_axis_pad`。

2. **教程引用不存在的 `scripts/` 目录**
   - `docs/expand_qwen3_0.6b.md`、`docs/expand_qwen3_30b_a3b.md`、`docs/expand_kimi_k2.md`、`docs/expand_longcat_flash.md` 多处使用 `python scripts/safetensor_expand.py` / `scripts/verify_safetensor.py`。
   - 实际文件在 `examples/common/safetensor_expand.py` 和 `examples/common/verify_safetensor.py`。

3. **`README.md:397` `verify_fp` 调用签名错误**
   - README 写 `verify_fp("path/to/original", "path/to/expanded", atol=1e-4)`。
   - 实际签名为 `verify_fp(original, expanded, *, num_samples=4, seq_len=64, atol=1e-4, ...)`，`atol` 是 keyword-only。

4. **`README.md` 项目结构路径过时**
   - 第 471-495 行把 `zero_block_insert.py` / `overlap_copy.py` / `multi_axis_pad.py` 列在 `safetensor/` 根目录，实际在 `safetensor/methods/`。

5. **`docs/expand_qwen3_0.6b.md:114` 使用 `msg` 作为算法名**
   - `python scripts/safetensor_expand.py msg ...` 中 `msg` 不是 `safetensor_expand.py` 支持的选择，应为 `multi_axis_pad`。

6. **`MoEWidthExpander`、`GrowthScheduler`、`Net2Net`、`SVDInterpInsert` 内存级实现缺失文档**
   - README 方法表和 API 参考未覆盖这些功能。

### 建议

- 对 README 和 4 篇教程做一次同步校对，确保所有 import 路径、脚本路径、CLI 参数与代码一致。
- 为 Safetensor 的 `MoEWidthExpander`、训练的 `GrowthScheduler`、以及内存级 `Net2Net`/`SVDInterpInsert` 补充 API 示例。

---

## 五、优先级行动清单

### P0（立即修复，影响运行时正确性）

1. 修复 `src/llm_grow/safetensor/auto.py:249` 的 import 路径：`llm_grow.safetensor.moe_generic` -> `llm_grow.safetensor.models.moe_generic`。
2. 修复 `README.md:300` 的 import 路径：`expanders.width.msg` -> `expanders.width.multi_axis_pad`。
3. 修正所有教程中 `scripts/` 路径为 `examples/common/`。
4. 修正 `README.md:397` 的 `verify_fp` 调用示例。

### P1（短期修复，显著提升质量）

5. 修复 mypy 10 个错误，尤其是 monkey-patch 属性和类型注解。
6. 在 `safetensor/base.py:303` 将缺失 `src_meta` 的静默跳过改为抛出清晰异常。
7. 为 `expanders/__init__.py` 添加公共 API 导出。
8. 将 `insert_positions()` 从 `safetensor/utils.py` 迁移到 `utils/`。
9. 修复 `growth_scheduler.py:55` 的阶梯边界 bug。
10. 减少 passthrough 张量的不必要 `.clone()`。

### P2（中期补强）

11. 为 Safetensor 层补充核心集成测试（`detect.py`、`auto.py`、端到端文件 I/O）。
12. 为 CLI 增加测试。
13. 补充 `MultiAxisPad` 组合（depth+width）FP 测试。
14. 引入 expander 注册表模式，替代 `auto.py` 硬编码分发。
15. 统一公开 API docstring 语言（建议英文），并补齐缺失的 API 文档。

### P3（长期优化）

16. 统一内存级与 Safetensor 级的 config 类，减少重复。
17. 考虑将 `TensorRecipe` 拆分为更小的组合类型。

---

## 总结

`llm-grow` 的架构设计和核心抽象是扎实的，但当前有 3 处已确认的运行时 bug 需要立即修复，同时 Safetensor 层测试、文档同步、类型安全是接下来最值得投入的三个方面。
