from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from llm_grow.configs.base import ExpansionConfig
from llm_grow.utils import get_vocab_size
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


class AbstractExpander(ABC):
    """所有 **in-memory** 模型扩增算法的统一接口。

    llm-grow 提供两层扩增体系，适用于不同场景：

    1. **In-Memory Expanders** (本模块 ``expanders/``)
       - 输入：已加载到 RAM/GPU 的 ``nn.Module``。
       - 输出：原地或拷贝修改后的 ``nn.Module``。
       - 场景：模型可以完整加载时（<= 单机显存或 CPU 内存）。
       - 优点：可直接进行 function-preserving 验证、配合 training 流程。

    2. **Safetensor Expanders** (``safetensor/base.SafetensorExpanderBase``)
       - 输入：磁盘上的 ``.safetensors`` 文件目录。
       - 输出：新的 ``.safetensors`` 文件目录。
       - 场景：模型太大无法一次性加载（100B+ 参数），或仅需产出权重文件。
       - 优点：峰值内存 ≈ 单个输出 shard（~4GB），支持并行写入。

    两者共享相同的数学算法，但操作层级不同：
      - In-Memory 操作 ``nn.Module`` 的 Python 对象和 ``nn.Parameter``。
      - Safetensor 操作序列化的 tensor 字节流，不实例化 ``nn.Module``。

    子类须实现：
      - ``expand(model, config) -> nn.Module``
      - 可选覆盖 ``verify(original, expanded) -> bool``
    """

    def __call__(self, model: nn.Module, config: ExpansionConfig) -> nn.Module:
        expanded = self.expand(model, config)
        return expanded

    @abstractmethod
    def expand(self, model: nn.Module, config: ExpansionConfig) -> nn.Module:
        """对 model 执行参数扩增（原地修改），返回修改后的模型。"""

    def verify(
        self,
        original: nn.Module,
        expanded: nn.Module,
        *,
        num_samples: int = 4,
        seq_len: int = 32,
        atol: float = 1e-4,
        device: str | None = None,
        seed: int = 42,
    ) -> bool:
        """验证 function-preserving：随机输入下两个模型输出是否一致。

        适用于恒等初始化方法（ZeroBlockInsert、MultiAxisPad）。非 FP 方法可覆盖此方法
        返回 True 并打印 skip 提示。
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        original.eval().to(device)
        expanded.eval().to(device)

        orig_vocab = get_vocab_size(original)
        torch.manual_seed(seed)
        input_ids = torch.randint(0, orig_vocab, (num_samples, seq_len), device=device)

        with torch.no_grad():
            out_orig = original(input_ids=input_ids).logits
            out_exp = expanded(input_ids=input_ids).logits

        max_err = (out_orig - out_exp).abs().max().item()
        passed = max_err < atol
        status = "PASSED" if passed else "FAILED"
        logger.info(
            "[FP verify] max |Δlogit| = %.2e  %s (atol=%s)", max_err, status, atol
        )
        return passed
