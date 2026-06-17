from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ExpansionConfig:
    """通用扩增配置基类，各方法可继承并扩展字段。"""

    model_name_or_path: str = ""
    output_dir: str = "expanded_model"
    extra: dict[str, Any] = field(default_factory=dict)


class AbstractExpander(ABC):
    """所有扩增算法的统一接口。

    子类须实现：
      - expand(model, config) -> nn.Module
      - verify(original, expanded) -> bool
    """

    def __call__(self, model: nn.Module, config: ExpansionConfig) -> nn.Module:
        expanded = self.expand(model, config)
        return expanded

    @abstractmethod
    def expand(self, model: nn.Module, config: ExpansionConfig) -> nn.Module:
        """对 model 执行参数扩增，返回扩增后的新模型。"""

    def verify(
        self,
        original: nn.Module,
        expanded: nn.Module,
        *,
        num_samples: int = 4,
        seq_len: int = 32,
        atol: float = 1e-4,
        device: str | None = None,
    ) -> bool:
        """验证 function-preserving：随机输入下两个模型输出是否一致。

        适用于恒等初始化方法（LLaMA-Pro、MSG）。非 FP 方法可覆盖此方法
        返回 True 并打印 skip 提示。
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        original.eval().to(device)
        expanded.eval().to(device)

        orig_vocab = _get_vocab_size(original)
        input_ids = torch.randint(0, orig_vocab, (num_samples, seq_len), device=device)

        with torch.no_grad():
            out_orig = original(input_ids=input_ids).logits
            out_exp = expanded(input_ids=input_ids).logits

        max_err = (out_orig - out_exp).abs().max().item()
        passed = max_err < atol
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"[FP verify] max |Δlogit| = {max_err:.2e}  {status} (atol={atol})")
        return passed

    @staticmethod
    def _deep_copy_block(block: nn.Module) -> nn.Module:
        return copy.deepcopy(block)


def _get_vocab_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "vocab_size"):
        return cfg.vocab_size
    for name, param in model.named_parameters():
        if "embed" in name and param.dim() == 2:
            return param.shape[0]
    return 32000
