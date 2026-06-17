"""Function-preserving verifier: check output consistency after expansion."""

from __future__ import annotations

import torch
import torch.nn as nn


def verify_fp(
    original: nn.Module | str,
    expanded: nn.Module | str,
    *,
    num_samples: int = 4,
    seq_len: int = 64,
    atol: float = 1e-4,
    device: str | None = None,
    verbose: bool = True,
) -> bool:
    """验证扩增后模型与原始模型在随机输入下的输出一致性。

    Args:
        original:    原始模型（nn.Module 或 HF 模型路径）。
        expanded:    扩增后模型（nn.Module 或 HF 模型路径）。
        num_samples: 随机测试样本数。
        seq_len:     输入序列长度。
        atol:        允许的最大 logit 误差。
        device:      运行设备，默认自动选择。
        verbose:     是否打印详细报告。

    Returns:
        True 表示通过验证（max error < atol）。
    """
    if isinstance(original, str):
        original = _load_model(original)
    if isinstance(expanded, str):
        expanded = _load_model(expanded)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    original.eval().to(device)
    expanded.eval().to(device)

    vocab_size = _get_vocab_size(original)
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)

    results = []
    with torch.no_grad():
        for i in range(num_samples):
            ids = input_ids[i].unsqueeze(0)
            logits_orig = original(input_ids=ids).logits
            logits_exp = expanded(input_ids=ids).logits
            max_err = (logits_orig - logits_exp).abs().max().item()
            mean_err = (logits_orig - logits_exp).abs().mean().item()
            results.append((max_err, mean_err))

    max_errs = [r[0] for r in results]
    mean_errs = [r[1] for r in results]
    overall_max = max(max_errs)
    overall_mean = sum(mean_errs) / len(mean_errs)
    passed = overall_max < atol

    if verbose:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(
            f"[FP Verify] {status}\n"
            f"  max |Δlogit|  = {overall_max:.4e}  (threshold={atol})\n"
            f"  mean |Δlogit| = {overall_mean:.4e}\n"
            f"  samples={num_samples}, seq_len={seq_len}, device={device}"
        )
    return passed


def _load_model(path: str) -> nn.Module:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto")


def _get_vocab_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "vocab_size"):
        return cfg.vocab_size
    return 32000
