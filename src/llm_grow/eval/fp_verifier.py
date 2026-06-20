"""Function-preserving verifier: check output consistency after expansion."""

from __future__ import annotations

import torch
import torch.nn as nn

from llm_grow.utils import get_vocab_size
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


def verify_fp(
    original: nn.Module | str,
    expanded: nn.Module | str,
    *,
    num_samples: int = 4,
    seq_len: int = 64,
    atol: float = 1e-4,
    device: str | None = None,
    verbose: bool = True,
    seed: int = 42,
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
        seed:        随机种子，确保结果可复现。

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

    vocab_size = get_vocab_size(original)
    torch.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)

    results = []
    with torch.inference_mode():
        logits_orig = original(input_ids=input_ids).logits
        logits_exp = expanded(input_ids=input_ids).logits
        diff = (logits_orig - logits_exp).abs()
        # Per-sample max/mean while keeping the forward pass batched.
        max_errs = diff.view(num_samples, -1).max(dim=1).values.tolist()
        mean_errs = diff.view(num_samples, -1).mean(dim=1).tolist()
        results = list(zip(max_errs, mean_errs, strict=True))

    max_errs = [r[0] for r in results]
    mean_errs = [r[1] for r in results]
    overall_max = max(max_errs)
    overall_mean = sum(mean_errs) / len(mean_errs)
    passed = bool(overall_max < atol)

    if verbose:
        status = "PASSED" if passed else "FAILED"
        logger.info(
            "[FP Verify] %s  max|Δlogit|=%.4e (threshold=%s)  "
            "mean|Δlogit|=%.4e  samples=%d seq_len=%d device=%s",
            status,
            overall_max,
            atol,
            overall_mean,
            num_samples,
            seq_len,
            device,
        )
    return passed


def _load_model(path: str) -> nn.Module:
    from llm_grow.utils.model_io import load_model

    return load_model(path, dtype=torch.float32, device_map="cpu")
