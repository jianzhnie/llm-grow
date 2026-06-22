"""Centralized model path configuration.

Pre-configured for the local HuggingFace hub at
``/home/jianzhnie/llmtuner/hfhub/models``.  Set the corresponding
environment variables to override any path at runtime.

Environment variables (optional — fall back to the local paths below):

    export LLM_GROW_QWEN3_06B="/path/to/Qwen3-0.6B"
    export LLM_GROW_QWEN3_30B="/path/to/Qwen3-30B-A3B"
    export LLM_GROW_KIMI_K2="/path/to/Kimi-K2-Thinking"
    export LLM_GROW_LONGCAT="/path/to/LongCat-Flash-Lite"
"""

from __future__ import annotations

import os
from pathlib import Path

_HF_HUB = Path("/home/jianzhnie/llmtuner/hfhub/models")

# ── Qwen (Dense) ──────────────────────────────────────────────────────────────
QWEN25_05B = os.environ.get(
    "LLM_GROW_QWEN25_05B",
    str(_HF_HUB / "Qwen" / "Qwen2.5-0.5B"),
)
QWEN25_15B = os.environ.get(
    "LLM_GROW_QWEN25_15B",
    str(_HF_HUB / "Qwen" / "Qwen2.5-1.5B"),
)
QWEN3_06B = os.environ.get(
    "LLM_GROW_QWEN3_06B",
    str(_HF_HUB / "Qwen" / "Qwen3-0.6B"),
)
QWEN3_4B = os.environ.get(
    "LLM_GROW_QWEN3_4B",
    str(_HF_HUB / "Qwen" / "Qwen3-4B"),
)

# ── Qwen (MoE) ────────────────────────────────────────────────────────────────
QWEN3_30B = os.environ.get(
    "LLM_GROW_QWEN3_30B",
    str(_HF_HUB / "Qwen" / "Qwen3-30B-A3B"),
)

# ── DeepSeek / Kimi ───────────────────────────────────────────────────────────
KIMI_K2 = os.environ.get(
    "LLM_GROW_KIMI_K2",
    str(_HF_HUB / "moonshotai" / "Kimi-K2-Thinking"),
)

# ── LongCat ───────────────────────────────────────────────────────────────────
LONGCAT = os.environ.get(
    "LLM_GROW_LONGCAT",
    str(_HF_HUB / "meituan-longcat" / "LongCat-Flash-Lite"),
)

# ── helpers ────────────────────────────────────────────────────────────────────


def require_path(name: str, path: str) -> str:
    """Validate that a model path exists and return it.

    Raises:
        FileNotFoundError: if the path does not exist and no override was
            provided via the corresponding environment variable.
    """
    if not Path(path).exists():
        env_var = f"LLM_GROW_{name}"
        raise FileNotFoundError(
            f"Model path not found: {path}\n"
            f"Set environment variable {env_var} to override."
        )
    return path
