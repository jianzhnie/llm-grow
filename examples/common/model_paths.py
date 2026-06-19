"""Centralized model path configuration via environment variables.

Set these environment variables before running example scripts:

    export LLM_GROW_QWEN3_06B="/path/to/Qwen/Qwen3-0.6B"
    export LLM_GROW_QWEN3_30B="/path/to/Qwen/Qwen3-30B-A3B"
    export LLM_GROW_KIMI_K2="/path/to/moonshotai/Kimi-K2-Base"
    export LLM_GROW_LONGCAT="/path/to/meituan-longcat/LongCat-Flash-Chat"
    export LLM_GROW_REPO_DIR="/path/to/llm-grow"
"""

from __future__ import annotations

import os

QWEN3_06B = os.environ.get("LLM_GROW_QWEN3_06B", "")
QWEN3_30B = os.environ.get("LLM_GROW_QWEN3_30B", "")
KIMI_K2 = os.environ.get("LLM_GROW_KIMI_K2", "")
LONGCAT = os.environ.get("LLM_GROW_LONGCAT", "")
REPO_DIR = os.environ.get("LLM_GROW_REPO_DIR", "")


def require_path(name: str, path: str) -> str:
    """Validate that a model path is set and non-empty."""
    if not path:
        env_var = f"LLM_GROW_{name}"
        raise RuntimeError(
            f"Model path not configured. Set environment variable: {env_var}"
        )
    return path
