"""Shared constants used across multiple modules to avoid magic numbers."""

from __future__ import annotations

# ── Verification defaults ─────────────────────────────────────────────────────

DEFAULT_VERIFY_ATOL: float = 1e-4
"""Absolute tolerance for function-preserving logit comparison."""

DEFAULT_VERIFY_SEQ_LEN: int = 32
"""Sequence length for verification random inputs."""

DEFAULT_VERIFY_NUM_SAMPLES: int = 4
"""Number of random samples for verification."""

DEFAULT_VERIFY_SEED: int = 42
"""Random seed for reproducible verification."""

# ── Weight preservation thresholds ────────────────────────────────────────────

WEIGHT_PRESERVE_ATOL: float = 1e-6
"""Threshold for checking if original weights are preserved."""

ZERO_CHECK_ATOL: float = 1e-9
"""Threshold for checking if a tensor is effectively zero."""

# ── Shard size ────────────────────────────────────────────────────────────────

DEFAULT_TARGET_SHARD_BYTES: int = 4 * 1024**3
"""Default output shard size (4 GB)."""

# ── FP verification ───────────────────────────────────────────────────────────

DEFAULT_FP_MAX_SIZE_GB: float = 80.0
"""Max model size (GB) allowed for function-preserving verification."""

# ── Identity block projection names ──────────────────────────────────────────

ATTN_OUTPUT_PROJ_NAMES: list[str] = ["o_proj", "out_proj"]
"""Default attention output projection layer names to zero for identity blocks."""

MLP_OUTPUT_PROJ_NAMES: list[str] = ["down_proj", "fc2"]
"""Default MLP output projection layer names to zero for identity blocks."""

DEFAULT_IDENTITY_ZERO_SUFFIXES: list[str] = [
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
]
"""Default layer suffixes that must be zeroed in identity blocks.

Combines attention and MLP output projections into a single list.
"""
