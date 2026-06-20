"""Tests for configs/constants.py."""

from __future__ import annotations

from llm_grow.configs.constants import (
    DEFAULT_TARGET_SHARD_BYTES,
    DEFAULT_VERIFY_ATOL,
    DEFAULT_VERIFY_NUM_SAMPLES,
    DEFAULT_VERIFY_SEED,
    DEFAULT_VERIFY_SEQ_LEN,
    WEIGHT_PRESERVE_ATOL,
    ZERO_CHECK_ATOL,
)


class TestConstants:
    def test_shard_bytes_is_4gb(self):
        assert DEFAULT_TARGET_SHARD_BYTES == 4 * 1024**3

    def test_verify_defaults(self):
        assert DEFAULT_VERIFY_ATOL == 1e-4
        assert DEFAULT_VERIFY_SEQ_LEN == 32
        assert DEFAULT_VERIFY_NUM_SAMPLES == 4
        assert DEFAULT_VERIFY_SEED == 42

    def test_thresholds(self):
        assert WEIGHT_PRESERVE_ATOL < DEFAULT_VERIFY_ATOL
        assert ZERO_CHECK_ATOL < WEIGHT_PRESERVE_ATOL
