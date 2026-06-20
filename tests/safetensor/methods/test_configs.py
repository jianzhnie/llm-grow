"""Tests for the safetensor methods module (ZeroBlockInsert, MultiAxisPad configs)."""

from __future__ import annotations

from llm_grow.safetensor.methods.multi_axis_pad import MultiAxisPadSafetensorConfig
from llm_grow.safetensor.methods.zero_block_insert import (
    ZeroBlockInsertSafetensorConfig,
)


class TestZeroBlockInsertSafetensorConfig:
    def test_num_new_layers(self):
        cfg = ZeroBlockInsertSafetensorConfig(num_new_layers=5)
        assert cfg.num_new_layers == 5

    def test_default_zero_suffixes(self):
        cfg = ZeroBlockInsertSafetensorConfig()
        assert "self_attn.o_proj.weight" in cfg.attn_zero_suffixes
        assert "mlp.down_proj.weight" in cfg.mlp_zero_suffixes

    def test_insert_strategy_default(self):
        cfg = ZeroBlockInsertSafetensorConfig()
        assert cfg.insert_strategy == "uniform"


class TestMultiAxisPadSafetensorConfig:
    def test_num_new_layers(self):
        cfg = MultiAxisPadSafetensorConfig(num_new_layers=3)
        assert cfg.num_new_layers == 3

    def test_depth_disabled_by_default(self):
        cfg = MultiAxisPadSafetensorConfig()
        assert cfg.num_new_layers == 0

    def test_ffn_defaults_to_zero(self):
        cfg = MultiAxisPadSafetensorConfig()
        assert cfg.ffn_size_expansion == 0
        assert cfg.hidden_size_expansion == 0
