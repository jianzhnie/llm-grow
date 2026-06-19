"""Tests for llm_grow.safetensor.utils."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from llm_grow.safetensor.utils import (
    expert_idx,
    expert_key_offset,
    is_expert_key,
    layer_suffix,
    parse_layer_idx,
    peek_model_config,
    rename_layer_idx,
)


class TestParseLayerIdx:
    def test_valid_key(self):
        assert parse_layer_idx("model.layers.0.self_attn.q_proj.weight") == 0
        assert parse_layer_idx("model.layers.31.mlp.gate_proj.weight") == 31

    def test_non_layer_key(self):
        assert parse_layer_idx("model.embed_tokens.weight") is None
        assert parse_layer_idx("lm_head.weight") is None


class TestRenameLayerIdx:
    def test_rename(self):
        key = "model.layers.5.self_attn.q_proj.weight"
        assert rename_layer_idx(key, 10) == "model.layers.10.self_attn.q_proj.weight"

    def test_non_layer_key_unchanged(self):
        key = "model.embed_tokens.weight"
        assert rename_layer_idx(key, 99) == key


class TestLayerSuffix:
    def test_returns_suffix(self):
        assert (
            layer_suffix("model.layers.3.mlp.gate_proj.weight")
            == "mlp.gate_proj.weight"
        )
        assert (
            layer_suffix("model.layers.0.self_attn.o_proj.weight")
            == "self_attn.o_proj.weight"
        )

    def test_non_layer_key(self):
        assert layer_suffix("model.embed_tokens.weight") is None


class TestIsExpertKey:
    def test_expert_key(self):
        assert is_expert_key("model.layers.0.mlp.experts.0.gate_proj.weight") is True

    def test_non_expert_key(self):
        assert is_expert_key("model.layers.0.mlp.gate_proj.weight") is False
        assert is_expert_key("model.layers.0.self_attn.q_proj.weight") is False


class TestExpertIdx:
    def test_returns_index(self):
        assert expert_idx("model.layers.0.mlp.experts.3.gate_proj.weight") == 3
        assert expert_idx("model.layers.2.mlp.experts.0.up_proj.weight") == 0

    def test_non_expert_returns_neg1(self):
        assert expert_idx("model.layers.0.mlp.gate_proj.weight") == -1


class TestExpertKeyOffset:
    def test_offset(self):
        key = "model.layers.0.mlp.experts.2.gate_proj.weight"
        assert (
            expert_key_offset(key, 4) == "model.layers.0.mlp.experts.6.gate_proj.weight"
        )

    def test_non_expert_unchanged(self):
        key = "model.layers.0.mlp.gate_proj.weight"
        assert expert_key_offset(key, 4) == key

    def test_zero_offset(self):
        key = "model.layers.1.mlp.experts.5.down_proj.weight"
        assert expert_key_offset(key, 0) == key


class TestPeekModelConfig:
    def test_reads_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"hidden_size": 1024, "num_hidden_layers": 32}
            (Path(tmp) / "config.json").write_text(json.dumps(cfg))
            result = peek_model_config(Path(tmp))
            assert result == cfg

    def test_missing_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = peek_model_config(Path(tmp))
            assert result == {}
