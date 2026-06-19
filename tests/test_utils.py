"""Tests for arch_info and safetensor utility modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch.nn as nn

from llm_grow.safetensor.utils import (
    expert_idx,
    expert_key_offset,
    insert_positions,
    is_expert_key,
    layer_suffix,
    parse_layer_idx,
    peek_model_config,
    rename_layer_idx,
)
from llm_grow.utils.arch_info import ArchInfo, count_params, parse_arch_info
from tests.conftest import FakeModel

# ---------------------------------------------------------------------------
# arch_info tests
# ---------------------------------------------------------------------------


class TestParseArchInfo:
    def test_model_with_config(self):
        model = FakeModel(num_layers=4, d=32)
        info = parse_arch_info(model)
        assert info.num_hidden_layers == 4
        assert info.hidden_size == 32
        assert info.vocab_size == 256
        assert info.intermediate_size == 64

    def test_model_without_config(self):
        model = nn.Linear(10, 10)
        info = parse_arch_info(model)
        assert info == ArchInfo()
        assert info.hidden_size == 0
        assert info.model_type == ""


class TestCountParams:
    def test_total_params(self):
        model = FakeModel(num_layers=2, d=16)
        total = count_params(model)
        assert total == sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_trainable_only(self):
        model = FakeModel(num_layers=2, d=16)
        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad = False
        assert count_params(model, trainable_only=True) == 0
        assert count_params(model, trainable_only=False) > 0


class TestArchInfoDataclass:
    def test_defaults(self):
        info = ArchInfo()
        assert info.hidden_size == 0
        assert info.extra == {}

    def test_extra_initialized(self):
        info = ArchInfo(hidden_size=128)
        assert info.extra == {}

    def test_extra_passed(self):
        info = ArchInfo(extra={"foo": "bar"})
        assert info.extra == {"foo": "bar"}


# ---------------------------------------------------------------------------
# safetensor/utils tests
# ---------------------------------------------------------------------------


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


class TestInsertPositions:
    def test_uniform(self):
        positions = insert_positions(8, 4, "uniform")
        assert len(positions) == 4
        assert all(0 <= p < 8 for p in positions)
        assert positions == sorted(positions)

    def test_front(self):
        positions = insert_positions(8, 3, "front")
        assert positions == [0, 1, 2]

    def test_rear(self):
        positions = insert_positions(8, 3, "rear")
        assert positions == [5, 6, 7]

    def test_zero_new(self):
        assert insert_positions(8, 0, "uniform") == []

    def test_num_new_exceeds_orig_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            insert_positions(4, 5, "uniform")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown insert_strategy"):
            insert_positions(8, 2, "middle")

    def test_single_insert_uniform(self):
        positions = insert_positions(8, 1, "uniform")
        assert len(positions) == 1

    def test_front_single(self):
        assert insert_positions(8, 1, "front") == [0]

    def test_rear_single(self):
        assert insert_positions(8, 1, "rear") == [7]


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
