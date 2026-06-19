"""Tests for llm_grow.utils.arch_info."""

from __future__ import annotations

import torch.nn as nn

from llm_grow.utils.arch_info import ArchInfo, count_params, parse_arch_info
from tests.conftest import FakeModel


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
