"""Tests for ZeroBlockInsertExpander."""

from __future__ import annotations

import copy

import torch

from llm_grow.expanders.depth.zero_block_insert import (
    ZeroBlockInsertConfig,
    ZeroBlockInsertExpander,
)
from tests.conftest import FakeModel


class TestZeroBlockInsertExpander:
    def _make_model(self, num_layers=8):
        return FakeModel(num_layers=num_layers, d=32)

    def test_layer_count_increases(self):
        model = self._make_model(8)
        config = ZeroBlockInsertConfig(num_new_layers=4, insert_strategy="uniform")
        ZeroBlockInsertExpander().expand(model, config)
        assert len(model.layers) == 12

    def test_function_preserving(self):
        model = self._make_model(8)
        original = copy.deepcopy(model)
        config = ZeroBlockInsertConfig(
            num_new_layers=4, insert_strategy="uniform", freeze_original=False
        )
        expanded = ZeroBlockInsertExpander().expand(model, config)

        input_ids = torch.randint(0, 256, (2, 16))
        original.eval()
        expanded.eval()
        with torch.no_grad():
            out_orig = original(input_ids).logits
            out_exp = expanded(input_ids).logits

        max_err = (out_orig - out_exp).abs().max().item()
        assert max_err < 1e-4, f"FP check failed: max_err={max_err:.4e}"

    def test_num_hidden_layers_updated(self):
        model = self._make_model(8)
        ZeroBlockInsertExpander().expand(model, ZeroBlockInsertConfig(num_new_layers=2))
        assert model.config.num_hidden_layers == 10

    def test_freeze_original_works(self):
        model = self._make_model(8)
        ZeroBlockInsertExpander().expand(
            model, ZeroBlockInsertConfig(num_new_layers=2, freeze_original=True)
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        frozen = [p for p in model.parameters() if not p.requires_grad]
        assert len(trainable) > 0
        assert len(frozen) > 0
