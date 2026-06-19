"""Tests for MultiAxisPadExpander."""

from __future__ import annotations

import copy

import torch

from llm_grow.expanders.width.multi_axis_pad import (
    MultiAxisPadConfig,
    MultiAxisPadExpander,
)
from tests.conftest import FakeModel


class TestMultiAxisPadExpander:
    def setup_method(self):
        torch.manual_seed(42)

    def _make_model(self, num_layers=8):
        return FakeModel(num_layers=num_layers, d=32)

    def test_depth_only_increases_layers(self):
        model = self._make_model(8)
        config = MultiAxisPadConfig(depth_expansion=4)
        MultiAxisPadExpander().expand(model, config)
        assert len(model.layers) == 12

    def test_width_expansion_changes_sizes(self):
        model = self._make_model(4)
        config = MultiAxisPadConfig(
            intermediate_size_expansion=16, freeze_original=False
        )
        expanded = MultiAxisPadExpander().expand(model, config)
        assert expanded.config.intermediate_size == 64 + 16

    def test_hidden_size_expansion(self):
        model = self._make_model(4)
        config = MultiAxisPadConfig(hidden_size_expansion=8, freeze_original=False)
        expanded = MultiAxisPadExpander().expand(model, config)
        assert expanded.config.hidden_size == 32 + 8

    def test_function_preserving_width(self):
        model = self._make_model(4)
        original = copy.deepcopy(model)
        config = MultiAxisPadConfig(
            intermediate_size_expansion=16, freeze_original=False
        )
        expanded = MultiAxisPadExpander().expand(model, config)

        input_ids = torch.randint(0, 256, (2, 8))
        original.eval()
        expanded.eval()
        with torch.no_grad():
            out_orig = original(input_ids).logits
            out_exp = expanded(input_ids).logits

        max_err = (out_orig - out_exp).abs().max().item()
        assert max_err < 1e-4, f"FP check failed: max_err={max_err:.4e}"

    def test_function_preserving_depth(self):
        model = self._make_model(8)
        original = copy.deepcopy(model)
        config = MultiAxisPadConfig(depth_expansion=2, freeze_original=False)
        expanded = MultiAxisPadExpander().expand(model, config)

        input_ids = torch.randint(0, 256, (2, 8))
        original.eval()
        expanded.eval()
        with torch.no_grad():
            out_orig = original(input_ids).logits
            out_exp = expanded(input_ids).logits

        max_err = (out_orig - out_exp).abs().max().item()
        assert max_err < 1e-4, f"FP check failed: max_err={max_err:.4e}"

    def test_function_preserving_depth_and_ffn_width(self):
        model = self._make_model(8)
        original = copy.deepcopy(model)
        config = MultiAxisPadConfig(
            depth_expansion=2,
            intermediate_size_expansion=16,
            freeze_original=False,
        )
        expanded = MultiAxisPadExpander().expand(model, config)

        input_ids = torch.randint(0, 256, (2, 8))
        original.eval()
        expanded.eval()
        with torch.no_grad():
            out_orig = original(input_ids).logits
            out_exp = expanded(input_ids).logits

        max_err = (out_orig - out_exp).abs().max().item()
        assert max_err < 1e-4, f"FP check failed: max_err={max_err:.4e}"

    def test_freeze_original(self):
        model = self._make_model(4)
        config = MultiAxisPadConfig(depth_expansion=2, freeze_original=True)
        MultiAxisPadExpander().expand(model, config)
        frozen = [p for p in model.parameters() if not p.requires_grad]
        assert len(frozen) > 0
