"""Tests for Net2NetExpander."""

from __future__ import annotations

import pytest
import torch

from llm_grow.expanders.width.net2net import Net2NetConfig, Net2NetExpander
from tests.conftest import FakeModel


class TestWider:
    """Tests for Net2NetExpander.wider()."""

    def setup_method(self):
        self.expander = Net2NetExpander()
        torch.manual_seed(42)
        self.w_in = torch.randn(8, 4)   # (out_features=8, in_features=4)
        self.w_out = torch.randn(6, 8)  # (next_out, out_features=8)

    def test_output_shapes(self):
        new_width = 16
        w_in_new, w_out_new = self.expander.wider(
            self.w_in, self.w_out, new_width, add_noise=False
        )
        assert w_in_new.shape == (new_width, 4)
        assert w_out_new.shape == (6, new_width)

    def test_function_preserving(self):
        new_width = 16
        w_in_new, w_out_new = self.expander.wider(
            self.w_in, self.w_out, new_width, add_noise=False
        )
        original = self.w_out @ self.w_in
        expanded = w_out_new @ w_in_new
        torch.testing.assert_close(expanded, original, atol=1e-5, rtol=1e-5)

    def test_no_expansion_when_width_equal(self):
        w_in_new, w_out_new = self.expander.wider(
            self.w_in, self.w_out, new_width=8, add_noise=False
        )
        assert torch.equal(w_in_new, self.w_in)
        assert torch.equal(w_out_new, self.w_out)

    def test_no_expansion_when_width_smaller(self):
        w_in_new, w_out_new = self.expander.wider(
            self.w_in, self.w_out, new_width=4, add_noise=False
        )
        assert torch.equal(w_in_new, self.w_in)
        assert torch.equal(w_out_new, self.w_out)


class TestExpand:
    """Tests for Net2NetExpander.expand()."""

    def test_expand_raises_not_implemented(self):
        expander = Net2NetExpander()
        model = FakeModel(num_layers=2, d=32)
        config = Net2NetConfig()
        with pytest.raises(NotImplementedError):
            expander.expand(model, config)
