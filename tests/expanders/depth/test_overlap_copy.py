"""Tests for OverlapCopyExpander."""

from __future__ import annotations

import copy

from llm_grow.expanders.depth.overlap_copy import (
    OverlapCopyConfig,
    OverlapCopyExpander,
)
from tests.conftest import FakeModel


class TestOverlapCopyExpander:
    def test_layer_count(self):
        model = FakeModel(num_layers=8)
        config = OverlapCopyConfig(num_overlap=2)
        OverlapCopyExpander().expand(model, config)
        assert len(model.layers) == 12  # 2*(8-2) = 12

    def test_verify_returns_false(self):
        model = FakeModel(num_layers=8)
        original = copy.deepcopy(model)
        expanded = copy.deepcopy(model)
        result = OverlapCopyExpander().verify(original, expanded)
        assert result is False
