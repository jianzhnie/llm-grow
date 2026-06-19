"""Tests for SVDInterpInsertExpander."""

from __future__ import annotations

from llm_grow.expanders.depth.svd_interp_insert import (
    SVDInterpInsertConfig,
    SVDInterpInsertExpander,
)
from tests.conftest import FakeModel


class TestSVDInterpInsertExpander:
    def _make_model(self, num_layers=8):
        return FakeModel(num_layers=num_layers, d=32)

    def test_layer_count_increases(self):
        model = self._make_model(8)
        config = SVDInterpInsertConfig(insert_between=[(1, 2), (4, 5)])
        SVDInterpInsertExpander().expand(model, config)
        assert len(model.layers) == 10

    def test_insert_between_single(self):
        model = self._make_model(8)
        config = SVDInterpInsertConfig(insert_between=[(3, 4)])
        SVDInterpInsertExpander().expand(model, config)
        assert len(model.layers) == 9
