"""Tests for ExpertCloneExpander."""

from __future__ import annotations

from llm_grow.expanders.sparse.expert_clone import (
    ExpertCloneConfig,
    ExpertCloneExpander,
    ExpertSelectionStrategy,
)
from tests.conftest import FakeMoEModel


class TestExpertCloneExpander:
    def _make_moe_model(self):
        return FakeMoEModel(num_layers=4, d=32, num_experts=4)

    def test_expert_count_doubles(self):
        model = self._make_moe_model()
        config = ExpertCloneConfig(
            expand_factor=2,
            selection_strategy=ExpertSelectionStrategy.UNIFORM,
            moe_layer_cls_name="FakeMoELayer",
        )
        ExpertCloneExpander().expand(model, config)
        for layer in model.layers:
            assert len(layer.experts) == 8

    def test_router_weight_expands(self):
        model = self._make_moe_model()
        config = ExpertCloneConfig(
            expand_factor=2,
            selection_strategy=ExpertSelectionStrategy.UNIFORM,
            moe_layer_cls_name="FakeMoELayer",
        )
        ExpertCloneExpander().expand(model, config)
        for layer in model.layers:
            assert layer.router.weight.shape[0] == 8

    def test_utility_strategy(self):
        model = self._make_moe_model()
        config = ExpertCloneConfig(
            expand_factor=2,
            selection_strategy=ExpertSelectionStrategy.UTILITY,
            moe_layer_cls_name="FakeMoELayer",
        )
        ExpertCloneExpander().expand(model, config)
        for layer in model.layers:
            assert len(layer.experts) == 8

    def test_verify_skips_strict_fp(self):
        model = self._make_moe_model()
        result = ExpertCloneExpander().verify(model, model)
        assert result is True
