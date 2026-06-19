"""Tests for DenseToMoEExpander and ExpertCloneExpander."""

from __future__ import annotations

from llm_grow.expanders.sparse.dense_to_moe import (
    DenseToMoEConfig,
    DenseToMoEExpander,
    MoELayer,
)
from llm_grow.expanders.sparse.expert_clone import (
    ExpertCloneConfig,
    ExpertCloneExpander,
    ExpertSelectionStrategy,
)
from tests.conftest import FakeModel, FakeMoEModel


class TestDenseToMoEExpander:
    def _make_model(self, num_layers=4):
        return FakeModel(num_layers=num_layers, d=32)

    def test_replaces_ffn_with_moe(self):
        model = self._make_model(4)
        config = DenseToMoEConfig(num_experts=4, top_k=2)
        expanded = DenseToMoEExpander().expand(model, config)
        moe_count = sum(
            1 for _, m in expanded.named_modules() if isinstance(m, MoELayer)
        )
        assert moe_count == 4

    def test_expert_count(self):
        model = self._make_model(4)
        config = DenseToMoEConfig(num_experts=8, top_k=2)
        expanded = DenseToMoEExpander().expand(model, config)
        for _, m in expanded.named_modules():
            if isinstance(m, MoELayer):
                assert len(m.experts) == 8
                assert m.num_experts == 8
                break

    def test_verify_returns_false(self):
        model = self._make_model(4)
        result = DenseToMoEExpander().verify(model, model)
        assert result is False


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

    def test_verify_returns_false(self):
        model = self._make_moe_model()
        result = ExpertCloneExpander().verify(model, model)
        assert result is False
