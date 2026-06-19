"""Tests for llm_grow.training.freeze."""

from __future__ import annotations

import pytest
import torch.nn as nn

from llm_grow.training.freeze import (
    freeze_layers_by_index,
    freeze_original_layers,
    mark_new_params,
    report_trainable,
    snapshot_param_ids,
    unfreeze_all,
)
from tests.conftest import FakeModel


class TestFreezeOriginalLayers:
    def _make_model(self):
        return FakeModel(num_layers=4, d=32)

    def test_freezes_unmarked_params(self):
        model = self._make_model()
        # Mark one layer as new growth
        for p in model.layers[0].parameters():
            p._is_new_growth = True
        frozen_count = freeze_original_layers(model)
        # All params without _is_new_growth should be frozen
        for p in model.layers[1].parameters():
            assert not p.requires_grad
        # New growth params should remain trainable
        for p in model.layers[0].parameters():
            assert p.requires_grad
        assert frozen_count > 0

    def test_returns_correct_numel(self):
        model = self._make_model()
        total = sum(p.numel() for p in model.parameters())
        frozen_count = freeze_original_layers(model)
        assert frozen_count == total


class TestMarkNewParams:
    def test_marks_new_parameters(self):
        model = FakeModel(num_layers=4, d=32)
        original_ids = snapshot_param_ids(model)
        # Simulate expansion by adding a new layer
        model.layers.append(nn.Linear(32, 32))
        count = mark_new_params(model, original_ids)
        new_layer = model.layers[-1]
        assert getattr(new_layer.weight, "_is_new_growth", False) is True
        assert getattr(new_layer.bias, "_is_new_growth", False) is True
        expected = new_layer.weight.numel() + new_layer.bias.numel()
        assert count == expected


class TestSnapshotParamIds:
    def test_returns_set_of_ids(self):
        model = FakeModel(num_layers=2, d=32)
        ids = snapshot_param_ids(model)
        assert isinstance(ids, set)
        assert len(ids) == len(list(model.parameters()))


class TestUnfreezeAll:
    def test_unfreezes_all_params(self):
        model = FakeModel(num_layers=4, d=32)
        # Freeze everything first
        for p in model.parameters():
            p.requires_grad_(False)
        count = unfreeze_all(model)
        for p in model.parameters():
            assert p.requires_grad
        assert count == sum(p.numel() for p in model.parameters())


class TestReportTrainable:
    def test_all_trainable(self):
        model = FakeModel(num_layers=2, d=32)
        report = report_trainable(model)
        total = sum(p.numel() for p in model.parameters())
        assert report["trainable"] == total
        assert report["frozen"] == 0
        assert report["total"] == total

    def test_mixed_trainable(self):
        model = FakeModel(num_layers=2, d=32)
        # Freeze embed layer
        for p in model.embed.parameters():
            p.requires_grad_(False)
        report = report_trainable(model)
        assert report["frozen"] == sum(p.numel() for p in model.embed.parameters())
        assert report["trainable"] + report["frozen"] == report["total"]


class TestFreezeLayersByIndex:
    def test_freezes_specific_layers(self):
        model = FakeModel(num_layers=4, d=32)
        freeze_layers_by_index(model, [0, 2], layer_attr="layers")
        for p in model.layers[0].parameters():
            assert not p.requires_grad
        for p in model.layers[1].parameters():
            assert p.requires_grad
        for p in model.layers[2].parameters():
            assert not p.requires_grad
        for p in model.layers[3].parameters():
            assert p.requires_grad

    def test_raises_on_bad_attr(self):
        model = FakeModel(num_layers=4, d=32)
        with pytest.raises(AttributeError):
            freeze_layers_by_index(model, [0], layer_attr="nonexistent.attr")
