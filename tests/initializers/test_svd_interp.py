"""Tests for SVD interpolation initializer."""

from __future__ import annotations

import torch
import torch.nn as nn

from llm_grow.initializers import svd_interp
from llm_grow.initializers.svd_interp import (
    LayerPredictor,
    predict_layer,
    svd_features,
    train_predictor,
)
from tests.conftest import FakeDecoderLayer


class TestSvdFeatures:
    def test_returns_correct_shape_2d(self):
        weight = torch.randn(16, 16)
        rank = 8
        feat = svd_features(weight, rank=rank)
        expected_len = rank * weight.shape[1]
        assert feat.shape == (expected_len,), (
            f"Expected shape ({expected_len},), got {feat.shape}"
        )

    def test_rank_clamped_to_min_dim(self):
        weight = torch.randn(4, 16)
        feat = svd_features(weight, rank=64)
        # rank is clamped to min(rank, S.shape[0]) = min(64, 4) = 4
        expected_len = 4 * 16
        assert feat.shape == (expected_len,), (
            f"Expected shape ({expected_len},), got {feat.shape}"
        )


class TestLayerPredictor:
    def test_forward_returns_correct_shape(self):
        feat_dim = 128
        param_numel = 256
        predictor = LayerPredictor(feat_dim, param_numel, hidden=64)
        feat_a = torch.randn(feat_dim)
        feat_b = torch.randn(feat_dim)
        out = predictor(feat_a, feat_b)
        assert out.shape == (param_numel,), (
            f"Expected shape ({param_numel},), got {out.shape}"
        )


class TestTrainPredictor:
    def test_trains_on_tiny_model(self):
        d = 16
        num_layers = 4
        layers = nn.ModuleList([FakeDecoderLayer(d) for _ in range(num_layers)])
        predictors = train_predictor(
            layers, svd_rank=4, predictor_hidden=32, lr=1e-3, steps=10, device="cpu"
        )
        assert isinstance(predictors, dict)
        assert len(predictors) > 0, "Expected at least one predictor"
        # Every predictor value should be a LayerPredictor
        for name, pred in predictors.items():
            assert isinstance(pred, LayerPredictor), (
                f"Predictor for '{name}' is not a LayerPredictor"
            )

    def test_svd_features_are_cached(self, monkeypatch):
        """SVD features should be computed once per parameter, not per step."""
        d = 16
        num_layers = 4
        layers = nn.ModuleList([FakeDecoderLayer(d) for _ in range(num_layers)])

        call_count = 0
        original_fn = svd_interp._layer_svd_features

        def counting_fn(layer, param_name, rank):
            nonlocal call_count
            call_count += 1
            return original_fn(layer, param_name, rank)

        monkeypatch.setattr(svd_interp, "_layer_svd_features", counting_fn)

        # Each layer has 4 2D params; features computed once per param name per layer.
        predictors = train_predictor(
            layers, svd_rank=4, predictor_hidden=16, lr=1e-3, steps=5, device="cpu"
        )
        assert predictors
        # Expected calls = num_layers * num_2d_params
        expected_calls = num_layers * sum(
            1 for _, p in layers[0].named_parameters() if p.dim() >= 2
        )
        assert call_count == expected_calls, (
            f"Expected {expected_calls} SVD calls, got {call_count}"
        )


class TestPredictLayer:
    def test_produces_correct_param_shapes(self):
        d = 16
        layers = nn.ModuleList([FakeDecoderLayer(d) for _ in range(4)])
        predictors = train_predictor(
            layers, svd_rank=4, predictor_hidden=32, lr=1e-3, steps=5, device="cpu"
        )
        layer_a = layers[0]
        layer_b = layers[1]
        new_layer = predict_layer(layer_a, layer_b, predictors, svd_rank=4)

        # Every parameter in the new layer should match the original shape
        orig_params = dict(layer_a.named_parameters())
        for name, param in new_layer.named_parameters():
            assert param.shape == orig_params[name].shape, (
                f"Parameter '{name}' shape mismatch: "
                f"{param.shape} vs {orig_params[name].shape}"
            )
