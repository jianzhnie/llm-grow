"""Tests for advanced initializers: SVD interpolation and cluster-aware."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from llm_grow.initializers.svd_interp import (
    LayerPredictor,
    predict_layer,
    svd_features,
    train_predictor,
)
from llm_grow.initializers.symmetry_break import (
    cluster_aware_upcycling,
    router_noise_init,
)
from tests.conftest import FakeDecoderLayer, FakeMLP


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


class TestRouterNoiseInit:
    def test_changes_router_weights(self):
        router = nn.Linear(16, 4, bias=False)
        orig_weight = router.weight.clone()
        router_noise_init(router, std=0.01)
        assert not torch.allclose(router.weight, orig_weight), (
            "Router weights should change after noise init"
        )


class TestClusterAwareUpcycling:
    def test_changes_expert_weights(self):
        d = 16
        mlp = FakeMLP(d=d)
        experts = nn.ModuleList([copy.deepcopy(mlp) for _ in range(4)])
        orig_w = experts[1].down_proj.weight.clone()
        cluster_assignments = [0, 1, 2, 3]
        cluster_aware_upcycling(
            experts,
            cluster_assignments,
            skip_first=True,
            drop_ratio=0.1,
            noise_std=0.01,
        )
        assert not torch.allclose(experts[1].down_proj.weight, orig_w), (
            "Expert weights should change after cluster-aware upcycling"
        )

    def test_respects_skip_first(self):
        d = 16
        mlp = FakeMLP(d=d)
        experts = nn.ModuleList([copy.deepcopy(mlp) for _ in range(4)])
        orig_first = experts[0].down_proj.weight.clone()
        cluster_assignments = [0, 1, 2, 3]
        cluster_aware_upcycling(
            experts,
            cluster_assignments,
            skip_first=True,
            drop_ratio=0.1,
            noise_std=0.01,
        )
        assert torch.allclose(experts[0].down_proj.weight, orig_first), (
            "First expert should be unchanged when skip_first=True"
        )

    def test_different_clusters_produce_different_masks(self):
        d = 16
        mlp = FakeMLP(d=d)
        experts = nn.ModuleList([copy.deepcopy(mlp) for _ in range(3)])
        # Assign experts 1 and 2 to different clusters
        cluster_assignments = [0, 1, 2]
        cluster_aware_upcycling(
            experts, cluster_assignments, skip_first=True, drop_ratio=0.3, noise_std=0.0
        )
        # With noise_std=0 and different clusters, the zero masks should differ
        mask_1 = experts[1].down_proj.weight == 0
        mask_2 = experts[2].down_proj.weight == 0
        assert not torch.equal(mask_1, mask_2), (
            "Different cluster assignments should produce different drop masks"
        )
