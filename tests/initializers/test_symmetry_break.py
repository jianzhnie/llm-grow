"""Tests for symmetry-breaking initializers."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from llm_grow.initializers.symmetry_break import (
    cluster_aware_upcycling,
    router_noise_init,
)
from tests.conftest import FakeMLP


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
