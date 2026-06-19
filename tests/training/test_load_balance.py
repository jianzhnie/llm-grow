"""Tests for llm_grow.training.load_balance."""

from __future__ import annotations

import math

import torch

from llm_grow.training.load_balance import (
    combined_moe_loss,
    load_balance_loss,
    z_loss,
)


class TestLoadBalanceLoss:
    def test_uniform_routing(self):
        num_experts = 4
        # Uniform logits => uniform routing => loss should be small
        logits = torch.zeros(16, num_experts)
        loss = load_balance_loss(logits, num_experts, top_k=2, coeff=1e-2)
        assert loss.item() >= 0

    def test_skewed_routing_higher_loss(self):
        num_experts = 4
        uniform_logits = torch.zeros(16, num_experts)
        skewed_logits = torch.zeros(16, num_experts)
        skewed_logits[:, 0] = 100.0  # Everything routes to expert 0
        loss_uniform = load_balance_loss(
            uniform_logits, num_experts, top_k=1, coeff=1e-2
        )
        loss_skewed = load_balance_loss(skewed_logits, num_experts, top_k=1, coeff=1e-2)
        assert loss_skewed.item() > loss_uniform.item()

    def test_scalar_output(self):
        logits = torch.randn(8, 4)
        loss = load_balance_loss(logits, num_experts=4)
        assert loss.shape == ()


class TestZLoss:
    def test_zero_logits(self):
        logits = torch.zeros(8, 4)
        loss = z_loss(logits, coeff=1e-3)
        # logsumexp(zeros) = log(4), z_loss = coeff * log(4)^2
        expected = 1e-3 * (math.log(4) ** 2)
        assert abs(loss.item() - expected) < 1e-5

    def test_positive_loss(self):
        logits = torch.randn(16, 8)
        loss = z_loss(logits, coeff=1e-3)
        assert loss.item() > 0

    def test_scalar_output(self):
        logits = torch.randn(8, 4)
        loss = z_loss(logits)
        assert loss.shape == ()


class TestCombinedMoeLoss:
    def test_combined_includes_all(self):
        lm_loss = torch.tensor(2.0)
        router_logits_list = [torch.randn(8, 4), torch.randn(8, 4)]
        total = combined_moe_loss(
            lm_loss,
            router_logits_list,
            num_experts=4,
            top_k=2,
            balance_coeff=1e-2,
            z_coeff=1e-3,
        )
        assert total.item() > lm_loss.item()

    def test_empty_router_list(self):
        lm_loss = torch.tensor(3.0)
        total = combined_moe_loss(lm_loss, [], num_experts=4)
        assert total.item() == lm_loss.item()

    def test_scalar_output(self):
        lm_loss = torch.tensor(1.0)
        total = combined_moe_loss(lm_loss, [torch.randn(4, 2)], num_experts=2)
        assert total.shape == ()
