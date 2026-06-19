"""Tests for identity initializer."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from llm_grow.initializers.identity import is_identity_block, zero_output_projections
from tests.conftest import FakeDecoderLayer, FakeMLP


class TestIdentityInitializer:
    def test_zero_output_projections_makes_identity(self):
        block = FakeDecoderLayer(d=32)
        block = zero_output_projections(block)

        x = torch.randn(2, 16, 32)
        out = block(x)
        assert torch.allclose(out, x, atol=1e-5), (
            f"Expected identity output, got max diff {(out - x).abs().max():.4e}"
        )

    def test_is_identity_block_detection(self):
        block = FakeDecoderLayer(d=32)
        assert not is_identity_block(block)
        zero_output_projections(block)
        assert is_identity_block(block)

    def test_zero_proj_weights_are_zero(self):
        block = FakeDecoderLayer(d=32)
        zero_output_projections(block)
        assert block.self_attn.o_proj.weight.abs().max().item() == 0.0
        assert block.mlp.down_proj.weight.abs().max().item() == 0.0

    def test_other_weights_unchanged(self):
        block = FakeDecoderLayer(d=32)
        orig_q = block.self_attn.q_proj.weight.clone()
        zero_output_projections(block)
        assert torch.allclose(block.self_attn.q_proj.weight, orig_q)


class TestSymmetryBreak:
    def test_add_noise_changes_params(self):
        from llm_grow.initializers.symmetry_break import add_noise_to_experts

        mlp = FakeMLP(d=16)
        experts = nn.ModuleList([mlp, copy.deepcopy(mlp)])
        orig_w = experts[1].down_proj.weight.clone()
        add_noise_to_experts(experts, std=0.1, skip_first=True)
        assert not torch.allclose(experts[1].down_proj.weight, orig_w)
        assert torch.allclose(experts[0].down_proj.weight, orig_w)

    def test_drop_upcycling_zeroes_some_params(self):
        from llm_grow.initializers.symmetry_break import drop_upcycling

        mlp = FakeMLP(d=32)
        experts = nn.ModuleList([mlp, copy.deepcopy(mlp)])
        drop_upcycling(experts, drop_ratio=0.5, skip_first=True)
        w = experts[1].down_proj.weight
        zero_ratio = (w == 0).float().mean().item()
        assert zero_ratio > 0.3, f"Expected >30% zeros, got {zero_ratio:.1%}"
