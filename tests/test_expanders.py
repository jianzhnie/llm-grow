"""Tests for IdentityGraftExpander and OverlapSplitExpander."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from llm_grow.expanders.depth.identity_graft import (
    IdentityGraftConfig,
    IdentityGraftExpander,
)
from llm_grow.expanders.depth.overlap_split import (
    OverlapSplitConfig,
    OverlapSplitExpander,
)

# ---------------------------------------------------------------------------
# Minimal Transformer-like model for unit testing
# ---------------------------------------------------------------------------


class FakeMLP(nn.Module):
    def __init__(self, d=32, mid=64):
        super().__init__()
        self.gate_proj = nn.Linear(d, mid, bias=False)
        self.up_proj = nn.Linear(d, mid, bias=False)
        self.down_proj = nn.Linear(mid, d, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class FakeAttn(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class FakeDecoderLayer(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.self_attn = FakeAttn(d)
        self.mlp = FakeMLP(d)

    def forward(self, x):
        return x + self.self_attn(x) + self.mlp(x)


class FakeConfig:
    num_hidden_layers = 8
    vocab_size = 256
    hidden_size = 32


class FakeModel(nn.Module):
    def __init__(self, num_layers=8, d=32):
        super().__init__()
        self.config = FakeConfig()
        self.config.num_hidden_layers = num_layers
        self.embed = nn.Embedding(256, d)
        self.layers = nn.ModuleList([FakeDecoderLayer(d) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d, 256, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)

        class Out:
            pass

        out = Out()
        out.logits = self.lm_head(x)
        return out


# ---------------------------------------------------------------------------
# LLaMA-Pro tests
# ---------------------------------------------------------------------------


class TestIdentityGraftExpander:
    def _make_model(self, num_layers=8):
        return FakeModel(num_layers=num_layers, d=32)

    def test_layer_count_increases(self):
        model = self._make_model(8)
        config = IdentityGraftConfig(num_new_blocks=4, insert_strategy="uniform")
        IdentityGraftExpander().expand(model, config)
        assert len(model.layers) == 12

    def test_function_preserving(self):
        model = self._make_model(8)
        original = copy.deepcopy(model)
        config = IdentityGraftConfig(
            num_new_blocks=4, insert_strategy="uniform", freeze_original=False
        )
        expanded = IdentityGraftExpander().expand(model, config)

        input_ids = torch.randint(0, 256, (2, 16))
        original.eval()
        expanded.eval()
        with torch.no_grad():
            out_orig = original(input_ids).logits
            out_exp = expanded(input_ids).logits

        max_err = (out_orig - out_exp).abs().max().item()
        assert max_err < 1e-4, f"FP check failed: max_err={max_err:.4e}"

    def test_num_hidden_layers_updated(self):
        model = self._make_model(8)
        IdentityGraftExpander().expand(model, IdentityGraftConfig(num_new_blocks=2))
        assert model.config.num_hidden_layers == 10

    def test_freeze_original_works(self):
        model = self._make_model(8)
        IdentityGraftExpander().expand(
            model, IdentityGraftConfig(num_new_blocks=2, freeze_original=True)
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        frozen = [p for p in model.parameters() if not p.requires_grad]
        assert len(trainable) > 0
        assert len(frozen) > 0


# ---------------------------------------------------------------------------
# SOLAR DUS tests
# ---------------------------------------------------------------------------


class TestOverlapSplitExpander:
    def test_layer_count(self):
        model = FakeModel(num_layers=8)
        config = OverlapSplitConfig(num_overlap=2)
        OverlapSplitExpander().expand(model, config)
        assert len(model.layers) == 12  # 2*(8-2) = 12

    def test_verify_returns_false(self):
        model = FakeModel(num_layers=8)
        original = copy.deepcopy(model)
        expanded = copy.deepcopy(model)
        result = OverlapSplitExpander().verify(original, expanded)
        assert result is False
