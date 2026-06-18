"""Shared test fixtures for llm-grow tests."""

from __future__ import annotations

import torch.nn as nn


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
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
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
    intermediate_size = 64


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


class FakeMoELayer(nn.Module):
    """Fake MoE layer for testing expert upcycling."""

    def __init__(self, d=32, mid=64, num_experts=4, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([FakeMLP(d, mid) for _ in range(num_experts)])
        self.router = nn.Linear(d, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        return self.experts[0](x)


class FakeMoEModel(nn.Module):
    """Fake model with MoE layers."""

    def __init__(self, num_layers=4, d=32, num_experts=4):
        super().__init__()
        self.config = FakeConfig()
        self.config.num_hidden_layers = num_layers
        self.embed = nn.Embedding(256, d)
        self.layers = nn.ModuleList(
            [FakeMoELayer(d, 64, num_experts) for _ in range(num_layers)]
        )
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
