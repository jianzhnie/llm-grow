"""Tests for llm_grow.safetensor.auto auto_expand entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as safetensors_save

from llm_grow.safetensor.auto import auto_expand


def _make_dense_model_dir(base: Path, num_layers: int = 4) -> Path:
    """Create a tiny dense model directory with real safetensors file."""
    tmp = base / "src"
    tmp.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": num_layers,
        "hidden_size": 32,
        "intermediate_size": 64,
        "vocab_size": 128,
    }
    (tmp / "config.json").write_text(json.dumps(config))

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(128, 32),
        "model.norm.weight": torch.randn(32),
        "lm_head.weight": torch.randn(128, 32),
    }
    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        layer_tensors = {
            f"{prefix}input_layernorm.weight": torch.randn(32),
            f"{prefix}self_attn.q_proj.weight": torch.randn(32, 32),
            f"{prefix}self_attn.o_proj.weight": torch.randn(32, 32),
            f"{prefix}mlp.gate_proj.weight": torch.randn(64, 32),
            f"{prefix}mlp.up_proj.weight": torch.randn(64, 32),
            f"{prefix}mlp.down_proj.weight": torch.randn(32, 64),
        }
        tensors.update(layer_tensors)

    safetensors_save(tensors, str(tmp / "model.safetensors"))
    return tmp


class TestAutoExpandDense:
    def test_depth_dry_run(self, tmp_path):
        src_dir = _make_dense_model_dir(tmp_path, num_layers=4)
        dst_dir = tmp_path / "dst"
        auto_expand(
            src_dir=src_dir,
            dst_dir=dst_dir,
            method="depth",
            num_new_layers=2,
            dry_run=True,
            verbose=False,
        )

    def test_width_dry_run(self, tmp_path):
        src_dir = _make_dense_model_dir(tmp_path, num_layers=4)
        dst_dir = tmp_path / "dst"
        auto_expand(
            src_dir=src_dir,
            dst_dir=dst_dir,
            method="width",
            ffn_size_expansion=32,
            dry_run=True,
            verbose=False,
        )

    def test_expert_on_dense_raises(self, tmp_path):
        src_dir = _make_dense_model_dir(tmp_path, num_layers=4)
        dst_dir = tmp_path / "dst"
        with pytest.raises(ValueError, match="method='expert' requires a MoE model"):
            auto_expand(
                src_dir=src_dir,
                dst_dir=dst_dir,
                method="expert",
                expand_factor=2,
                dry_run=True,
                verbose=False,
            )

    def test_unknown_method_raises(self, tmp_path):
        src_dir = _make_dense_model_dir(tmp_path, num_layers=4)
        dst_dir = tmp_path / "dst"
        with pytest.raises(ValueError, match="No expander registered"):
            auto_expand(
                src_dir=src_dir,
                dst_dir=dst_dir,
                method="not_a_method",
                dry_run=True,
                verbose=False,
            )


def _make_moe_model_dir(
    base: Path, num_layers: int = 4, num_experts: int = 4
) -> Path:
    """Create a tiny MoE model directory with real safetensors file."""
    tmp = base / "src"
    tmp.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_hidden_layers": num_layers,
        "hidden_size": 32,
        "moe_intermediate_size": 64,
        "num_experts": num_experts,
        "num_experts_per_tok": 2,
    }
    (tmp / "config.json").write_text(json.dumps(config))

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(128, 32),
        "model.norm.weight": torch.randn(32),
        "lm_head.weight": torch.randn(128, 32),
    }
    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        layer_tensors = {
            f"{prefix}input_layernorm.weight": torch.randn(32),
            f"{prefix}self_attn.q_proj.weight": torch.randn(32, 32),
            f"{prefix}self_attn.o_proj.weight": torch.randn(32, 32),
            f"{prefix}mlp.gate.weight": torch.randn(num_experts, 32),
        }
        for e in range(num_experts):
            layer_tensors[f"{prefix}mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                64, 32
            )
            layer_tensors[f"{prefix}mlp.experts.{e}.down_proj.weight"] = torch.randn(
                32, 64
            )
        tensors.update(layer_tensors)

    safetensors_save(tensors, str(tmp / "model.safetensors"))
    return tmp


class TestAutoExpandMoE:
    def test_expert_dry_run(self, tmp_path):
        src_dir = _make_moe_model_dir(tmp_path, num_layers=4, num_experts=4)
        dst_dir = tmp_path / "dst"
        auto_expand(
            src_dir=src_dir,
            dst_dir=dst_dir,
            method="expert",
            expand_factor=2,
            dry_run=True,
            verbose=False,
        )

    def test_depth_dry_run(self, tmp_path):
        src_dir = _make_moe_model_dir(tmp_path, num_layers=4, num_experts=4)
        dst_dir = tmp_path / "dst"
        auto_expand(
            src_dir=src_dir,
            dst_dir=dst_dir,
            method="depth",
            num_new_layers=2,
            dry_run=True,
            verbose=False,
        )
