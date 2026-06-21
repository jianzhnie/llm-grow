"""Tests for the SafetensorModelInspector abstraction."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file as safetensors_save

from llm_grow.safetensor.inspector import SafetensorModelInspector


def _make_tiny_model(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 32,
        "intermediate_size": 64,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(128, 32),
        "model.norm.weight": torch.randn(32),
        "lm_head.weight": torch.randn(128, 32),
    }
    for i in range(2):
        prefix = f"model.layers.{i}."
        tensors[f"{prefix}input_layernorm.weight"] = torch.randn(32)
        tensors[f"{prefix}self_attn.q_proj.weight"] = torch.randn(32, 32)
        tensors[f"{prefix}self_attn.o_proj.weight"] = torch.randn(32, 32)
        tensors[f"{prefix}mlp.gate_proj.weight"] = torch.randn(64, 32)
        tensors[f"{prefix}mlp.up_proj.weight"] = torch.randn(64, 32)
        tensors[f"{prefix}mlp.down_proj.weight"] = torch.randn(32, 64)

    safetensors_save(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


class TestSafetensorModelInspector:
    def test_peek_config(self, tmp_path):
        model_dir = _make_tiny_model(tmp_path / "src")
        inspector = SafetensorModelInspector(model_dir)
        cfg = inspector.peek_config()
        assert cfg["num_hidden_layers"] == 2
        assert cfg["hidden_size"] == 32

    def test_num_hidden_layers(self, tmp_path):
        model_dir = _make_tiny_model(tmp_path / "src")
        inspector = SafetensorModelInspector(model_dir)
        assert inspector.num_hidden_layers() == 2

    def test_layer_suffixes(self, tmp_path):
        model_dir = _make_tiny_model(tmp_path / "src")
        inspector = SafetensorModelInspector(model_dir)
        suffixes = inspector.layer_suffixes()
        assert "self_attn.o_proj.weight" in suffixes
        assert "mlp.down_proj.weight" in suffixes

    def test_weight_map(self, tmp_path):
        model_dir = _make_tiny_model(tmp_path / "src")
        inspector = SafetensorModelInspector(model_dir)
        wmap = inspector.weight_map()
        assert wmap["model.layers.0.self_attn.o_proj.weight"] == "model.safetensors"

    def test_get_tensor(self, tmp_path):
        model_dir = _make_tiny_model(tmp_path / "src")
        inspector = SafetensorModelInspector(model_dir)
        t = inspector.get_tensor("model.layers.0.self_attn.o_proj.weight")
        assert t.shape == (32, 32)
        inspector.close()

    def test_context_manager_closes_handles(self, tmp_path):
        model_dir = _make_tiny_model(tmp_path / "src")
        with SafetensorModelInspector(model_dir) as inspector:
            _ = inspector.get_tensor("model.layers.0.self_attn.o_proj.weight")
            assert len(inspector._handles) == 1
        assert len(inspector._handles) == 0
