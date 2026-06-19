"""Tests for llm_grow.eval verification utilities."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file as safetensors_save

from llm_grow.eval.fp_verifier import verify_fp
from llm_grow.eval.structural import StructuralVerifier
from tests.conftest import FakeModel


class TestVerifyFp:
    def test_identical_models_pass(self):
        model = FakeModel(num_layers=4, d=16)
        identical = copy.deepcopy(model)
        assert verify_fp(model, identical, num_samples=2, seq_len=8, atol=1e-4)

    def test_different_models_fail(self):
        model_a = FakeModel(num_layers=4, d=16)
        model_b = FakeModel(num_layers=4, d=16)
        # Reset parameters to make outputs differ
        for p in model_b.parameters():
            torch.nn.init.normal_(p)
        assert not verify_fp(
            model_a, model_b, num_samples=2, seq_len=8, atol=1e-4, verbose=False
        )


def _make_fake_safetensor_dir(base_dir: Path, num_layers: int = 4) -> Path:
    tmp = base_dir / f"model_{num_layers}L"
    tmp.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": num_layers,
        "hidden_size": 32,
    }
    (tmp / "config.json").write_text(json.dumps(config))
    tensors: dict[str, torch.Tensor] = {}
    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        tensors[f"{prefix}self_attn.o_proj.weight"] = torch.randn(32, 32)
        tensors[f"{prefix}mlp.gate_proj.weight"] = torch.randn(64, 32)
        tensors[f"{prefix}mlp.down_proj.weight"] = torch.randn(32, 64)
    safetensors_save(tensors, str(tmp / "model.safetensors"))
    return tmp


class TestStructuralVerifier:
    def test_runs_all_checks(self, tmp_path):
        src_dir = _make_fake_safetensor_dir(tmp_path, num_layers=4)
        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        verifier = StructuralVerifier(src_dir, dst_dir)
        results = verifier.run_all()
        assert "config" in results
        assert "tensor_counts" in results
        assert "weights_preserved" in results
        assert "identity_zeroed" in results
        assert all(results.values())

    def test_tensor_counts_match_for_same_architecture(self, tmp_path):
        src_dir = _make_fake_safetensor_dir(tmp_path, num_layers=4)
        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        verifier = StructuralVerifier(src_dir, dst_dir)
        assert verifier.check_tensor_counts()

    def test_tensor_counts_for_different_depth(self, tmp_path):
        src_dir = _make_fake_safetensor_dir(tmp_path / "src", num_layers=4)
        dst_dir = _make_fake_safetensor_dir(tmp_path / "dst", num_layers=6)
        verifier = StructuralVerifier(src_dir, dst_dir)
        # The structural verifier accounts for depth changes in its expected count.
        result = verifier.check_tensor_counts()
        assert result is True
