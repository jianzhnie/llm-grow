"""Tests for llm_grow.cli."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as safetensors_save

from llm_grow.cli import main


def _make_model_dir(num_layers: int = 4) -> Path:
    tmp = Path(tempfile.mkdtemp())
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": num_layers,
        "hidden_size": 32,
        "intermediate_size": 64,
        "vocab_size": 128,
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


class TestCliInfo:
    def test_info_command(self, monkeypatch, capsys):
        src_dir = _make_model_dir(num_layers=4)
        monkeypatch.setattr(
            "sys.argv", ["llm-grow", "info", "--src", str(src_dir)]
        )
        main()
        captured = capsys.readouterr()
        assert "dense" in captured.out
        assert "LlamaForCausalLM" in captured.out


class TestCliExpand:
    def test_expand_depth_dry_run(self, monkeypatch):
        src_dir = _make_model_dir(num_layers=4)
        dst_dir = Path(tempfile.mkdtemp())
        monkeypatch.setattr(
            "sys.argv",
            [
                "llm-grow",
                "expand",
                "--src",
                str(src_dir),
                "--dst",
                str(dst_dir),
                "--method",
                "depth",
                "--num-new-layers",
                "2",
                "--dry-run",
                "--quiet",
            ],
        )
        main()

    def test_expand_expert_on_dense_fails(self, monkeypatch):
        src_dir = _make_model_dir(num_layers=4)
        dst_dir = Path(tempfile.mkdtemp())
        monkeypatch.setattr(
            "sys.argv",
            [
                "llm-grow",
                "expand",
                "--src",
                str(src_dir),
                "--dst",
                str(dst_dir),
                "--method",
                "expert",
                "--dry-run",
                "--quiet",
            ],
        )
        with pytest.raises(ValueError, match="method='expert' requires a MoE model"):
            main()


class TestCliVerify:
    def test_verify_command(self, monkeypatch):
        src_dir = _make_model_dir(num_layers=4)
        dst_dir = Path(tempfile.mkdtemp())
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        monkeypatch.setattr(
            "sys.argv",
            ["llm-grow", "verify", "--src", str(src_dir), "--dst", str(dst_dir)],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_verify_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["llm-grow"])
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "llm-grow" in captured.out
