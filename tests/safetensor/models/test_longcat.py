"""Tests for llm_grow.safetensor.models.longcat expanders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as safetensors_save

from llm_grow.safetensor.models.longcat import (
    LongcatDepthConfig,
    LongcatDepthExpander,
    LongcatExpertCloneConfig,
    LongcatExpertCloneExpander,
)


def _make_longcat_dir(
    tmp_path: Path,
    num_layers: int = 4,
    num_experts: int = 4,
    hidden: int = 32,
) -> Path:
    """Create a tiny LongCat-Flash safetensor model directory."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "longcat_flash",
        "architectures": ["LongcatFlashForCausalLM"],
        "num_hidden_layers": num_layers,
        "hidden_size": hidden,
        "n_routed_experts": num_experts,
        "moe_topk": 2,
        "zero_expert_num": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(64, hidden),
        "model.norm.weight": torch.randn(hidden),
        "lm_head.weight": torch.randn(64, hidden),
    }
    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        tensors[f"{prefix}input_layernorm.0.weight"] = torch.randn(hidden)
        tensors[f"{prefix}input_layernorm.1.weight"] = torch.randn(hidden)
        tensors[f"{prefix}self_attn.0.o_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"{prefix}self_attn.1.o_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"{prefix}mlps.0.down_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"{prefix}mlps.1.down_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"{prefix}mlp.router.classifier.weight"] = torch.randn(
            num_experts, hidden
        )
        tensors[f"{prefix}mlp.router.e_score_correction_bias"] = torch.randn(
            num_experts
        )
        for e in range(num_experts):
            tensors[f"{prefix}mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                hidden, hidden
            )
            tensors[f"{prefix}mlp.experts.{e}.down_proj.weight"] = torch.randn(
                hidden, hidden
            )

    safetensors_save(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


class TestLongcatExpertClone:
    def test_dry_run(self, tmp_path):
        src_dir = _make_longcat_dir(tmp_path / "src", num_layers=2, num_experts=4)
        cfg = LongcatExpertCloneConfig(expand_factor=2)
        expander = LongcatExpertCloneExpander(cfg)
        plan = expander.dry_run(str(src_dir))
        assert plan is not None
        assert plan.config_patches["n_routed_experts"] == 8

    def test_expand_factor_3_raises(self, tmp_path):
        src_dir = _make_longcat_dir(tmp_path / "src", num_layers=2, num_experts=4)
        cfg = LongcatExpertCloneConfig(expand_factor=3)
        expander = LongcatExpertCloneExpander(cfg)
        with pytest.raises(NotImplementedError, match="expand_factor=3"):
            expander.dry_run(str(src_dir))

    def test_expert_keys_duplicated(self, tmp_path):
        src_dir = _make_longcat_dir(tmp_path / "src", num_layers=2, num_experts=2)
        cfg = LongcatExpertCloneConfig(expand_factor=2)
        expander = LongcatExpertCloneExpander(cfg)
        plan = expander.dry_run(str(src_dir))
        cloned_keys = [k for k in plan.recipes if "experts.2." in k]
        assert len(cloned_keys) > 0


class TestLongcatDepthExpander:
    def test_dry_run_inserts_layers(self, tmp_path):
        src_dir = _make_longcat_dir(tmp_path / "src", num_layers=4, num_experts=2)
        cfg = LongcatDepthConfig(num_new_layers=2)
        expander = LongcatDepthExpander(cfg)
        plan = expander.dry_run(str(src_dir))
        assert plan.new_num_hidden_layers == 6

    def test_identity_blocks_zero_dual_attn(self, tmp_path):
        src_dir = _make_longcat_dir(tmp_path / "src", num_layers=4, num_experts=2)
        cfg = LongcatDepthConfig(num_new_layers=2)
        expander = LongcatDepthExpander(cfg)
        plan = expander.dry_run(str(src_dir))

        attn0_keys = [
            k for k in plan.recipes if k.endswith("self_attn.0.o_proj.weight")
        ]
        assert len(attn0_keys) == 6
        zeroed = [k for k in attn0_keys if plan.recipes[k].zero_out]
        assert len(zeroed) == 2
