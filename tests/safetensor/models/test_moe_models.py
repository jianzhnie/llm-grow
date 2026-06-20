"""Tests for safetensor/models/ MoE expanders (moe_generic, moe_width)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file as safetensors_save

from llm_grow.safetensor.models.moe_generic import (
    GenericDenseToMoEConfig,
    GenericMoEDepthConfig,
    GenericMoEDepthExpander,
    GenericMoEExpertCloneExpander,
    make_qwen3moe_expert_clone,
    make_qwen3moe_zero_block_insert,
)
from llm_grow.safetensor.models.moe_width import MoEWidthConfig, MoEWidthExpander


def _make_moe_dir(
    num_layers: int = 4, num_experts: int = 4, hidden: int = 32, ffn: int = 64
) -> Path:
    """Create a tiny MoE safetensor model directory."""
    tmp = Path(tempfile.mkdtemp())
    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_hidden_layers": num_layers,
        "hidden_size": hidden,
        "moe_intermediate_size": ffn,
        "num_experts": num_experts,
        "num_experts_per_tok": 2,
    }
    (tmp / "config.json").write_text(json.dumps(config))

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(128, hidden),
        "model.norm.weight": torch.randn(hidden),
        "lm_head.weight": torch.randn(128, hidden),
    }
    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        tensors[f"{prefix}input_layernorm.weight"] = torch.randn(hidden)
        tensors[f"{prefix}self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"{prefix}self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"{prefix}mlp.gate.weight"] = torch.randn(num_experts, hidden)
        for e in range(num_experts):
            tensors[f"{prefix}mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                ffn, hidden
            )
            tensors[f"{prefix}mlp.experts.{e}.up_proj.weight"] = torch.randn(
                ffn, hidden
            )
            tensors[f"{prefix}mlp.experts.{e}.down_proj.weight"] = torch.randn(
                hidden, ffn
            )

    safetensors_save(tensors, str(tmp / "model.safetensors"))
    return tmp


class TestGenericMoEExpertClone:
    def test_dry_run_doubles_experts(self):
        src_dir = _make_moe_dir(num_layers=4, num_experts=4)
        cfg = GenericDenseToMoEConfig(
            expand_factor=2,
            router_weight_suffixes=["mlp.gate.weight"],
        )
        expander = GenericMoEExpertCloneExpander(cfg)
        plan = expander.dry_run(str(src_dir))
        assert plan is not None

    def test_actual_write_produces_output(self):
        src_dir = _make_moe_dir(num_layers=2, num_experts=2)
        cfg = GenericDenseToMoEConfig(
            expand_factor=2,
            router_weight_suffixes=["mlp.gate.weight"],
        )
        expander = GenericMoEExpertCloneExpander(cfg)
        plan = expander.dry_run(str(src_dir))
        assert len(plan.recipes) > 0
        expert_keys = [k for k in plan.recipes if "experts.2." in k]
        assert len(expert_keys) > 0

    def test_factory_make_qwen3moe(self):
        expander = make_qwen3moe_expert_clone(expand_factor=2, noise_scale=1e-5)
        assert expander.config.expand_factor == 2
        assert expander.config.noise_scale == 1e-5


class TestGenericMoEDepthExpander:
    def test_dry_run_inserts_layers(self):
        src_dir = _make_moe_dir(num_layers=4, num_experts=4)
        cfg = GenericMoEDepthConfig(num_new_layers=2)
        expander = GenericMoEDepthExpander(cfg)
        plan = expander.dry_run(str(src_dir))
        assert plan is not None

    def test_actual_write_increases_layers(self):
        src_dir = _make_moe_dir(num_layers=4, num_experts=2)
        cfg = GenericMoEDepthConfig(num_new_layers=2)
        expander = GenericMoEDepthExpander(cfg)
        plan = expander.dry_run(str(src_dir))
        assert plan.new_num_hidden_layers == 6

    def test_identity_blocks_zeroed(self):
        src_dir = _make_moe_dir(num_layers=4, num_experts=2)
        cfg = GenericMoEDepthConfig(num_new_layers=2)
        expander = GenericMoEDepthExpander(cfg)
        plan = expander.dry_run(str(src_dir))

        o_proj_keys = [k for k in plan.recipes if k.endswith("self_attn.o_proj.weight")]
        assert len(o_proj_keys) == 6
        zeroed = [k for k in o_proj_keys if plan.recipes[k].zero_out]
        assert len(zeroed) == 2

    def test_factory_make_qwen3moe_depth(self):
        expander = make_qwen3moe_zero_block_insert(num_new_layers=4)
        assert expander.config.num_new_layers == 4


class TestMoEWidthExpander:
    def test_dry_run_width_expansion(self):
        src_dir = _make_moe_dir(num_layers=4, num_experts=2)
        cfg = MoEWidthConfig(ffn_size_expansion=16, num_new_layers=0)
        expander = MoEWidthExpander(cfg)
        plan = expander.dry_run(str(src_dir))
        assert plan is not None

    def test_should_zero_uses_shared_helper(self):
        cfg = MoEWidthConfig(num_new_layers=2)
        expander = MoEWidthExpander(cfg)
        assert expander._should_zero("self_attn.o_proj.weight")
        assert expander._should_zero("mlp.experts.0.down_proj.weight")
        assert not expander._should_zero("mlp.experts.0.gate_proj.weight")
