"""Tests for llm_grow.safetensor.detect architecture auto-detection."""

from __future__ import annotations

import json
from pathlib import Path

from llm_grow.safetensor.detect import (
    ModelProfile,
    _count_experts,
    _derive_zero_suffixes,
    _detect_config_keys,
    detect_model,
)


def _make_model_dir(
    config: dict,
    weight_map: dict[str, str],
    tmp_path: Path | None = None,
) -> Path:
    """Create a temporary model directory with config.json and index.json."""
    if tmp_path is None:
        import tempfile

        tmp = Path(tempfile.mkdtemp())
    else:
        tmp = tmp_path
        tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "config.json").write_text(json.dumps(config))
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return tmp


class TestDetectModelDense:
    def test_detects_dense_qwen3(self):
        src_dir = _make_model_dir(
            config={
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "num_hidden_layers": 28,
                "hidden_size": 1024,
            },
            weight_map={
                "model.embed_tokens.weight": "model.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
                "model.layers.0.self_attn.o_proj.weight": "model.safetensors",
                "model.layers.0.mlp.gate_proj.weight": "model.safetensors",
                "model.layers.0.mlp.down_proj.weight": "model.safetensors",
                "model.layers.27.mlp.down_proj.weight": "model.safetensors",
                "lm_head.weight": "model.safetensors",
            },
        )
        profile = detect_model(src_dir)
        assert profile.family == "dense"
        assert profile.model_type == "qwen3"
        assert profile.num_hidden_layers == 28
        assert not profile.is_moe
        assert profile.attn_zero_suffixes == ["self_attn.o_proj.weight"]
        assert profile.dense_mlp_zero_suffixes == ["mlp.down_proj.weight"]


class TestDetectModelMoEStandard:
    def test_detects_qwen3_moe(self):
        num_experts = 4
        weight_map = {
            "model.embed_tokens.weight": "model.safetensors",
            "model.layers.0.self_attn.o_proj.weight": "model.safetensors",
            "model.layers.0.mlp.gate.weight": "model.safetensors",
        }
        for e in range(num_experts):
            weight_map[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = (
                "model.safetensors"
            )
            weight_map[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = (
                "model.safetensors"
            )
        weight_map["model.layers.47.mlp.experts.0.down_proj.weight"] = (
            "model.safetensors"
        )
        src_dir = _make_model_dir(
            config={
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "num_hidden_layers": 48,
                "num_experts": num_experts,
                "num_experts_per_tok": 2,
            },
            weight_map=weight_map,
        )
        profile = detect_model(src_dir)
        assert profile.family == "standard_moe"
        assert profile.is_moe
        assert profile.experts_per_moe_layer == num_experts
        assert profile.router_weight_suffix == "mlp.gate.weight"
        assert profile.expert_count_config_key == "num_experts"
        assert profile.topk_config_key == "num_experts_per_tok"


class TestDetectModelDeepSeekMoE:
    def test_detects_kimi_k2(self):
        weight_map = {
            "model.embed_tokens.weight": "model.safetensors",
            "model.layers.0.mlp.gate_proj.weight": "model.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model.safetensors",
            "model.layers.1.mlp.gate.weight": "model.safetensors",
            "model.layers.1.mlp.experts.0.gate_proj.weight": "model.safetensors",
            "model.layers.1.mlp.experts.0.weight_scale_inv": "model.safetensors",
            "model.layers.1.mlp.experts.383.down_proj.weight": "model.safetensors",
            "model.layers.1.mlp.shared_experts.gate_proj.weight": "model.safetensors",
            "model.layers.1.self_attn.q_a_proj.weight": "model.safetensors",
        }
        src_dir = _make_model_dir(
            config={
                "model_type": "deepseek_v3",
                "architectures": ["DeepseekV3ForCausalLM"],
                "num_hidden_layers": 61,
                "n_routed_experts": 384,
                "num_experts_per_tok": 8,
            },
            weight_map=weight_map,
        )
        profile = detect_model(src_dir)
        assert profile.family == "deepseek_moe"
        assert profile.is_moe
        assert profile.has_fp8
        assert profile.has_mla_attn
        assert profile.has_shared_expert
        assert profile.dense_only_layers == [0]
        assert profile.expert_count_config_key == "n_routed_experts"

    def test_mixed_dense_moe_zero_suffixes(self):
        """Layer 0 dense + layer 1 MoE should still zero mlp.down_proj.weight."""
        weight_map = {
            "model.embed_tokens.weight": "model.safetensors",
            "model.layers.0.mlp.gate_proj.weight": "model.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model.safetensors",
            "model.layers.1.mlp.gate.weight": "model.safetensors",
            "model.layers.1.mlp.experts.0.gate_proj.weight": "model.safetensors",
            "model.layers.1.mlp.experts.0.down_proj.weight": "model.safetensors",
        }
        src_dir = _make_model_dir(
            config={
                "model_type": "deepseek_v3",
                "architectures": ["DeepseekV3ForCausalLM"],
                "num_hidden_layers": 2,
                "n_routed_experts": 1,
                "num_experts_per_tok": 1,
            },
            weight_map=weight_map,
        )
        profile = detect_model(src_dir)
        assert profile.dense_mlp_zero_suffixes == ["mlp.down_proj.weight"]
        assert profile.attn_zero_suffixes == ["self_attn.o_proj.weight"]


class TestDetectModelLongCat:
    def test_detects_longcat(self):
        weight_map = {
            "model.embed_tokens.weight": "model.safetensors",
            "model.layers.0.self_attn.0.o_proj.weight": "model.safetensors",
            "model.layers.0.self_attn.1.o_proj.weight": "model.safetensors",
            "model.layers.0.mlps.0.down_proj.weight": "model.safetensors",
            "model.layers.0.mlps.1.down_proj.weight": "model.safetensors",
            "model.layers.0.mlp.router.classifier.weight": "model.safetensors",
            "model.layers.0.mlp.experts.0.down_proj.weight": "model.safetensors",
            "model.layers.0.mlp.experts.511.down_proj.weight": "model.safetensors",
        }
        src_dir = _make_model_dir(
            config={
                "model_type": "longcat_flash",
                "architectures": ["LongcatFlashForCausalLM"],
                "num_hidden_layers": 28,
                "n_routed_experts": 512,
                "moe_topk": 12,
            },
            weight_map=weight_map,
        )
        profile = detect_model(src_dir)
        assert profile.family == "longcat"
        assert profile.has_dual_attn
        assert profile.has_dual_dense_mlp
        assert profile.router_weight_suffix == "mlp.router.classifier.weight"
        assert profile.attn_zero_suffixes == [
            "self_attn.0.o_proj.weight",
            "self_attn.1.o_proj.weight",
        ]


class TestDetectHelpers:
    def test_detect_router_via_model_profile(self):
        wmap = {
            "model.layers.0.mlp.gate.weight": "s",
            "model.layers.1.mlp.gate.weight": "s",
        }
        src_dir = _make_model_dir(
            config={
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "num_hidden_layers": 2,
                "num_experts": 2,
                "num_experts_per_tok": 1,
            },
            weight_map=wmap,
        )
        profile = detect_model(src_dir)
        assert profile.router_weight_suffix == "mlp.gate.weight"
        assert profile.router_bias_suffix is None

    def test_detect_router_deepseek_via_model_profile(self):
        wmap = {
            "model.layers.1.mlp.gate.weight": "s",
            "model.layers.1.mlp.gate.e_score_correction_bias": "s",
        }
        src_dir = _make_model_dir(
            config={
                "model_type": "deepseek_v3",
                "architectures": ["DeepseekV3ForCausalLM"],
                "num_hidden_layers": 2,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
            },
            weight_map=wmap,
        )
        profile = detect_model(src_dir)
        assert profile.router_weight_suffix == "mlp.gate.weight"
        assert profile.router_bias_suffix == "mlp.gate.e_score_correction_bias"

    def test_count_experts(self):
        wmap = {
            "model.layers.0.mlp.experts.0.weight": "s",
            "model.layers.0.mlp.experts.1.weight": "s",
            "model.layers.1.mlp.experts.0.weight": "s",
            "model.layers.1.mlp.experts.1.weight": "s",
            "model.layers.1.mlp.experts.2.weight": "s",
            "model.layers.2.mlp.gate_proj.weight": "s",
        }
        n_experts, dense_layers = _count_experts(set(wmap), num_layers=3)
        assert n_experts == 3
        assert dense_layers == [2]

    def test_detect_config_keys(self):
        assert _detect_config_keys({"num_experts": 8, "num_experts_per_tok": 2}) == (
            "num_experts",
            "num_experts_per_tok",
        )
        assert _detect_config_keys({"n_routed_experts": 64, "moe_topk": 4}) == (
            "n_routed_experts",
            "moe_topk",
        )
        assert _detect_config_keys({}) == ("num_experts", "num_experts_per_tok")

    def test_derive_zero_suffixes_dense(self):
        attn, dense = _derive_zero_suffixes(
            has_dual_attn=False,
            has_dual_mlp=False,
            is_moe=False,
            dense_layers=[],
        )
        assert attn == ["self_attn.o_proj.weight"]
        assert dense == ["mlp.down_proj.weight"]

    def test_derive_zero_suffixes_longcat(self):
        attn, dense = _derive_zero_suffixes(
            has_dual_attn=True,
            has_dual_mlp=True,
            is_moe=True,
            dense_layers=[],
        )
        assert attn == ["self_attn.0.o_proj.weight", "self_attn.1.o_proj.weight"]
        assert dense == ["mlps.0.down_proj.weight", "mlps.1.down_proj.weight"]


class TestModelProfile:
    def test_summary_contains_family(self):
        profile = ModelProfile(
            src_dir=Path("/tmp"),
            model_type="qwen3",
            arch_class="Qwen3ForCausalLM",
            num_hidden_layers=28,
        )
        assert "dense" in profile.summary()
        assert "Qwen3ForCausalLM" in profile.summary()
