"""Tests for shared expansion-rule helpers."""

from __future__ import annotations

import pytest

from llm_grow.utils.expansion_rules import (
    build_identity_zero_suffixes,
    build_overlap_sequence,
    classify_linear_suffix,
    compute_pad_deltas,
)


class TestBuildIdentityZeroSuffixes:
    def test_default_suffixes(self):
        suffixes = build_identity_zero_suffixes()
        assert suffixes == [
            "self_attn.o_proj.weight",
            "self_attn.out_proj.weight",
            "mlp.down_proj.weight",
            "mlp.fc2.weight",
        ]

    def test_custom_names(self):
        suffixes = build_identity_zero_suffixes(
            attn_output_proj_names=["o_proj"],
            mlp_output_proj_names=["down_proj"],
        )
        assert suffixes == [
            "self_attn.o_proj.weight",
            "mlp.down_proj.weight",
        ]


class TestClassifyLinearSuffix:
    def test_attention_projections_are_hidden_to_hidden(self):
        for suf in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ):
            assert classify_linear_suffix(suf) == "hidden_to_hidden"

    def test_ffn_up_and_gate_are_hidden_to_inter(self):
        for suf in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            assert classify_linear_suffix(suf) == "hidden_to_inter"

    def test_ffn_down_is_inter_to_hidden(self):
        assert classify_linear_suffix("mlp.down_proj.weight") == "inter_to_hidden"

    def test_expert_projections_use_leaf_name(self):
        gate = classify_linear_suffix("mlp.experts.0.gate_proj.weight")
        down = classify_linear_suffix("mlp.experts.15.down_proj.weight")
        assert gate == "hidden_to_inter"
        assert down == "inter_to_hidden"

    def test_skip_embeddings_and_norms(self):
        for suf in (
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "layers.0.input_layernorm.weight",
        ):
            assert classify_linear_suffix(suf) == "skip"

    def test_module_names_without_weight_suffix(self):
        # In-memory expanders walk named_modules(), so suffixes may not end
        # in .weight.
        assert classify_linear_suffix("self_attn.q_proj") == "hidden_to_hidden"
        assert classify_linear_suffix("mlp.gate_proj") == "hidden_to_inter"
        assert classify_linear_suffix("mlp.down_proj") == "inter_to_hidden"


class TestComputePadDeltas:
    def test_no_expansion_returns_zero(self):
        assert compute_pad_deltas("self_attn.q_proj.weight") == (0, 0)

    def test_ffn_width_only(self):
        gate = compute_pad_deltas("mlp.gate_proj.weight", ffn_size_expansion=16)
        up = compute_pad_deltas("mlp.up_proj.weight", ffn_size_expansion=16)
        down = compute_pad_deltas("mlp.down_proj.weight", ffn_size_expansion=16)
        assert gate == (16, 0)
        assert up == (16, 0)
        assert down == (0, 16)

    def test_hidden_size_only(self):
        attn = compute_pad_deltas(
            "self_attn.q_proj.weight", hidden_size_expansion=8
        )
        gate = compute_pad_deltas("mlp.gate_proj.weight", hidden_size_expansion=8)
        down = compute_pad_deltas("mlp.down_proj.weight", hidden_size_expansion=8)
        assert attn == (8, 8)
        assert gate == (0, 8)
        assert down == (8, 0)

    def test_combined_expansion(self):
        # gate_proj: rows from ffn, cols from hidden
        gate = compute_pad_deltas(
            "mlp.gate_proj.weight",
            ffn_size_expansion=16,
            hidden_size_expansion=8,
        )
        assert gate == (16, 8)
        # down_proj: cols from ffn, rows from hidden
        down = compute_pad_deltas(
            "mlp.down_proj.weight",
            ffn_size_expansion=16,
            hidden_size_expansion=8,
        )
        assert down == (8, 16)

    def test_skip_layers_get_no_padding(self):
        deltas = compute_pad_deltas(
            "model.embed_tokens.weight",
            ffn_size_expansion=16,
            hidden_size_expansion=8,
        )
        assert deltas == (0, 0)


class TestBuildOverlapSequence:
    def test_typical_overlap(self):
        seq = build_overlap_sequence(10, 2)
        # upper = layers 0..7, lower = layers 2..9
        expected = [(i, False) for i in list(range(8)) + list(range(2, 10))]
        assert seq == expected
        assert len(seq) == 16

    def test_zero_overlap(self):
        seq = build_overlap_sequence(4, 0)
        upper = [(i, False) for i in range(4)]
        lower = [(i, False) for i in range(4)]
        assert seq == upper + lower

    def test_overlap_equal_to_num_layers_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            build_overlap_sequence(4, 4)

    def test_negative_overlap_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            build_overlap_sequence(4, -1)

    def test_non_integer_overlap_rejected(self):
        with pytest.raises(TypeError, match="integer"):
            build_overlap_sequence(4, 1.5)
