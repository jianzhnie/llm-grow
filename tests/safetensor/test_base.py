"""Tests for llm_grow.safetensor.base: TensorRecipe, _apply_recipe, ExpansionPlan."""

import torch

from llm_grow.safetensor.base import ExpansionPlan, TensorRecipe, _apply_recipe

# ── TensorRecipe + _apply_recipe ─────────────────────────────────────────────


class TestApplyRecipeZeroOut:
    def test_zero_out_produces_all_zeros(self):
        src = torch.randn(4, 8)
        recipe = TensorRecipe(src_shard="s.safetensors", src_key="k", zero_out=True)
        out = _apply_recipe(src, recipe)
        assert out.shape == src.shape
        assert torch.all(out == 0)


class TestApplyRecipePad:
    def test_pad_rows_2d(self):
        src = torch.randn(4, 8)
        recipe = TensorRecipe(src_shard="s", src_key="k", pad_rows=2)
        out = _apply_recipe(src, recipe)
        assert out.shape == (6, 8)
        # original data preserved
        assert torch.allclose(out[:4, :], src)
        # padded rows are zero
        assert torch.all(out[4:, :] == 0)

    def test_pad_cols_2d(self):
        src = torch.randn(4, 8)
        recipe = TensorRecipe(src_shard="s", src_key="k", pad_cols=3)
        out = _apply_recipe(src, recipe)
        assert out.shape == (4, 11)
        assert torch.allclose(out[:, :8], src)
        assert torch.all(out[:, 8:] == 0)

    def test_pad_rows_and_cols(self):
        src = torch.randn(4, 8)
        recipe = TensorRecipe(src_shard="s", src_key="k", pad_rows=2, pad_cols=3)
        out = _apply_recipe(src, recipe)
        assert out.shape == (6, 11)

    def test_pad_rows_1d(self):
        src = torch.randn(10)
        recipe = TensorRecipe(src_shard="s", src_key="k", pad_rows=5)
        out = _apply_recipe(src, recipe)
        assert out.shape == (15,)
        assert torch.allclose(out[:10], src)
        assert torch.all(out[10:] == 0)


class TestApplyRecipeDupRows:
    def test_dup_rows_doubles_row_count(self):
        src = torch.randn(4, 8)
        recipe = TensorRecipe(src_shard="s", src_key="k", dup_rows=True)
        out = _apply_recipe(src, recipe)
        assert out.shape == (8, 8)
        # First half is original
        assert torch.allclose(out[:4, :], src)

    def test_dup_rows_with_router_split(self):
        src = torch.randn(6, 4)
        router_split = 4  # 4 real, 2 zero
        recipe = TensorRecipe(
            src_shard="s", src_key="k", dup_rows=True, router_split=router_split
        )
        out = _apply_recipe(src, recipe)
        # Layout: [real(4), real_dup(4), zeros(2), zeros_clone(2)] = 12 rows
        assert out.shape == (12, 4)
        # Original real rows preserved exactly
        assert torch.allclose(out[:4, :], src[:4, :])
        # Zero expert rows (indices 8:10 and 10:12) equal the original zero rows
        assert torch.allclose(out[8:10, :], src[4:, :])
        assert torch.allclose(out[10:12, :], src[4:, :])


class TestApplyRecipeNoise:
    def test_add_noise_changes_values(self):
        torch.manual_seed(42)
        src = torch.ones(4, 4)
        recipe = TensorRecipe(src_shard="s", src_key="k", add_noise_std=0.1)
        out = _apply_recipe(src, recipe)
        assert out.shape == src.shape
        # Values should differ from the original ones tensor
        assert not torch.allclose(out, src)


class TestApplyRecipeCreateShape:
    def test_create_shape_produces_zero_tensor(self):
        src = torch.randn(2, 2)  # ignored
        recipe = TensorRecipe(
            src_shard="s", src_key="k", create_shape=(8, 16), create_dtype="F32"
        )
        out = _apply_recipe(src, recipe)
        assert out.shape == (8, 16)
        assert out.dtype == torch.float32
        assert torch.all(out == 0)

    def test_create_shape_bf16(self):
        src = torch.randn(1)
        recipe = TensorRecipe(
            src_shard="s", src_key="k", create_shape=(3, 5), create_dtype="BF16"
        )
        out = _apply_recipe(src, recipe)
        assert out.dtype == torch.bfloat16
        assert out.shape == (3, 5)


class TestApplyRecipeInterpolation:
    def test_interp_produces_weighted_average(self):
        src = torch.ones(4, 4, dtype=torch.float32) * 10.0
        interp = torch.ones(4, 4, dtype=torch.float32) * 20.0
        alpha = 0.3
        recipe = TensorRecipe(
            src_shard="s",
            src_key="k",
            interp_src_shard="s2",
            interp_src_key="k2",
            interp_alpha=alpha,
        )
        out = _apply_recipe(src, recipe, interp_tensor=interp)
        expected = alpha * src + (1 - alpha) * interp
        assert torch.allclose(out, expected)


# ── ExpansionPlan ─────────────────────────────────────────────────────────────


class TestExpansionPlan:
    def test_add_and_passthrough(self):
        plan = ExpansionPlan(new_num_hidden_layers=4)
        recipe = TensorRecipe(src_shard="shard.safetensors", src_key="layer.0.weight")
        plan.add("layer.0.weight", recipe)
        plan.passthrough("embed.weight", "shard.safetensors")

        assert "layer.0.weight" in plan.recipes
        assert "embed.weight" in plan.recipes
        pt = plan.recipes["embed.weight"]
        assert pt.src_key == "embed.weight"
        assert pt.zero_out is False

    def test_to_dict_from_dict_roundtrip(self):
        plan = ExpansionPlan(
            new_num_hidden_layers=8,
            config_patches={"num_experts": 16},
        )
        plan.add(
            "layer.0.weight",
            TensorRecipe(
                src_shard="shard-00001.safetensors",
                src_key="layer.0.weight",
                zero_out=True,
                pad_rows=4,
            ),
        )
        plan.passthrough("embed.weight", "shard-00001.safetensors")

        data = plan.to_dict()
        restored = ExpansionPlan.from_dict(data)

        assert restored.new_num_hidden_layers == 8
        assert restored.config_patches == {"num_experts": 16}
        assert len(restored.recipes) == 2
        r = restored.recipes["layer.0.weight"]
        assert r.zero_out is True
        assert r.pad_rows == 4
        assert r.src_shard == "shard-00001.safetensors"
