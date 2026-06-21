"""Tests for safetensor/writer.py — tensor transform and byte prediction."""

from __future__ import annotations

import pytest
import torch

from llm_grow.safetensor.base import TensorRecipe
from llm_grow.safetensor.writer import apply_recipe, predict_recipe_bytes


class TestApplyRecipe:
    def test_passthrough_clones(self):
        src = torch.randn(4, 8)
        recipe = TensorRecipe(src_shard="s.safetensors", src_key="k")
        out = apply_recipe(src, recipe)
        assert out.shape == src.shape
        assert torch.equal(out, src)
        assert out.is_contiguous()

    def test_zero_out(self):
        src = torch.randn(4, 8)
        recipe = TensorRecipe(src_shard="s", src_key="k", zero_out=True)
        out = apply_recipe(src, recipe)
        assert out.shape == src.shape
        assert out.abs().max().item() == 0.0

    def test_create_shape(self):
        recipe = TensorRecipe(
            src_shard="", src_key="", create_shape=(3, 5), create_dtype="BF16"
        )
        out = apply_recipe(torch.zeros(1), recipe)
        assert out.shape == (3, 5)
        assert out.dtype == torch.bfloat16

    def test_dup_rows(self):
        src = torch.randn(4, 8)
        recipe = TensorRecipe(
            src_shard="s", src_key="k", dup_rows=True, dup_rows_noise_scale=1e-6
        )
        out = apply_recipe(src, recipe)
        assert out.shape == (8, 8)

    def test_dup_rows_with_router_split(self):
        src = torch.randn(6, 4)
        recipe = TensorRecipe(
            src_shard="s",
            src_key="k",
            dup_rows=True,
            dup_rows_noise_scale=1e-6,
            router_split=4,
        )
        out = apply_recipe(src, recipe)
        assert out.shape == (12, 4)

    def test_pad_2d(self):
        src = torch.ones(4, 8)
        recipe = TensorRecipe(src_shard="s", src_key="k", pad_rows=2, pad_cols=3)
        out = apply_recipe(src, recipe)
        assert out.shape == (6, 11)
        assert out[:4, :8].sum().item() == 32.0
        assert out[4:, :].sum().item() == 0.0
        assert out[:, 8:].sum().item() == 0.0

    def test_pad_1d(self):
        src = torch.ones(5)
        recipe = TensorRecipe(src_shard="s", src_key="k", pad_rows=3)
        out = apply_recipe(src, recipe)
        assert out.shape == (8,)
        assert out[:5].sum().item() == 5.0
        assert out[5:].sum().item() == 0.0

    def test_interpolation(self):
        src = torch.ones(4, 4)
        interp = torch.zeros(4, 4)
        recipe = TensorRecipe(
            src_shard="s",
            src_key="k",
            interp_src_shard="i",
            interp_src_key="ik",
            interp_alpha=0.5,
        )
        out = apply_recipe(src, recipe, interp_tensor=interp)
        assert torch.allclose(out, torch.full((4, 4), 0.5))

    def test_add_noise(self):
        src = torch.zeros(4, 4)
        recipe = TensorRecipe(src_shard="s", src_key="k", add_noise_std=0.1)
        out = apply_recipe(src, recipe)
        assert out.shape == (4, 4)
        assert out.abs().max().item() > 0


class TestTensorRecipeValidation:
    def test_zero_out_with_padding_allowed(self):
        """Width-expanded identity blocks legitimately set both flags."""
        recipe = TensorRecipe(
            src_shard="s", src_key="k", zero_out=True, pad_rows=2, pad_cols=3
        )
        assert recipe.zero_out
        assert recipe.pad_rows == 2

    def test_zero_out_with_noise_rejected(self):
        with pytest.raises(ValueError):
            TensorRecipe(
                src_shard="s",
                src_key="k",
                zero_out=True,
                add_noise_std=0.1,
            )

    def test_dup_rows_with_padding_rejected(self):
        with pytest.raises(ValueError):
            TensorRecipe(
                src_shard="s", src_key="k", dup_rows=True, pad_rows=2
            )

    def test_router_split_requires_dup_rows(self):
        with pytest.raises(ValueError):
            TensorRecipe(
                src_shard="s", src_key="k", router_split=4, dup_rows=False
            )

    def test_create_shape_exclusive(self):
        with pytest.raises(ValueError):
            TensorRecipe(
                src_shard="s",
                src_key="k",
                create_shape=(2, 3),
                zero_out=True,
            )

    def test_multiple_primary_ops_rejected(self):
        with pytest.raises(ValueError):
            TensorRecipe(
                src_shard="s",
                src_key="k",
                dup_rows=True,
                interp_src_shard="i",
                interp_src_key="ik",
            )


class TestPredictRecipeBytes:
    def test_passthrough(self):
        meta = ("F32", [10, 20])
        recipe = TensorRecipe(src_shard="s", src_key="k")
        assert predict_recipe_bytes(meta, recipe) == 10 * 20 * 4

    def test_zero_out(self):
        meta = ("BF16", [100, 200])
        recipe = TensorRecipe(src_shard="s", src_key="k", zero_out=True)
        assert predict_recipe_bytes(meta, recipe) == 100 * 200 * 2

    def test_dup_rows(self):
        meta = ("F16", [8, 16])
        recipe = TensorRecipe(
            src_shard="s", src_key="k", dup_rows=True, dup_rows_noise_scale=1e-6
        )
        assert predict_recipe_bytes(meta, recipe) == 8 * 16 * 2 * 2

    def test_pad(self):
        meta = ("F32", [10, 20])
        recipe = TensorRecipe(src_shard="s", src_key="k", pad_rows=5, pad_cols=10)
        assert predict_recipe_bytes(meta, recipe) == 15 * 30 * 4

    def test_create_shape(self):
        recipe = TensorRecipe(
            src_shard="", src_key="", create_shape=(64, 128), create_dtype="BF16"
        )
        assert predict_recipe_bytes(("F32", [1]), recipe) == 64 * 128 * 2
