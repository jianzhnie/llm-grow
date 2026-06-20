"""Shard writer utilities for safetensor-level expansion.

Extracted from ``safetensor/base.py`` to keep that module focused on the
expansion plan logic. Contains the tensor transform function, byte size
prediction, and the parallel worker entry point.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import save_file

from llm_grow.safetensor.recipe import TensorRecipe
from llm_grow.safetensor.utils import read_safetensors_header

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


def apply_recipe(
    src: torch.Tensor,
    recipe: TensorRecipe,
    interp_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a TensorRecipe to produce the output tensor.

    This is the core tensor transform used by both serial and parallel writers.
    Kept as a module-level function for picklability in multiprocessing.
    """
    if recipe.create_shape:
        dtype = _TORCH_DTYPES.get(recipe.create_dtype, torch.float32)
        return torch.zeros(recipe.create_shape, dtype=dtype)

    if recipe.zero_out:
        return torch.zeros_like(src)

    if interp_tensor is not None and recipe.interp_src_key:
        alpha = recipe.interp_alpha
        return cast(
            torch.Tensor,
            (alpha * src.float() + (1 - alpha) * interp_tensor.float()).to(src.dtype),
        )

    if recipe.dup_rows and recipe.router_split > 0:
        real = src[: recipe.router_split]
        zeros = src[recipe.router_split :]
        noise = (
            torch.randn_like(real) * recipe.dup_rows_noise_scale * real.float().std()
        )
        real_dup = real + noise.to(real.dtype)
        return torch.cat([real, real_dup, zeros, zeros], dim=0)

    if recipe.dup_rows:
        noise = torch.randn_like(src) * recipe.dup_rows_noise_scale * src.float().std()
        dup = src + noise.to(src.dtype)
        return torch.cat([src, dup], dim=0)

    if recipe.pad_rows > 0 or recipe.pad_cols > 0:
        if src.dim() == 2:
            tensor = torch.zeros(
                src.shape[0] + recipe.pad_rows,
                src.shape[1] + recipe.pad_cols,
                dtype=src.dtype,
            )
            tensor[: src.shape[0], : src.shape[1]] = src
        elif src.dim() == 1:
            tensor = torch.zeros(src.shape[0] + recipe.pad_rows, dtype=src.dtype)
            tensor[: src.shape[0]] = src
        else:
            raise ValueError(f"Unsupported tensor dim {src.dim()} for padding")
    else:
        tensor = src.contiguous()

    if recipe.add_noise_std > 0:
        noise = torch.randn_like(tensor) * recipe.add_noise_std
        return cast(torch.Tensor, tensor + noise)

    return tensor


def predict_recipe_bytes(src_meta: tuple[str, list[int]], recipe: TensorRecipe) -> int:
    """Predict output tensor byte size from header metadata (no tensor load).

    Used in Pass 1 of shard writing to plan assignments purely from
    the binary JSON header.
    """
    from llm_grow.safetensor.utils import DTYPE_SIZES

    dtype, shape = src_meta
    out_dtype = recipe.output_dtype(dtype)
    out_shape = recipe.output_shape(shape)
    elem = DTYPE_SIZES.get(out_dtype, 4)
    numel = 1
    for dim in out_shape:
        numel *= dim
    return elem * numel


def worker_write_shard(
    args: tuple[str, list[tuple[str, str, str, tuple]], set[str], bool],
) -> tuple[str, list[str]]:
    """Write one output shard in a worker process.

    Args:
        args: (output_path_str, items, expected_keys, resume) where each item is
              (src_shard_path_str, src_key, out_key, recipe_fields_tuple).

    Returns:
        (shard_basename, list_of_output_tensor_names)
    """
    from collections import defaultdict

    from safetensors import safe_open

    out_path_str, items, expected_keys, resume = args
    out_path = Path(out_path_str)

    if resume:
        # Atomic-ish resume check: try to read the header in one go.  If the
        # file is missing, truncated, or doesn't contain all expected keys, we
        # rewrite it.  This avoids the non-atomic exists()+header-read pair.
        try:
            header = read_safetensors_header(out_path)
            if expected_keys.issubset(header):
                return (out_path.name, list(expected_keys))
        except (FileNotFoundError, OSError, ValueError):
            pass

    by_src: dict[str, list[tuple[str, str, tuple]]] = defaultdict(list)
    for src_path, src_key, out_key, recipe_fields in items:
        by_src[src_path].append((src_key, out_key, recipe_fields))

    interp_handles: dict[str, Any] = {}

    def _get_interp_handle(path: str) -> Any:
        if path not in interp_handles:
            interp_handles[path] = safe_open(path, framework="pt", device="cpu")
        return interp_handles[path]

    tensors: dict[str, torch.Tensor] = {}
    for src_path, triples in by_src.items():
        with safe_open(src_path, framework="pt", device="cpu") as sf:
            for src_key, out_key, (
                zero_out,
                pad_rows,
                pad_cols,
                dup_rows,
                noise_scale,
                router_split,
                interp_shard_path,
                interp_key,
                interp_alpha,
                add_noise_std,
                create_shape,
                create_dtype,
            ) in triples:
                tensor = torch.zeros(1) if create_shape else sf.get_tensor(src_key)
                interp_t = None
                if interp_key and interp_shard_path:
                    interp_t = _get_interp_handle(interp_shard_path).get_tensor(
                        interp_key
                    )
                recipe = TensorRecipe(
                    src_shard="",
                    src_key=src_key,
                    zero_out=zero_out,
                    pad_rows=pad_rows,
                    pad_cols=pad_cols,
                    dup_rows=dup_rows,
                    dup_rows_noise_scale=noise_scale,
                    router_split=router_split,
                    interp_src_shard="",
                    interp_src_key=interp_key,
                    interp_alpha=interp_alpha,
                    add_noise_std=add_noise_std,
                    create_shape=create_shape,
                    create_dtype=create_dtype,
                )
                tensors[out_key] = apply_recipe(tensor, recipe, interp_t)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    save_file(tensors, str(tmp_path))
    os.replace(str(tmp_path), str(out_path))
    return (out_path.name, list(tensors.keys()))
