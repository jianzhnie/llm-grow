"""Base class for safetensor-level model expansion.

Core design
-----------
Each expander subclass produces an ``ExpansionPlan``: a mapping from every
output tensor name to a ``TensorRecipe`` that describes how to derive it from
a source tensor.  The base class then streams tensors from memory-mapped
source shards into output shards, keeping at most one output shard in RAM
at a time.

Memory profile
--------------
- Source shards: memory-mapped (only requested bytes are paged in).
- Active output shard: ≤ ``target_shard_bytes`` in RAM (default 4 GB).
- Total peak: ~one output shard + metadata.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from llm_grow.safetensor.utils import ShardIndex, parse_layer_idx, rename_layer_idx


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class TensorRecipe:
    """Describes how to produce one output tensor from a source tensor."""

    src_shard: str      # source shard filename (basename)
    src_key: str        # source tensor name
    zero_out: bool = False       # replace with all-zeros (identity block trick)
    pad_rows: int = 0            # zero-pad output dimension (rows / out_features)
    pad_cols: int = 0            # zero-pad input  dimension (cols / in_features)
    dup_rows: bool = False       # duplicate existing rows → [original; copy + noise]
    dup_rows_noise_scale: float = 1e-6  # noise std relative to tensor std


@dataclass
class ExpansionPlan:
    """Complete description of the expansion: one recipe per output tensor."""

    recipes: dict[str, TensorRecipe] = field(default_factory=dict)
    new_num_hidden_layers: int = 0
    config_patches: dict[str, Any] = field(default_factory=dict)

    def add(self, new_key: str, recipe: TensorRecipe) -> None:
        self.recipes[new_key] = recipe

    def passthrough(self, key: str, shard: str) -> None:
        """Add a tensor that is copied unchanged."""
        self.add(key, TensorRecipe(src_shard=shard, src_key=key))


# ── base expander ─────────────────────────────────────────────────────────────

class SafetensorExpanderBase(ABC):
    """Expand a HuggingFace safetensor model without loading it fully into RAM.

    Subclasses override ``_build_plan`` to describe the transformation;
    the base class handles all I/O and shard management.
    """

    #: Tensor suffixes zeroed in function-preserving identity blocks
    IDENTITY_ZERO_SUFFIXES: frozenset[str] = frozenset({
        "self_attn.o_proj.weight",
        "mlp.down_proj.weight",
    })

    #: Default output shard size (4 GB); override per-instance if needed
    DEFAULT_TARGET_SHARD_BYTES: int = 4 * 1024 ** 3

    # ── public API ────────────────────────────────────────────────────────────

    def expand(
        self,
        src_dir: str | Path,
        dst_dir: str | Path,
        *,
        target_shard_bytes: int | None = None,
        verbose: bool = True,
    ) -> None:
        """Expand model at ``src_dir``, write results to ``dst_dir``."""
        src_dir, dst_dir = Path(src_dir), Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        target = target_shard_bytes or self.DEFAULT_TARGET_SHARD_BYTES

        src_index = ShardIndex.load(src_dir)

        if verbose:
            _log(f"{type(self).__name__}  {src_dir} → {dst_dir}")
            _log(f"  source: {len(src_index.shard_files)} shard(s), "
                 f"{src_index.total_size_bytes()/1e9:.2f} GB, "
                 f"{len(src_index.all_keys)} tensors")

        plan = self._build_plan(src_index)

        if verbose:
            _log(f"  plan: {len(plan.recipes)} output tensors, "
                 f"new num_hidden_layers={plan.new_num_hidden_layers}")

        self._write_shards(src_index, dst_dir, plan, target, verbose)
        self._write_config(src_dir, dst_dir, plan)
        src_index.copy_non_weight_files(dst_dir)

        if verbose:
            _log(f"  Done → {dst_dir}")

    # ── abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        """Build the full tensor-level expansion plan."""

    def dry_run(self, src_dir: str | Path) -> ExpansionPlan:
        """Build and print the expansion plan without loading any tensor data."""
        src_dir = Path(src_dir)
        src_index = ShardIndex.load(src_dir)
        plan = self._build_plan(src_index)

        _log(f"[dry_run] {type(self).__name__}  src={src_dir}")
        _log(f"  source:  {len(src_index.all_keys)} tensors, "
             f"{len(src_index.shard_files)} shard(s)")
        _log(f"  output:  {len(plan.recipes)} tensors, "
             f"num_hidden_layers → {plan.new_num_hidden_layers}")
        _log(f"  config patches: {plan.config_patches}")

        zero_keys    = [k for k, r in plan.recipes.items() if r.zero_out]
        dup_keys     = [k for k, r in plan.recipes.items() if r.dup_rows]
        padded_keys  = [k for k, r in plan.recipes.items()
                        if (r.pad_rows or r.pad_cols) and not r.zero_out]
        new_keys     = [k for k in plan.recipes if k not in src_index.weight_map]

        _log(f"  zero-out tensors : {len(zero_keys)}")
        _log(f"  dup-rows tensors : {len(dup_keys)}")
        _log(f"  padded tensors   : {len(padded_keys)}")
        _log(f"  brand-new keys   : {len(new_keys)}")
        if new_keys[:3]:
            _log(f"    sample: {new_keys[:3]}")
        return plan

    def _write_shards(
        self,
        src_index: ShardIndex,
        dst_dir: Path,
        plan: ExpansionPlan,
        target_bytes: int,
        verbose: bool,
    ) -> None:
        """Stream output tensors into shards of ≤ target_bytes each."""
        src_handles = src_index.open_all_shards()
        sorted_keys = sorted(plan.recipes.keys())

        # Accumulate tensors for current output shard
        buf: dict[str, torch.Tensor] = {}
        buf_bytes = 0
        tmp_shards: list[tuple[Path, list[str]]] = []   # (path, keys)
        weight_map: dict[str, str] = {}

        def _flush(final: bool = False) -> None:
            if not buf:
                return
            tmp_path = dst_dir / f"_tmp_shard_{len(tmp_shards)}.safetensors"
            save_file(buf, str(tmp_path))
            tmp_shards.append((tmp_path, list(buf.keys())))
            buf.clear()

        for new_key in sorted_keys:
            recipe = plan.recipes[new_key]
            src_t = src_handles[recipe.src_shard].get_tensor(recipe.src_key)
            out_t = _apply_recipe(src_t, recipe)
            buf[new_key] = out_t
            buf_bytes += out_t.nbytes
            if buf_bytes >= target_bytes:
                _flush()
                buf_bytes = 0

        _flush()  # write remaining

        # Rename temp files to final shard names
        n = len(tmp_shards)
        for i, (tmp_path, keys) in enumerate(tmp_shards):
            final_name = (
                "model.safetensors"
                if n == 1
                else f"model-{i+1:05d}-of-{n:05d}.safetensors"
            )
            tmp_path.rename(dst_dir / final_name)
            for k in keys:
                weight_map[k] = final_name
            size_mb = (dst_dir / final_name).stat().st_size / 1e6
            if verbose:
                _log(f"  wrote {final_name}  ({len(keys)} tensors, {size_mb:.0f} MB)")

        # Write index for multi-shard output
        dst_index = ShardIndex(dst_dir, weight_map)
        if n > 1:
            dst_index.write_index_json(dst_dir)

    def _write_config(self, src_dir: Path, dst_dir: Path, plan: ExpansionPlan) -> None:
        cfg_path = src_dir / "config.json"
        if not cfg_path.exists():
            return
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["num_hidden_layers"] = plan.new_num_hidden_layers
        cfg.update(plan.config_patches)
        with open(dst_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    # ── shared plan-building helpers ──────────────────────────────────────────

    def _build_layer_plan(
        self,
        src_index: ShardIndex,
        *,
        layer_sequence: list[tuple[int, bool]],
    ) -> ExpansionPlan:
        """Construct an ExpansionPlan from a layer sequence.

        Args:
            src_index:      Source shard index.
            layer_sequence: Ordered list of (src_layer_idx, is_identity).
                            is_identity=True → zero o_proj + down_proj.
        """
        plan = ExpansionPlan(new_num_hidden_layers=len(layer_sequence))
        wmap = src_index.weight_map
        suffixes = src_index.layer_suffixes()

        # Per-layer tensors
        for new_idx, (src_idx, is_identity) in enumerate(layer_sequence):
            for suf in suffixes:
                src_key = f"model.layers.{src_idx}.{suf}"
                new_key = f"model.layers.{new_idx}.{suf}"
                zero = is_identity and suf in self.IDENTITY_ZERO_SUFFIXES
                plan.add(new_key, TensorRecipe(
                    src_shard=wmap[src_key],
                    src_key=src_key,
                    zero_out=zero,
                ))

        # Non-layer tensors pass through unchanged
        for key, shard in wmap.items():
            if parse_layer_idx(key) is None:
                plan.passthrough(key, shard)

        return plan

    @staticmethod
    def uniform_insert_positions(num_orig: int, num_new: int) -> list[int]:
        """Positions (original layer indices) after which to insert identity blocks."""
        if num_new == 0:
            return []
        step = num_orig / (num_new + 1)
        return sorted(set(int(round(step * (i + 1))) - 1 for i in range(num_new)))


# ── tensor transform ──────────────────────────────────────────────────────────

def _apply_recipe(src: torch.Tensor, recipe: TensorRecipe) -> torch.Tensor:
    # ── dup_rows: [A] → [A ; A+noise]  (used for router classifier)
    if recipe.dup_rows:
        noise = torch.randn_like(src) * recipe.dup_rows_noise_scale * src.float().std()
        dup = (src + noise.to(src.dtype))
        return torch.cat([src.clone(), dup], dim=0)

    # ── pad then optionally zero ──────────────────────────────────────────────
    if recipe.pad_rows > 0 or recipe.pad_cols > 0:
        if src.dim() == 2:
            t = torch.zeros(
                src.shape[0] + recipe.pad_rows,
                src.shape[1] + recipe.pad_cols,
                dtype=src.dtype,
            )
            t[: src.shape[0], : src.shape[1]] = src
        elif src.dim() == 1:
            t = torch.zeros(src.shape[0] + recipe.pad_rows, dtype=src.dtype)
            t[: src.shape[0]] = src
        else:
            raise ValueError(f"Unsupported tensor dim {src.dim()} for padding")
    else:
        t = src.clone()

    if recipe.zero_out:
        return torch.zeros_like(t)

    return t


def _log(msg: str) -> None:
    print(f"[SafetensorExpand] {msg}")
