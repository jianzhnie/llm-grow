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

from llm_grow.safetensor.utils import ShardIndex, parse_layer_idx

# ── data classes ─────────────────────────────────────────────────────────────


@dataclass
class TensorRecipe:
    """Describes how to produce one output tensor from a source tensor."""

    src_shard: str  # source shard filename (basename)
    src_key: str  # source tensor name
    zero_out: bool = False  # replace with all-zeros (identity block trick)
    pad_rows: int = 0  # zero-pad output dimension (rows / out_features)
    pad_cols: int = 0  # zero-pad input  dimension (cols / in_features)
    dup_rows: bool = False  # duplicate existing rows → [original; copy + noise]
    dup_rows_noise_scale: float = 1e-6  # noise std relative to tensor std

    # ── router-aware expansion ────────────────────────────────────────────────
    # When router_split > 0 and dup_rows=True:
    #   rows [0 : router_split]       → real experts  → duplicate WITH noise
    #   rows [router_split : end]     → zero experts  → duplicate WITHOUT noise
    # This preserves the identity-block semantics of zero experts.
    router_split: int = 0  # row index separating real from zero experts (0 = disabled)


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
    IDENTITY_ZERO_SUFFIXES: frozenset[str] = frozenset(
        {
            "self_attn.o_proj.weight",
            "mlp.down_proj.weight",
        }
    )

    #: Default output shard size (4 GB); override per-instance if needed
    DEFAULT_TARGET_SHARD_BYTES: int = 4 * 1024**3

    # ── public API ────────────────────────────────────────────────────────────

    def expand(
        self,
        src_dir: str | Path,
        dst_dir: str | Path,
        *,
        target_shard_bytes: int | None = None,
        workers: int = 1,
        verbose: bool = True,
    ) -> None:
        """Expand model at ``src_dir``, write results to ``dst_dir``.

        Args:
            target_shard_bytes: Maximum bytes per output shard.  Defaults to
                ``auto_detect_shard_size`` (mean of input shards, or 4 GB).
            workers: Number of parallel writer processes (1 = serial).
        """
        src_dir, dst_dir = Path(src_dir), Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        src_index = ShardIndex.load(src_dir)

        if target_shard_bytes is None:
            from llm_grow.safetensor.utils import auto_detect_shard_size

            target_shard_bytes = auto_detect_shard_size(src_dir, src_index.shard_files)

        if verbose:
            _log(f"{type(self).__name__}  {src_dir} → {dst_dir}")
            _log(
                f"  source: {len(src_index.shard_files)} shard(s), "
                f"{src_index.total_size_bytes() / 1e9:.2f} GB, "
                f"{len(src_index.all_keys)} tensors"
            )

        plan = self._build_plan(src_index)

        if verbose:
            _log(
                f"  plan: {len(plan.recipes)} output tensors, "
                f"new num_hidden_layers={plan.new_num_hidden_layers}"
            )

        self._write_shards(src_index, dst_dir, plan, target_shard_bytes, verbose, workers)
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
        _log(f"  source:  {len(src_index.all_keys)} tensors, {len(src_index.shard_files)} shard(s)")
        _log(f"  output:  {len(plan.recipes)} tensors, num_hidden_layers → {plan.new_num_hidden_layers}")
        _log(f"  config patches: {plan.config_patches}")

        zero_keys = [k for k, r in plan.recipes.items() if r.zero_out]
        dup_keys = [k for k, r in plan.recipes.items() if r.dup_rows]
        padded_keys = [k for k, r in plan.recipes.items() if (r.pad_rows or r.pad_cols) and not r.zero_out]
        new_keys = [k for k in plan.recipes if k not in src_index.weight_map]

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
        workers: int = 1,
    ) -> None:
        """Two-pass shard writer.

        Pass 1  Read only the binary JSON headers of each source shard
                (O(header size), no tensor data touched) to compute exact
                output tensor sizes and assign tensors to output shards.

        Pass 2  For each output shard, open only the needed source shards
                (via mmap), apply recipes, and write directly to the final
                shard name — no temporary files, no renaming.
        """
        from llm_grow.safetensor.utils import read_safetensors_header

        # ── Pass 1: header scan → shard assignment ────────────────────────────
        src_headers: dict[str, dict[str, tuple[str, list[int]]]] = {
            sf: read_safetensors_header(src_index.model_dir / sf) for sf in src_index.shard_files
        }

        sorted_keys = sorted(plan.recipes.keys())
        shard_groups: list[list[str]] = [[]]
        current_bytes = 0
        for new_key in sorted_keys:
            recipe = plan.recipes[new_key]
            src_meta = src_headers[recipe.src_shard].get(recipe.src_key)
            if src_meta is None:
                continue
            out_bytes = _predict_recipe_bytes(src_meta, recipe)
            if current_bytes + out_bytes > target_bytes and current_bytes > 0:
                shard_groups.append([])
                current_bytes = 0
            shard_groups[-1].append(new_key)
            current_bytes += out_bytes

        n = len(shard_groups)
        weight_map: dict[str, str] = {}

        if verbose:
            _log(f"  Pass 1 done: {n} output shard(s) planned")

        # ── Pass 2: write output shards ───────────────────────────────────────
        if workers > 1:
            self._write_shards_parallel(src_index, dst_dir, plan, shard_groups, weight_map, verbose, workers)
        else:
            src_handles = src_index.open_all_shards()
            for i, group_keys in enumerate(shard_groups):
                shard_name = "model.safetensors" if n == 1 else f"model-{i + 1:05d}-of-{n:05d}.safetensors"
                tensors: dict[str, torch.Tensor] = {}
                for new_key in group_keys:
                    recipe = plan.recipes[new_key]
                    src_t = src_handles[recipe.src_shard].get_tensor(recipe.src_key)
                    tensors[new_key] = _apply_recipe(src_t, recipe)
                    weight_map[new_key] = shard_name
                save_file(tensors, str(dst_dir / shard_name))
                if verbose:
                    size_mb = (dst_dir / shard_name).stat().st_size / 1e6
                    _log(f"  wrote {shard_name}  ({len(tensors)} tensors, {size_mb:.0f} MB)")

        dst_index = ShardIndex(dst_dir, weight_map)
        if n > 1:
            dst_index.write_index_json(dst_dir)

    def _write_shards_parallel(
        self,
        src_index: ShardIndex,
        dst_dir: Path,
        plan: ExpansionPlan,
        shard_groups: list[list[str]],
        weight_map: dict[str, str],
        verbose: bool,
        workers: int,
    ) -> None:
        """Write output shards in parallel using ProcessPoolExecutor."""
        from concurrent.futures import ProcessPoolExecutor

        from tqdm import tqdm

        n = len(shard_groups)
        tasks = []
        for i, group_keys in enumerate(shard_groups):
            shard_name = "model.safetensors" if n == 1 else f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            # Gather (src_shard_path, src_key, out_key, recipe_tuple) per group
            items = []
            for new_key in group_keys:
                r = plan.recipes[new_key]
                items.append(
                    (
                        str(src_index.model_dir / r.src_shard),
                        r.src_key,
                        new_key,
                        (
                            r.zero_out,
                            r.pad_rows,
                            r.pad_cols,
                            r.dup_rows,
                            r.dup_rows_noise_scale,
                            r.router_split,
                        ),
                    )
                )
            tasks.append((str(dst_dir / shard_name), items))

        chunksize = max(1, len(tasks) // workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(
                tqdm(
                    executor.map(_worker_write_shard, tasks, chunksize=chunksize),
                    total=len(tasks),
                    desc="Writing shards",
                    disable=not verbose,
                )
            )

        for shard_name, tensor_names in results:
            for name in tensor_names:
                weight_map[name] = shard_name

    def _write_config(self, src_dir: Path, dst_dir: Path, plan: ExpansionPlan) -> None:
        """Write updated config.json to dst_dir.

        Correctly handles architectures that use ``num_layers`` instead of
        ``num_hidden_layers`` (e.g. LongCat-Flash).  After writing, verifies
        that every file referenced by ``auto_map`` is present in dst_dir so
        the model can be loaded with ``trust_remote_code=True``.
        """
        cfg_path = src_dir / "config.json"
        if not cfg_path.exists():
            return
        with open(cfg_path) as f:
            cfg = json.load(f)

        # ── update layer count with the correct key ───────────────────────────
        # Different architectures store layer count under different keys.
        # Preserve whichever key the source model uses; fall back to the standard.
        if plan.new_num_hidden_layers > 0:
            layers_key = next(
                (k for k in ("num_hidden_layers", "num_layers") if k in cfg),
                "num_hidden_layers",
            )
            cfg[layers_key] = plan.new_num_hidden_layers

        # ── apply expansion-specific patches (expert count, topk, etc.) ───────
        cfg.update(plan.config_patches)

        with open(dst_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        # ── verify auto_map referenced files are present ──────────────────────
        auto_map: dict[str, str] = cfg.get("auto_map", {})
        missing: list[str] = []
        for _cls, ref in auto_map.items():
            # ref format: "module_name.ClassName"
            module_name = ref.split(".")[0] + ".py"
            if not (dst_dir / module_name).exists():
                missing.append(module_name)
        if missing:
            print(
                f"[_write_config] WARNING: auto_map references missing files in {dst_dir}: "
                f"{missing}. Run expand() (not dry_run) to copy them automatically."
            )

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
                if src_key not in wmap:
                    continue  # suffix absent for this layer (mixed-arch models)
                new_key = f"model.layers.{new_idx}.{suf}"
                zero = is_identity and suf in self.IDENTITY_ZERO_SUFFIXES
                plan.add(
                    new_key,
                    TensorRecipe(
                        src_shard=wmap[src_key],
                        src_key=src_key,
                        zero_out=zero,
                    ),
                )

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
        return sorted({round(step * (i + 1)) - 1 for i in range(num_new)})


# ── tensor transform ──────────────────────────────────────────────────────────


def _apply_recipe(src: torch.Tensor, recipe: TensorRecipe) -> torch.Tensor:
    # ── dup_rows with router_split: handle real/zero expert rows separately ────
    # rows [0 : router_split]   → real experts → duplicate WITH noise
    # rows [router_split : end] → zero experts → duplicate WITHOUT noise
    if recipe.dup_rows and recipe.router_split > 0:
        real = src[: recipe.router_split]
        zeros = src[recipe.router_split :]
        noise = torch.randn_like(real) * recipe.dup_rows_noise_scale * real.float().std()
        real_dup = real + noise.to(real.dtype)
        return torch.cat([real, real_dup, zeros, zeros.clone()], dim=0)

    # ── dup_rows without split: duplicate all rows uniformly ──────────────────
    if recipe.dup_rows:
        noise = torch.randn_like(src) * recipe.dup_rows_noise_scale * src.float().std()
        dup = src + noise.to(src.dtype)
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


def _predict_recipe_bytes(src_meta: tuple[str, list[int]], recipe: TensorRecipe) -> int:
    """Predict output tensor byte size from header metadata (no tensor load).

    Used in Pass 1 of ``_write_shards`` to plan shard assignments purely from
    the binary JSON header — no mmap or tensor data needed.
    """
    from llm_grow.safetensor.utils import DTYPE_SIZES, nbytes_from_header

    dtype, shape = src_meta
    elem = DTYPE_SIZES.get(dtype, 4)
    src_bytes = nbytes_from_header(dtype, shape)

    if recipe.zero_out:
        return src_bytes  # same shape, just zeroed
    if recipe.dup_rows:
        return src_bytes * 2  # always doubles row count (expand_factor=2)
    if recipe.pad_rows > 0 or recipe.pad_cols > 0:
        ndim = len(shape)
        if ndim == 2:
            return elem * (shape[0] + recipe.pad_rows) * (shape[1] + recipe.pad_cols)
        if ndim == 1:
            return elem * (shape[0] + recipe.pad_rows)
    return src_bytes


# ── module-level worker for ProcessPoolExecutor (must be picklable) ───────────


def _worker_write_shard(
    args: tuple[str, list[tuple[str, str, str, tuple]]],
) -> tuple[str, list[str]]:
    """Write one output shard in a worker process.

    Args:
        args: (output_path_str, items) where each item is
              (src_shard_path_str, src_key, out_key, recipe_fields_tuple).

    Returns:
        (shard_basename, list_of_output_tensor_names)
    """
    from collections import defaultdict

    from safetensors import safe_open
    from safetensors.torch import save_file

    out_path_str, items = args
    out_path = Path(out_path_str)

    # Group by source shard path to open each file only once
    by_src: dict[str, list[tuple[str, str, tuple]]] = defaultdict(list)
    for src_path, src_key, out_key, recipe_fields in items:
        by_src[src_path].append((src_key, out_key, recipe_fields))

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
            ) in triples:
                t = sf.get_tensor(src_key)
                recipe = TensorRecipe(
                    src_shard="",
                    src_key=src_key,
                    zero_out=zero_out,
                    pad_rows=pad_rows,
                    pad_cols=pad_cols,
                    dup_rows=dup_rows,
                    dup_rows_noise_scale=noise_scale,
                    router_split=router_split,
                )
                tensors[out_key] = _apply_recipe(t, recipe)

    save_file(tensors, out_path_str)
    return (out_path.name, list(tensors.keys()))


def _log(msg: str) -> None:
    print(f"[SafetensorExpand] {msg}")
