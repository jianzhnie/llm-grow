"""Shard writer for safetensor-level model expansion.

This module encapsulates the full I/O pipeline (Pass 1 header scanning,
Pass 2 shard writing, config serialization, and post-write validation) so
that ``SafetensorExpanderBase`` can focus on plan construction.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from llm_grow.safetensor.recipe import ExpansionPlan, TensorRecipe
from llm_grow.safetensor.utils import (
    ShardIndex,
    peek_model_config,
    read_safetensors_header,
)
from llm_grow.safetensor.writer import (
    apply_recipe,
    predict_recipe_bytes,
    worker_write_shard,
)
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


def _safe_close_handle(handle: Any) -> None:
    """Close a safetensors mmap handle using the context-manager protocol.

    Avoids calling ``handle.__exit__()`` directly, which is fragile when the
    underlying library changes its cleanup logic.
    """
    with contextlib.suppress(Exception):
        handle.__exit__(None, None, None)


def _atomic_replace(tmp_path: Path, dst_path: Path) -> None:
    """Atomically rename *tmp_path* to *dst_path*, falling back to copy on EXDEV."""
    try:
        os.replace(str(tmp_path), str(dst_path))
    except OSError:
        shutil.move(str(tmp_path), str(dst_path))


class ShardWriter:
    """Write an ``ExpansionPlan`` to safetensor files.

    The writer performs a two-pass streaming expansion:

    Pass 1  Read only the binary JSON headers of each source shard to compute
            exact output tensor sizes and assign tensors to output shards.
    Pass 2  Open only the needed source shards via mmap, apply recipes, and
            write output shards — keeping at most one output shard in RAM.
    """

    def __init__(
        self,
        src_index: ShardIndex,
        dst_dir: Path,
        plan: ExpansionPlan,
        target_shard_bytes: int,
        verbose: bool,
        workers: int = 1,
        resume: bool = False,
    ) -> None:
        self.src_index = src_index
        self.dst_dir = dst_dir
        self.plan = plan
        self.target_shard_bytes = target_shard_bytes
        self.verbose = verbose
        self.workers = workers
        self.resume = resume
        self.weight_map: dict[str, str] = {}

    def write(self, src_dir: Path) -> None:
        """Execute the full write pipeline."""
        self.dst_dir.mkdir(parents=True, exist_ok=True)
        shard_groups = self._assign_shard_groups()
        self._write_shards(shard_groups)
        self._write_config(src_dir)

    def write_and_validate(self, src_dir: Path) -> None:
        """Execute the full write pipeline including post-write validation."""
        self.write(src_dir)
        self._validate_output()

    def _assign_shard_groups(self) -> list[list[str]]:
        """Pass 1: scan source headers and assign output tensors to shards."""
        src_headers: dict[str, dict[str, tuple[str, list[int]]]] = {}
        for sf in self.src_index.shard_files:
            src_headers[sf] = read_safetensors_header(self.src_index.model_dir / sf)

        sorted_keys = sorted(self.plan.recipes.keys())
        self.plan.validate_keys(set(self.src_index.all_keys), strict=False)
        shard_groups: list[list[str]] = [[]]
        current_bytes = 0
        for new_key in sorted_keys:
            recipe = self.plan.recipes[new_key]
            src_meta = src_headers[recipe.src_shard].get(recipe.src_key)
            if src_meta is None:
                raise KeyError(
                    f"Recipe for '{new_key}' references missing source tensor "
                    f"'{recipe.src_key}' in shard '{recipe.src_shard}'. "
                    "This indicates a corrupted or incorrectly built expansion plan."
                )
            out_bytes = predict_recipe_bytes(src_meta, recipe)
            if (
                current_bytes + out_bytes > self.target_shard_bytes
                and current_bytes > 0
            ):
                shard_groups.append([])
                current_bytes = 0
            shard_groups[-1].append(new_key)
            current_bytes += out_bytes

        if self.verbose:
            logger.info(f"  Pass 1 done: {len(shard_groups)} output shard(s) planned")

        # Keep source headers for resume validation (output shape/dtype must be
        # computed from source metadata, not from the existing output header).
        self._src_headers = src_headers
        return shard_groups

    def _write_shards(self, shard_groups: list[list[str]]) -> None:
        """Pass 2: write output shards and the weight-map index."""
        n = len(shard_groups)

        if self.workers > 1:
            self._write_shards_parallel(shard_groups)
        else:
            self._write_shards_serial(shard_groups)

        dst_index = ShardIndex(self.dst_dir, self.weight_map)
        if n > 1:
            dst_index.write_index_json(self.dst_dir)

    def _write_shards_serial(self, shard_groups: list[list[str]]) -> None:
        """Pass 2 (serial): write output shards one at a time."""
        from safetensors import safe_open

        n = len(shard_groups)
        src_handles: dict[str, Any] = {}

        # Precompute the last output-shard index that needs each source shard so
        # we can eagerly close source handles and release mmap references.
        last_use: dict[str, int] = {}
        for group_idx, group_keys in enumerate(shard_groups):
            for new_key in group_keys:
                recipe = self.plan.recipes[new_key]
                if recipe.src_shard:
                    last_use[recipe.src_shard] = group_idx
                if recipe.interp_src_shard:
                    last_use[recipe.interp_src_shard] = group_idx

        def _get_src_handle(shard_name: str) -> Any:
            if shard_name not in src_handles:
                src_handles[shard_name] = safe_open(
                    str(self.src_index.model_dir / shard_name),
                    framework="pt",
                    device="cpu",
                )
            return src_handles[shard_name]

        for i, group_keys in enumerate(shard_groups):
            shard_name = (
                "model.safetensors"
                if n == 1
                else f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            )
            shard_path = self.dst_dir / shard_name
            if self.resume and shard_path.exists():
                header = read_safetensors_header(shard_path)
                if all(k in header for k in group_keys) and all(
                    self._recipe_matches_header(
                        self.plan.recipes[k],
                        header[k],
                        self._src_headers,
                    )
                    for k in group_keys
                ):
                    for k in group_keys:
                        self.weight_map[k] = shard_name
                    if self.verbose:
                        logger.info(
                            f"  skipped {shard_name} (already exists, "
                            f"{len(group_keys)} tensors)"
                        )
                    continue

            tensors: dict[str, torch.Tensor] = {}
            for new_key in group_keys:
                recipe = self.plan.recipes[new_key]
                if recipe.create_shape:
                    src_t = None
                else:
                    src_t = _get_src_handle(recipe.src_shard).get_tensor(
                        recipe.src_key
                    )
                interp_t = None
                if recipe.interp_src_key:
                    interp_t = _get_src_handle(recipe.interp_src_shard).get_tensor(
                        recipe.interp_src_key
                    )
                tensors[new_key] = apply_recipe(src_t, recipe, interp_t)
                self.weight_map[new_key] = shard_name
            # Atomic write: save to temp file then rename.
            tmp_path = self.dst_dir / f"{shard_name}.tmp"
            save_file(tensors, str(tmp_path))
            _atomic_replace(tmp_path, shard_path)
            if self.verbose:
                size_mb = shard_path.stat().st_size / 1e6
                logger.info(
                    f"  wrote {shard_name}  "
                    f"({len(tensors)} tensors, {size_mb:.0f} MB)"
                )

            # Eagerly close source shards that are no longer needed by future groups.
            for shard_name_h in list(src_handles.keys()):
                if last_use.get(shard_name_h, -1) == i:
                    _safe_close_handle(src_handles[shard_name_h])
                    del src_handles[shard_name_h]

        for handle in src_handles.values():
            _safe_close_handle(handle)
        src_handles.clear()

    def _write_shards_parallel(self, shard_groups: list[list[str]]) -> None:
        """Pass 2 (parallel): write output shards using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor

        from tqdm import tqdm

        n = len(shard_groups)
        tasks = []
        for i, group_keys in enumerate(shard_groups):
            shard_name = (
                "model.safetensors"
                if n == 1
                else f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            )
            shard_path = self.dst_dir / shard_name

            # Pre-compute expected output header entries from source metadata so
            # the worker can validate an existing shard on resume.
            expected_header: dict[str, tuple[str, list[int]]] = {}
            for new_key in group_keys:
                r = self.plan.recipes[new_key]
                if r.create_shape:
                    expected_header[new_key] = (
                        r.create_dtype,
                        list(r.create_shape),
                    )
                else:
                    src_dtype, src_shape = self._src_headers[r.src_shard][r.src_key]
                    expected_header[new_key] = (
                        r.output_dtype(src_dtype),
                        r.output_shape(src_shape),
                    )

            # Gather (src_shard_path, src_key, out_key, recipe) per group.
            # TensorRecipe dataclass objects are passed directly so that
            # adding a new field does not break the tuple unpacking in the
            # worker thread.  interp_src_shard paths are resolved to absolute
            # paths here so the worker can open them directly.
            items = []
            for new_key in group_keys:
                r = self.plan.recipes[new_key]
                src_path = (
                    str(self.src_index.model_dir / r.src_shard)
                    if r.src_shard
                    else ""
                )
                # Shallow-copy the recipe so we can resolve interp paths
                # without mutating the original plan.
                rr = copy.copy(r)
                if rr.interp_src_shard:
                    rr.interp_src_shard = str(
                        self.src_index.model_dir / rr.interp_src_shard
                    )
                items.append((src_path, rr.src_key, new_key, rr))
            tasks.append(
                (
                    str(shard_path),
                    items,
                    set(group_keys),
                    self.resume,
                    expected_header,
                )
            )

        chunksize = max(1, len(tasks) // self.workers)
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            results = list(
                tqdm(
                    executor.map(worker_write_shard, tasks, chunksize=chunksize),
                    total=len(tasks),
                    desc="Writing shards",
                    disable=not self.verbose,
                )
            )

        for shard_name, tensor_names in results:
            for name in tensor_names:
                self.weight_map[name] = shard_name

    @staticmethod
    def _recipe_matches_header(
        recipe: TensorRecipe,
        output_entry: tuple[str, list[int]],
        src_headers: dict[str, dict[str, tuple[str, list[int]]]],
    ) -> bool:
        """Return True if an existing output header entry matches the recipe.

        The expected output dtype/shape is computed from the *source* tensor
        metadata, not from the output header itself.  This matters for recipes
        that change shape (padding, dup_rows, create_shape, etc.).
        """
        out_dtype, out_shape = output_entry
        if recipe.create_shape:
            expected_dtype = recipe.create_dtype
            expected_shape = list(recipe.create_shape)
        else:
            src_dtype, src_shape = src_headers[recipe.src_shard][recipe.src_key]
            expected_dtype = recipe.output_dtype(src_dtype)
            expected_shape = recipe.output_shape(src_shape)
        return out_dtype == expected_dtype and out_shape == expected_shape

    def _write_config(self, src_dir: Path) -> None:
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
        if self.plan.new_num_hidden_layers > 0:
            layers_key = next(
                (k for k in ("num_hidden_layers", "num_layers") if k in cfg),
                "num_hidden_layers",
            )
            cfg[layers_key] = self.plan.new_num_hidden_layers

        # ── apply expansion-specific patches (expert count, topk, etc.) ───────
        cfg.update(self.plan.config_patches)

        cfg_path = self.dst_dir / "config.json"
        tmp_path = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        _atomic_replace(tmp_path, cfg_path)

        # ── verify auto_map referenced files are present ──────────────────────
        auto_map: dict[str, str] = cfg.get("auto_map", {})
        missing: list[str] = []
        for _cls, ref in auto_map.items():
            # ref format: "module_name.ClassName"
            module_name = ref.split(".")[0] + ".py"
            if not (self.dst_dir / module_name).exists():
                missing.append(module_name)
        if missing:
            logger.warning(
                "auto_map references missing files in %s: %s. "
                "Run expand() (not dry_run) to copy them automatically.",
                self.dst_dir,
                missing,
            )

    def _validate_output(self) -> None:
        """Lightweight post-write validation.

        Verifies that ``config.json`` is valid JSON, every tensor key in the
        output weight map exists in its shard, and ``auto_map`` referenced
        Python files are present.  This does **not** load model weights.
        """
        cfg_path = self.dst_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Output validation failed: missing {cfg_path}")
        try:
            with open(cfg_path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Output validation failed: invalid config.json: {e}"
            ) from e

        dst_index = ShardIndex.load(self.dst_dir)
        missing_tensors: list[str] = []
        for key, shard_name in dst_index.weight_map.items():
            shard_path = self.dst_dir / shard_name
            if not shard_path.exists():
                missing_tensors.append(f"{key} -> {shard_name}")
                continue
            header = read_safetensors_header(shard_path)
            if key not in header:
                missing_tensors.append(f"{key} in {shard_name}")
        if missing_tensors:
            sample = missing_tensors[:5]
            raise RuntimeError(
                f"Output validation failed: {len(missing_tensors)} tensor(s) missing, "
                f"e.g. {sample}"
            )

        cfg = peek_model_config(self.dst_dir)
        auto_map: dict[str, str] = cfg.get("auto_map", {})
        missing_py: list[str] = []
        for _cls, ref in auto_map.items():
            module_name = ref.split(".")[0] + ".py"
            if not (self.dst_dir / module_name).exists():
                missing_py.append(module_name)
        if missing_py:
            raise RuntimeError(
                f"Output validation failed: missing auto_map files: {missing_py}"
            )

        logger.info(
            "Output validation passed: %d tensors in %d shard(s)",
            len(dst_index.all_keys),
            len(dst_index.shard_files),
        )
