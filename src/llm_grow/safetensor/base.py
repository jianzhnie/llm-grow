"""Base class for safetensor-level model expansion.

Architecture
------------
This is the **safetensor-level** expansion layer of llm-grow.  It operates
directly on serialized ``.safetensors`` files without instantiating any
``nn.Module``, making it suitable for 100B+ parameter models that cannot
fit in RAM.

For models that can be loaded in memory, use the in-memory expanders at
``llm_grow.expanders.base.AbstractExpander`` instead — they offer richer
integration with training, verification, and optimizer workflows.

Both layers share the same mathematical algorithms (ZeroBlockInsert identity
insertion, expert duplication, etc.) but operate at different abstraction
levels:

  +-----------------------+----------------------------+
  | In-Memory (expanders) | Safetensor (this module)   |
  +-----------------------+----------------------------+
  | nn.Module in/out      | .safetensors dir in/out    |
  | Peak = full model     | Peak ≈ 1 output shard      |
  | FP verify: yes        | FP verify: structural only |
  | Training-ready        | Needs reload for training  |
  +-----------------------+----------------------------+

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
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from llm_grow.configs.constants import DEFAULT_TARGET_SHARD_BYTES
from llm_grow.safetensor.recipe import ExpansionPlan, TensorRecipe
from llm_grow.safetensor.utils import (
    ShardIndex,
    parse_layer_idx,
    peek_model_config,
    read_safetensors_header,
)
from llm_grow.safetensor.writer import (
    apply_recipe as _apply_recipe,
)
from llm_grow.safetensor.writer import (
    predict_recipe_bytes as _predict_recipe_bytes,
)
from llm_grow.safetensor.writer import (
    worker_write_shard as _worker_write_shard,
)
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)

# Re-exported from recipe.py for backward compatibility.
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
    DEFAULT_TARGET_SHARD_BYTES: int = DEFAULT_TARGET_SHARD_BYTES

    # ── public API ────────────────────────────────────────────────────────────

    def expand(
        self,
        src_dir: str | Path,
        dst_dir: str | Path,
        *,
        target_shard_bytes: int | None = None,
        workers: int = 1,
        verbose: bool = True,
        validate_output: bool = False,
        resume: bool = False,
    ) -> None:
        """Expand model at ``src_dir``, write results to ``dst_dir``.

        Args:
            target_shard_bytes: Maximum bytes per output shard.  Defaults to
                ``auto_detect_shard_size`` (mean of input shards, or 4 GB).
            workers: Number of parallel writer processes (1 = serial).
            validate_output: If True, run a lightweight post-write check that
                ``config.json`` is valid and every tensor key exists in its shard.
            resume: If True, skip output shards that already exist and contain
                all expected tensor keys.  Useful for resuming interrupted runs.
        """
        src_dir, dst_dir = Path(src_dir), Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        src_index = ShardIndex.load(src_dir)

        if target_shard_bytes is None:
            from llm_grow.safetensor.utils import auto_detect_shard_size

            target_shard_bytes = auto_detect_shard_size(src_dir, src_index.shard_files)

        if verbose:
            logger.info(f"{type(self).__name__}  {src_dir} → {dst_dir}")
            logger.info(
                f"  source: {len(src_index.shard_files)} shard(s), "
                f"{src_index.total_size_bytes() / 1e9:.2f} GB, "
                f"{len(src_index.all_keys)} tensors"
            )

        plan = self._build_plan(src_index)

        if verbose:
            logger.info(
                f"  plan: {len(plan.recipes)} output tensors, "
                f"new num_hidden_layers={plan.new_num_hidden_layers}"
            )

        self._write_shards(
            src_index,
            dst_dir,
            plan,
            target_shard_bytes,
            verbose,
            workers,
            resume=resume,
        )
        self._write_config(src_dir, dst_dir, plan)
        src_index.copy_non_weight_files(dst_dir)

        if validate_output:
            self._validate_output(dst_dir, plan)

        if verbose:
            logger.info(f"  Done → {dst_dir}")

    # ── abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        """Build the full tensor-level expansion plan."""

    def dry_run(self, src_dir: str | Path) -> ExpansionPlan:
        """Build and print the expansion plan without loading any tensor data."""
        src_dir = Path(src_dir)
        src_index = ShardIndex.load(src_dir)
        plan = self._build_plan(src_index)

        logger.info(f"[dry_run] {type(self).__name__}  src={src_dir}")
        logger.info(
            f"  source:  {len(src_index.all_keys)} tensors, "
            f"{len(src_index.shard_files)} shard(s)"
        )
        logger.info(
            f"  output:  {len(plan.recipes)} tensors, "
            f"num_hidden_layers → {plan.new_num_hidden_layers}"
        )
        logger.info(f"  config patches: {plan.config_patches}")

        zero_keys = [k for k, r in plan.recipes.items() if r.zero_out]
        dup_keys = [k for k, r in plan.recipes.items() if r.dup_rows]
        padded_keys = [
            k
            for k, r in plan.recipes.items()
            if (r.pad_rows or r.pad_cols) and not r.zero_out
        ]
        new_keys = [k for k in plan.recipes if k not in src_index.weight_map]

        logger.info(f"  zero-out tensors : {len(zero_keys)}")
        logger.info(f"  dup-rows tensors : {len(dup_keys)}")
        logger.info(f"  padded tensors   : {len(padded_keys)}")
        logger.info(f"  brand-new keys   : {len(new_keys)}")
        if new_keys[:3]:
            logger.info(f"    sample: {new_keys[:3]}")
        return plan

    def _write_shards(
        self,
        src_index: ShardIndex,
        dst_dir: Path,
        plan: ExpansionPlan,
        target_bytes: int,
        verbose: bool,
        workers: int = 1,
        resume: bool = False,
    ) -> None:
        """Two-pass shard writer.

        Pass 1  Read only the binary JSON headers of each source shard
                (O(header size), no tensor data touched) to compute exact
                output tensor sizes and assign tensors to output shards.

        Pass 2  For each output shard, open only the needed source shards
                (via mmap), apply recipes, and write directly to the final
                shard name — no temporary files, no renaming.

        Args:
            resume: If True, skip shards whose final files already exist and
                whose headers contain all expected tensor keys.
        """
        from concurrent.futures import ThreadPoolExecutor

        from llm_grow.safetensor.utils import read_safetensors_header

        # ── Pass 1: header scan → shard assignment ────────────────────────────
        with ThreadPoolExecutor(max_workers=min(8, len(src_index.shard_files))) as pool:
            src_headers: dict[str, dict[str, tuple[str, list[int]]]] = dict(
                pool.map(
                    lambda sf: (sf, read_safetensors_header(src_index.model_dir / sf)),
                    src_index.shard_files,
                )
            )

        sorted_keys = sorted(plan.recipes.keys())
        shard_groups: list[list[str]] = [[]]
        current_bytes = 0
        for new_key in sorted_keys:
            recipe = plan.recipes[new_key]
            src_meta = src_headers[recipe.src_shard].get(recipe.src_key)
            if src_meta is None:
                raise KeyError(
                    f"Recipe for '{new_key}' references missing source tensor "
                    f"'{recipe.src_key}' in shard '{recipe.src_shard}'. "
                    "This indicates a corrupted or incorrectly built expansion plan."
                )
            out_bytes = _predict_recipe_bytes(src_meta, recipe)
            if current_bytes + out_bytes > target_bytes and current_bytes > 0:
                shard_groups.append([])
                current_bytes = 0
            shard_groups[-1].append(new_key)
            current_bytes += out_bytes

        n = len(shard_groups)
        weight_map: dict[str, str] = {}

        if verbose:
            logger.info(f"  Pass 1 done: {n} output shard(s) planned")

        # ── Pass 2: write output shards ───────────────────────────────────────
        if workers > 1:
            self._write_shards_parallel(
                src_index,
                dst_dir,
                plan,
                shard_groups,
                weight_map,
                verbose,
                workers,
                resume=resume,
            )
        else:
            from safetensors import safe_open

            src_handles: dict[str, Any] = {}

            def _get_src_handle(shard_name: str) -> Any:
                if shard_name not in src_handles:
                    src_handles[shard_name] = safe_open(
                        str(src_index.model_dir / shard_name),
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
                shard_path = dst_dir / shard_name
                if resume and shard_path.exists():
                    header = read_safetensors_header(shard_path)
                    if all(k in header for k in group_keys) and all(
                        self._recipe_matches_header(plan.recipes[k], header[k])
                        for k in group_keys
                    ):
                        for k in group_keys:
                            weight_map[k] = shard_name
                        if verbose:
                            logger.info(
                                f"  skipped {shard_name} (already exists, "
                                f"{len(group_keys)} tensors)"
                            )
                        continue

                tensors: dict[str, torch.Tensor] = {}
                for new_key in group_keys:
                    recipe = plan.recipes[new_key]
                    if recipe.create_shape:
                        src_t = torch.zeros(1)
                    else:
                        src_t = _get_src_handle(recipe.src_shard).get_tensor(
                            recipe.src_key
                        )
                    interp_t = None
                    if recipe.interp_src_key:
                        interp_t = _get_src_handle(recipe.interp_src_shard).get_tensor(
                            recipe.interp_src_key
                        )
                    tensors[new_key] = _apply_recipe(src_t, recipe, interp_t)
                    weight_map[new_key] = shard_name
                # Atomic write: save to temp file then rename.
                tmp_path = dst_dir / f"{shard_name}.tmp"
                save_file(tensors, str(tmp_path))
                os.replace(str(tmp_path), str(shard_path))
                if verbose:
                    size_mb = shard_path.stat().st_size / 1e6
                    logger.info(
                        f"  wrote {shard_name}  "
                        f"({len(tensors)} tensors, {size_mb:.0f} MB)"
                    )

            del src_handles

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
        resume: bool = False,
    ) -> None:
        """Write output shards in parallel using ProcessPoolExecutor."""
        from concurrent.futures import ProcessPoolExecutor

        from tqdm import tqdm

        n = len(shard_groups)
        tasks = []
        for i, group_keys in enumerate(shard_groups):
            shard_name = (
                "model.safetensors"
                if n == 1
                else f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            )
            # Gather (src_shard_path, src_key, out_key, recipe_tuple) per group
            items = []
            for new_key in group_keys:
                r = plan.recipes[new_key]
                items.append(
                    (
                        str(src_index.model_dir / r.src_shard) if r.src_shard else "",
                        r.src_key,
                        new_key,
                        (
                            r.zero_out,
                            r.pad_rows,
                            r.pad_cols,
                            r.dup_rows,
                            r.dup_rows_noise_scale,
                            r.router_split,
                            str(src_index.model_dir / r.interp_src_shard)
                            if r.interp_src_shard
                            else "",
                            r.interp_src_key,
                            r.interp_alpha,
                            r.add_noise_std,
                            tuple(r.create_shape) if r.create_shape else (),
                            r.create_dtype,
                        ),
                    )
                )
            tasks.append((str(dst_dir / shard_name), items, set(group_keys), resume))

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

    @staticmethod
    def _recipe_matches_header(
        recipe: TensorRecipe, header_entry: tuple[str, list[int]]
    ) -> bool:
        """Return True if a header entry matches the recipe's expected output."""
        dtype, shape = header_entry
        expected_dtype = recipe.output_dtype(dtype)
        expected_shape = recipe.output_shape(shape)
        return dtype == expected_dtype and shape == expected_shape

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
            logger.warning(
                "auto_map references missing files in %s: %s. "
                "Run expand() (not dry_run) to copy them automatically.",
                dst_dir,
                missing,
            )

    def _validate_output(self, dst_dir: Path, plan: ExpansionPlan) -> None:
        """Lightweight post-write validation.

        Verifies that ``config.json`` is valid JSON, every tensor key in the
        output weight map exists in its shard, and ``auto_map`` referenced
        Python files are present.  This does **not** load model weights.
        """
        cfg_path = dst_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Output validation failed: missing {cfg_path}")
        try:
            with open(cfg_path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Output validation failed: invalid config.json: {e}"
            ) from e

        dst_index = ShardIndex.load(dst_dir)
        missing_tensors: list[str] = []
        for key, shard_name in dst_index.weight_map.items():
            shard_path = dst_dir / shard_name
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

        cfg = peek_model_config(dst_dir)
        auto_map: dict[str, str] = cfg.get("auto_map", {})
        missing_py: list[str] = []
        for _cls, ref in auto_map.items():
            module_name = ref.split(".")[0] + ".py"
            if not (dst_dir / module_name).exists():
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

    # ── shared plan-building helpers ──────────────────────────────────────────

    @staticmethod
    def _passthrough_non_layer_keys(plan: ExpansionPlan, wmap: dict[str, str]) -> None:
        """Add passthrough recipes for all non-layer tensors (embed, norm, etc.)."""
        for key, shard in wmap.items():
            if parse_layer_idx(key) is None:
                plan.passthrough(key, shard)

    def _should_zero(self, suf: str) -> bool:
        """Determine if a tensor suffix should be zeroed in an identity block.

        Default implementation checks against ``IDENTITY_ZERO_SUFFIXES``.
        Subclasses (MoE, LongCat) override for architecture-specific logic.
        """
        return suf in self.IDENTITY_ZERO_SUFFIXES

    @staticmethod
    def _should_zero_moe(
        suf: str,
        *,
        zero_suffixes: frozenset[str] | set[str],
        zero_expert_down: bool = True,
        zero_shared_expert_down: bool = True,
    ) -> bool:
        """Shared helper for MoE identity-block zeroing logic.

        Reusable by subclass ``_should_zero`` overrides to avoid duplication.
        """
        if suf in zero_suffixes:
            return True
        if suf.endswith(".down_proj.weight") and "mlp.experts." in suf:
            return zero_expert_down
        return bool(
            zero_shared_expert_down and suf == "mlp.shared_experts.down_proj.weight"
        )

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
                            is_identity=True → zero suffixes per ``_should_zero()``.
        """
        plan = ExpansionPlan(new_num_hidden_layers=len(layer_sequence))
        wmap = src_index.weight_map
        suffixes = src_index.layer_suffixes()

        for new_idx, (src_idx, is_identity) in enumerate(layer_sequence):
            for suf in suffixes:
                src_key = f"model.layers.{src_idx}.{suf}"
                if src_key not in wmap:
                    continue
                new_key = f"model.layers.{new_idx}.{suf}"
                zero = is_identity and self._should_zero(suf)
                plan.add(
                    new_key,
                    TensorRecipe(
                        src_shard=wmap[src_key],
                        src_key=src_key,
                        zero_out=zero,
                    ),
                )

        self._passthrough_non_layer_keys(plan, wmap)

        return plan
