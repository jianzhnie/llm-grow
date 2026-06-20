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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from llm_grow.safetensor.utils import (
    ShardIndex,
    parse_layer_idx,
    peek_model_config,
    read_safetensors_header,
)
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}

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

    # ── interpolation (SVDInterpInsert) ───────────────────────────────────────
    # When set, output = interp_alpha * src + (1 - interp_alpha) * interp_src
    interp_src_shard: str = ""
    interp_src_key: str = ""
    interp_alpha: float = 0.5

    # ── noise injection (DenseToMoE expert copies) ────────────────────────────
    add_noise_std: float = 0.0  # Gaussian noise std to add (0 = disabled)

    # ── create new zero tensor (e.g. MoE router weights) ─────────────────────
    create_shape: tuple = ()  # non-empty → ignore src, create zero tensor
    create_dtype: str = "F32"  # safetensors dtype string for created tensor


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

    # ── serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize plan to a JSON-compatible dict."""
        from dataclasses import asdict

        return {
            "new_num_hidden_layers": self.new_num_hidden_layers,
            "config_patches": self.config_patches,
            "recipes": {k: asdict(r) for k, r in self.recipes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpansionPlan:
        """Deserialize plan from a dict (as produced by ``to_dict``)."""
        plan = cls(
            new_num_hidden_layers=data["new_num_hidden_layers"],
            config_patches=data.get("config_patches", {}),
        )
        for key, recipe_data in data.get("recipes", {}).items():
            plan.add(key, TensorRecipe(**recipe_data))
        return plan

    def save_json(self, path: str | Path) -> None:
        """Save plan to a JSON file for offline review or resume."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Expansion plan saved to %s (%d recipes)", path, len(self.recipes))

    @classmethod
    def load_json(cls, path: str | Path) -> ExpansionPlan:
        """Load plan from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        plan = cls.from_dict(data)
        logger.info(
            "Expansion plan loaded from %s (%d recipes)", path, len(plan.recipes)
        )
        return plan


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
                    if all(k in header for k in group_keys):
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
                tmp_path.replace(shard_path)
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
            tasks.append(
                (str(dst_dir / shard_name), items, set(group_keys), resume)
            )

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
        self._passthrough_non_layer_keys(plan, wmap)

        return plan


# ── tensor transform ──────────────────────────────────────────────────────────


def _apply_recipe(
    src: torch.Tensor,
    recipe: TensorRecipe,
    interp_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    # ── create new zero tensor (router weights etc.) ─────────────────────────
    if recipe.create_shape:
        dtype = _TORCH_DTYPES.get(recipe.create_dtype, torch.float32)
        return torch.zeros(recipe.create_shape, dtype=dtype)

    # ── zero_out can be applied directly without cloning the mmap-backed src ───
    if recipe.zero_out:
        return torch.zeros_like(src)

    # ── interpolation: alpha * src + (1 - alpha) * interp_tensor ─────────────
    if interp_tensor is not None and recipe.interp_src_key:
        alpha = recipe.interp_alpha
        return (alpha * src.float() + (1 - alpha) * interp_tensor.float()).to(src.dtype)

    # ── dup_rows with router_split: handle real/zero expert rows separately ────
    # rows [0 : router_split]   → real experts → duplicate WITH noise
    # rows [router_split : end] → zero experts → duplicate WITHOUT noise
    if recipe.dup_rows and recipe.router_split > 0:
        real = src[: recipe.router_split]
        zeros = src[recipe.router_split :]
        noise = (
            torch.randn_like(real) * recipe.dup_rows_noise_scale * real.float().std()
        )
        real_dup = real + noise.to(real.dtype)
        return torch.cat([real, real_dup, zeros, zeros.clone()], dim=0)

    # ── dup_rows without split: duplicate all rows uniformly ──────────────────
    if recipe.dup_rows:
        noise = torch.randn_like(src) * recipe.dup_rows_noise_scale * src.float().std()
        dup = src + noise.to(src.dtype)
        return torch.cat([src, dup], dim=0)

    # ── pad then optionally add noise ─────────────────────────────────────────
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
        # Passthrough tensors must be cloned because mmap-backed views share
        # storage and safetensors' save_file rejects non-owning memory.
        t = src.clone()

    if recipe.add_noise_std > 0:
        noise = torch.randn_like(t) * recipe.add_noise_std
        return t + noise

    return t


def _predict_recipe_bytes(src_meta: tuple[str, list[int]], recipe: TensorRecipe) -> int:
    """Predict output tensor byte size from header metadata (no tensor load).

    Used in Pass 1 of ``_write_shards`` to plan shard assignments purely from
    the binary JSON header — no mmap or tensor data needed.
    """
    from llm_grow.safetensor.utils import DTYPE_SIZES, nbytes_from_header

    if recipe.create_shape:
        elem = DTYPE_SIZES.get(recipe.create_dtype, 4)
        numel = 1
        for d in recipe.create_shape:
            numel *= d
        return elem * numel

    dtype, shape = src_meta
    elem = DTYPE_SIZES.get(dtype, 4)
    src_bytes = nbytes_from_header(dtype, shape)

    if recipe.zero_out:
        return src_bytes  # same shape, just zeroed
    if recipe.dup_rows:
        # dup_rows doubles the row count; with router_split the total is still 2x
        return src_bytes * 2
    if recipe.pad_rows > 0 or recipe.pad_cols > 0:
        ndim = len(shape)
        if ndim == 2:
            return elem * (shape[0] + recipe.pad_rows) * (shape[1] + recipe.pad_cols)
        if ndim == 1:
            return elem * (shape[0] + recipe.pad_rows)
    return src_bytes


# ── module-level worker for ProcessPoolExecutor (must be picklable) ───────────


def _worker_write_shard(
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
    from safetensors.torch import save_file

    out_path_str, items, expected_keys, resume = args
    out_path = Path(out_path_str)

    if resume and out_path.exists():
        header = read_safetensors_header(out_path)
        if expected_keys.issubset(header):
            return (out_path.name, list(expected_keys))

    # Group by source shard path to open each file only once
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
                t = torch.zeros(1) if create_shape else sf.get_tensor(src_key)
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
                tensors[out_key] = _apply_recipe(t, recipe, interp_t)

    # Atomic write in worker process.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    save_file(tensors, str(tmp_path))
    tmp_path.replace(out_path)
    return (out_path.name, list(tensors.keys()))
