"""Base class for safetensor-level model expansion.

Architecture
------------
This is the **plan-building** layer of safetensor expansion.  It operates
on ``ShardIndex`` metadata and produces an ``ExpansionPlan`` describing how
to derive every output tensor from a source tensor.

The actual I/O (header scanning, shard writing, config serialization, and
post-write validation) lives in ``llm_grow.safetensor.shard_writer.ShardWriter``.
This split keeps subclasses focused on the mathematical transformation while
allowing the write pipeline to evolve independently.

For models that can be loaded in memory, use the in-memory expanders at
``llm_grow.expanders.base.AbstractExpander`` instead.

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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from pathlib import Path

from llm_grow.configs.constants import DEFAULT_TARGET_SHARD_BYTES
from llm_grow.safetensor.recipe import ExpansionPlan, TensorRecipe
from llm_grow.safetensor.shard_writer import ShardWriter
from llm_grow.safetensor.utils import ShardIndex, parse_layer_idx
from llm_grow.safetensor.writer import apply_recipe as _apply_recipe
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "ExpansionPlan",
    "SafetensorExpanderBase",
    "TensorRecipe",
    "_apply_recipe",
]


class SafetensorExpanderBase(ABC):
    """Expand a HuggingFace safetensor model without loading it fully into RAM.

    Subclasses override ``_build_plan`` to describe the transformation;
    ``ShardWriter`` handles all I/O and shard management.
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
            workers: Number of parallel writer threads (1 = serial).
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

        writer = ShardWriter(
            src_index=src_index,
            dst_dir=dst_dir,
            plan=plan,
            target_shard_bytes=target_shard_bytes,
            verbose=verbose,
            workers=workers,
            resume=resume,
        )
        writer.write(src_dir)

        # Copy auxiliary files BEFORE validation so auto_map Python files
        # (configuration_*.py, modeling_*.py) are present for the check.
        src_index.copy_non_weight_files(dst_dir)

        if validate_output:
            writer._validate_output()

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
        zero_suffixes: Collection[str],
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
