"""Data classes describing safetensor expansion plans and tensor recipes.

Keeping these types in a dedicated module breaks the circular import between
``safetensor/base.py`` (plan logic) and ``safetensor/writer.py`` (tensor
transforms / workers).  Both modules import from here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


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
    create_shape: tuple[int, ...] = ()  # non-empty → ignore src, create zero tensor
    create_dtype: str = "F32"  # safetensors dtype string for created tensor

    _VALID_FLAGS: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "zero_out",
                "dup_rows",
                "pad_rows",
                "pad_cols",
                "interp_src_key",
                "create_shape",
                "add_noise_std",
            }
        ),
        repr=False,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.router_split > 0 and not self.dup_rows:
            raise ValueError(
                f"router_split ({self.router_split}) requires dup_rows=True"
            )

        has_create = bool(self.create_shape)
        has_zero = self.zero_out
        has_dup = self.dup_rows
        has_pad = self.pad_rows > 0 or self.pad_cols > 0
        has_interp = bool(self.interp_src_key)
        has_noise = self.add_noise_std > 0

        # create_shape produces a brand-new tensor; it cannot combine with any
        # source-based transform.
        if has_create and (has_zero or has_dup or has_pad or has_interp or has_noise):
            raise ValueError(
                "TensorRecipe create_shape is mutually exclusive with "
                "zero_out, dup_rows, padding, interpolation, and add_noise_std"
            )

        # These primary ops cannot combine with each other.  zero_out + padding
        # is intentionally allowed for width-expanded identity blocks.
        primary: list[str] = []
        if has_dup:
            primary.append("dup_rows")
        if has_interp:
            primary.append("interpolation")
        if has_noise:
            primary.append("add_noise_std")
        if len(primary) > 1:
            raise ValueError(
                "TensorRecipe primary transformation flags are mutually exclusive, "
                f"but multiple were set: {primary}"
            )

        # Primary ops are also exclusive with zero_out/padding.
        if primary and (has_zero or has_pad):
            active = primary + (
                ["zero_out"] if has_zero else []
            ) + (["padding"] if has_pad else [])
            raise ValueError(
                "TensorRecipe transformation flags are mutually exclusive, "
                f"but multiple were set: {active}"
            )

    def output_shape(self, src_shape: list[int]) -> list[int]:
        """Return the output tensor shape produced by this recipe.

        Args:
            src_shape: Shape of the source tensor (from safetensors header).
        """
        if self.create_shape:
            return list(self.create_shape)
        if self.zero_out or self.interp_src_key or self.add_noise_std > 0:
            return list(src_shape)
        if self.dup_rows:
            return [src_shape[0] * 2, *list(src_shape[1:])]
        if self.pad_rows > 0 or self.pad_cols > 0:
            if len(src_shape) == 2:
                return [src_shape[0] + self.pad_rows, src_shape[1] + self.pad_cols]
            if len(src_shape) == 1:
                return [src_shape[0] + self.pad_rows]
        return list(src_shape)

    def output_dtype(self, src_dtype: str) -> str:
        """Return the output safetensors dtype string produced by this recipe."""
        if self.create_shape:
            return self.create_dtype
        return src_dtype

    def _validate_source(self) -> None:
        """Validate that non-create recipes reference a source tensor."""
        if self.create_shape:
            return
        if not self.src_shard or not self.src_key:
            raise ValueError(
                "TensorRecipe for a non-create operation must have both "
                f"src_shard and src_key set (got src_shard={self.src_shard!r}, "
                f"src_key={self.src_key!r})"
            )


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
        return {
            "new_num_hidden_layers": self.new_num_hidden_layers,
            "config_patches": self.config_patches,
            "recipes": {
                k: {
                    fk: fv
                    for fk, fv in asdict(r).items()
                    if not fk.startswith("_")
                }
                for k, r in self.recipes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpansionPlan:
        """Deserialize plan from a dict (as produced by ``to_dict``)."""
        valid_plan_keys = {"new_num_hidden_layers", "config_patches", "recipes"}
        unknown = set(data) - valid_plan_keys
        if unknown:
            raise ValueError(f"Unknown ExpansionPlan keys: {sorted(unknown)}")

        plan = cls(
            new_num_hidden_layers=data.get("new_num_hidden_layers", 0),
            config_patches=data.get("config_patches", {}),
        )
        recipe_keys = {
            f.name
            for f in TensorRecipe.__dataclass_fields__.values()
            if not f.name.startswith("_")
        }
        for key, recipe_data in data.get("recipes", {}).items():
            if not isinstance(recipe_data, dict):
                raise ValueError(
                    f"Recipe for '{key}' must be a dict, got {type(recipe_data)}"
                )
            recipe_data = {
                k: v for k, v in recipe_data.items() if not k.startswith("_")
            }
            bad_keys = set(recipe_data) - recipe_keys
            if bad_keys:
                raise ValueError(
                    f"Unknown TensorRecipe keys for '{key}': {sorted(bad_keys)}"
                )
            recipe = TensorRecipe(**recipe_data)
            recipe._validate_source()
            plan.add(key, recipe)
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
