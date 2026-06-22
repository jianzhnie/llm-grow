"""Unified auto-expand entry point.

Detects model architecture and dispatches to the correct expander,
so callers never need to know whether a model is Dense or MoE.

Expansion axes
--------------
``method="depth"``
    Insert ZeroBlockInsert-style identity layers.

    *Dense* → ``ZeroBlockInsertSafetensorExpander``
               zeros: self_attn.o_proj + mlp.down_proj

    *MoE-standard* → ``GenericMoEDepthExpander``
               zeros: self_attn.o_proj + ALL mlp.experts.*.down_proj
               (+ shared_expert.down_proj + dense layer mlp.down_proj if mixed)

    *MoE-LongCat* → ``LongcatDepthExpander``
               zeros: self_attn.{0,1}.o_proj + ALL expert down_proj
                      + mlps.{0,1}.down_proj

``method="expert"``
    Copy existing experts to increase expert count.  MoE only.

    *Dense* → raises ``ValueError`` (no experts to copy)

    *Any MoE* → ``GenericMoEExpertCloneExpander``
               duplicates mlp.experts.{i}.* (including fp8 scale tensors)
               duplicates router weight rows (+ noise)
               duplicates router bias rows  (no noise)

``method="width"``
    Zero-pad FFN intermediate_size.  Dense only (MoE expert width not yet supported).

    *Dense* → ``MultiAxisPadSafetensorExpander``
    *MoE*   → raises ``NotImplementedError``

Key difference summary
----------------------
Dense identity block needs ONE zero proj:       ``mlp.down_proj.weight``
MoE   identity block needs N×experts zero prjs: ``mlp.experts.{0..N}.down_proj.weight``

If the wrong expander is used (e.g. ZeroBlockInsertSafetensor on a MoE model),
the identity block is incomplete: expert outputs are non-zero → function changes.
This module prevents that mistake via auto-detection.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_grow.configs.base import InsertStrategy
from llm_grow.safetensor.base import SafetensorExpanderBase
from llm_grow.safetensor.detect import ModelProfile, detect_model
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)

# ── expander registry ─────────────────────────────────────────────────────────

ExpanderFactory = Callable[..., SafetensorExpanderBase]
_EXPANDER_REGISTRY: dict[tuple[str, str], ExpanderFactory] = {}

# ── method → parameter names mapping ─────────────────────────────────────────
# When a new method is added, only this dict and the factory functions below
# need to be updated — the _build_expander dispatch logic stays unchanged.

_METHOD_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "depth": ("num_new_layers", "insert_strategy", "profile"),
    "expert": ("expand_factor", "noise_scale", "profile"),
    "width": ("num_new_layers", "insert_strategy", "ffn_size_expansion", "profile"),
}


def register_expander(
    method: str, family: str
) -> Callable[[ExpanderFactory], ExpanderFactory]:
    """Register a factory function for ``(method, family)``.

    Example::

        @register_expander("depth", "dense")
        def _build_dense_depth(profile, num_new_layers, insert_strategy):
            ...
    """

    def decorator(factory: ExpanderFactory) -> ExpanderFactory:
        _EXPANDER_REGISTRY[(method, family)] = factory
        return factory

    return decorator


def _available_families(method: str) -> list[str]:
    return sorted({f for m, f in _EXPANDER_REGISTRY if m == method})


# ── public API ────────────────────────────────────────────────────────────────


def auto_expand(
    src_dir: str | Path,
    dst_dir: str | Path,
    method: str,
    *,
    # depth params
    num_new_layers: int = 4,
    insert_strategy: InsertStrategy = "uniform",
    # expert params
    expand_factor: int = 2,
    noise_scale: float = 1e-6,
    # width params (dense only)
    ffn_size_expansion: int = 0,
    # I/O
    target_shard_gb: float = 4.0,
    verbose: bool = True,
    dry_run: bool = False,
    workers: int = 1,
    validate_output: bool = False,
    resume: bool = False,
) -> None:
    """Detect model type and run the appropriate safetensor expander.

    Args:
        src_dir:            Source model directory.
        dst_dir:            Output directory.
        method:             ``"depth"`` | ``"expert"`` | ``"width"``.
        num_new_layers:     [depth] Number of identity layers to insert.
        insert_strategy:    [depth] ``"uniform"`` | ``"front"`` | ``"rear"``.
        expand_factor:      [expert] Integer multiplier for expert count.
        noise_scale:        [expert] Noise added to router weight copies.
        ffn_size_expansion: [width] intermediate_size increment.
        target_shard_gb:    Output shard size limit.
        verbose:            Print progress.
        dry_run:            Build plan only, do not write files.
        validate_output:    If True, verify output can be structurally loaded.
        resume:             If True, skip already-written output shards.
    """
    src_dir = Path(src_dir)
    profile = detect_model(src_dir)

    if verbose:
        logger.info("\n%s\n", profile.summary())

    expander = _build_expander(
        method,
        profile,
        num_new_layers,
        insert_strategy,
        expand_factor,
        noise_scale,
        ffn_size_expansion,
    )

    if dry_run:
        expander.dry_run(src_dir)
    else:
        expander.expand(
            src_dir=src_dir,
            dst_dir=dst_dir,
            target_shard_bytes=int(target_shard_gb * 1024**3),
            workers=workers,
            verbose=verbose,
            validate_output=validate_output,
            resume=resume,
        )


def _build_expander(
    method: str,
    profile: ModelProfile,
    num_new_layers: int,
    insert_strategy: InsertStrategy,
    expand_factor: int,
    noise_scale: float,
    ffn_size_expansion: int,
) -> SafetensorExpanderBase:
    family = profile.family

    if method == "expert" and not profile.is_moe:
        raise ValueError(
            f"method='expert' requires a MoE model, but detected "
            f"family='{family}' (Dense). "
            f"Use method='depth' or method='width' for Dense models."
        )

    factory = _EXPANDER_REGISTRY.get((method, family))

    if factory is None:
        available = _available_families(method)
        raise ValueError(
            f"No expander registered for method={method!r} and family={family!r}. "
            f"Available families for this method: {available or 'none'}."
        )

    # Collect only the parameters the selected method needs, using the
    # declarative _METHOD_PARAM_KEYS mapping.  Adding a new method only
    # requires updating that dict — no if-elif chain changes needed.
    all_params: dict[str, Any] = {
        "num_new_layers": num_new_layers,
        "insert_strategy": insert_strategy,
        "expand_factor": expand_factor,
        "noise_scale": noise_scale,
        "ffn_size_expansion": ffn_size_expansion,
        "profile": profile,
    }
    needed = _METHOD_PARAM_KEYS.get(method, ())
    kwargs = {k: all_params[k] for k in needed}

    return factory(**kwargs)


# ── registered factories ──────────────────────────────────────────────────────


@register_expander("depth", "dense")
def _build_dense_depth(
    profile: ModelProfile,
    num_new_layers: int,
    insert_strategy: InsertStrategy,
    **_: Any,
) -> SafetensorExpanderBase:
    from llm_grow.safetensor.methods.zero_block_insert import (
        ZeroBlockInsertSafetensorConfig,
        ZeroBlockInsertSafetensorExpander,
    )

    return ZeroBlockInsertSafetensorExpander(
        ZeroBlockInsertSafetensorConfig(
            num_new_layers=num_new_layers,
            insert_strategy=insert_strategy,
            zero_suffixes=(
                profile.attn_zero_suffixes + profile.dense_mlp_zero_suffixes
            ),
        )
    )


@register_expander("depth", "standard_moe")
@register_expander("depth", "deepseek_moe")
def _build_moe_depth(
    profile: ModelProfile,
    num_new_layers: int,
    insert_strategy: InsertStrategy,
    **_: Any,
) -> SafetensorExpanderBase:
    from llm_grow.safetensor.models.moe_generic import (
        GenericMoEDepthConfig,
        GenericMoEDepthExpander,
    )

    return GenericMoEDepthExpander(
        GenericMoEDepthConfig(
            num_new_layers=num_new_layers,
            insert_strategy=insert_strategy,
            zero_suffixes=(
                profile.attn_zero_suffixes + profile.dense_mlp_zero_suffixes
            ),
            zero_shared_expert_down=profile.has_shared_expert,
        )
    )


@register_expander("depth", "longcat")
def _build_longcat_depth(
    profile: ModelProfile,
    num_new_layers: int,
    insert_strategy: InsertStrategy,
    **_: Any,
) -> SafetensorExpanderBase:
    from llm_grow.safetensor.models.longcat import (
        LongcatDepthConfig,
        LongcatDepthExpander,
    )

    return LongcatDepthExpander(
        LongcatDepthConfig(
            num_new_layers=num_new_layers,
            insert_strategy=insert_strategy,
        )
    )


@register_expander("expert", "longcat")
def _build_longcat_expert(
    profile: ModelProfile,
    expand_factor: int,
    noise_scale: float,
    **_: Any,
) -> SafetensorExpanderBase:
    from llm_grow.safetensor.models.longcat import (
        LongcatExpertCloneConfig,
        LongcatExpertCloneExpander,
    )

    return LongcatExpertCloneExpander(
        LongcatExpertCloneConfig(
            expand_factor=expand_factor,
            noise_scale=noise_scale,
        )
    )


@register_expander("expert", "standard_moe")
@register_expander("expert", "deepseek_moe")
def _build_moe_expert(
    profile: ModelProfile,
    expand_factor: int,
    noise_scale: float,
    **_: Any,
) -> SafetensorExpanderBase:
    from llm_grow.safetensor.models.moe_generic import (
        GenericDenseToMoEConfig,
        GenericMoEExpertCloneExpander,
    )

    return GenericMoEExpertCloneExpander(
        GenericDenseToMoEConfig(
            expand_factor=expand_factor,
            noise_scale=noise_scale,
            router_weight_suffixes=[profile.router_weight_suffix],
            router_bias_suffixes=(
                [profile.router_bias_suffix] if profile.router_bias_suffix else []
            ),
            config_expert_count_key=profile.expert_count_config_key,
            config_topk_key=profile.topk_config_key,
        )
    )


@register_expander("width", "dense")
def _build_dense_width(
    profile: ModelProfile,
    num_new_layers: int,
    insert_strategy: InsertStrategy,
    ffn_size_expansion: int,
    **_: Any,
) -> SafetensorExpanderBase:
    from llm_grow.safetensor.methods.multi_axis_pad import (
        MultiAxisPadSafetensorConfig,
        MultiAxisPadSafetensorExpander,
    )

    cfg = MultiAxisPadSafetensorConfig(
        num_new_layers=num_new_layers,
        insert_strategy=insert_strategy,
        ffn_size_expansion=ffn_size_expansion,
    )
    return MultiAxisPadSafetensorExpander(cfg)


@register_expander("width", "standard_moe")
@register_expander("width", "deepseek_moe")
@register_expander("width", "longcat")
def _build_moe_width(
    profile: ModelProfile,
    num_new_layers: int,
    insert_strategy: InsertStrategy,
    ffn_size_expansion: int,
    **_: Any,
) -> SafetensorExpanderBase:
    from llm_grow.safetensor.models.moe_width import (
        MoEWidthConfig,
        MoEWidthExpander,
    )

    return MoEWidthExpander(
        MoEWidthConfig(
            ffn_size_expansion=ffn_size_expansion,
            hidden_size_expansion=0,
            num_new_layers=num_new_layers,
            insert_strategy=insert_strategy,
            zero_suffixes=(
                profile.attn_zero_suffixes + profile.dense_mlp_zero_suffixes
            ),
            zero_shared_expert_down=profile.has_shared_expert,
        )
    )
