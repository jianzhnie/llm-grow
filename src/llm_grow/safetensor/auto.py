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

from pathlib import Path

from llm_grow.safetensor.detect import ModelProfile, detect_model
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)

# ── public API ────────────────────────────────────────────────────────────────


def auto_expand(
    src_dir: str | Path,
    dst_dir: str | Path,
    method: str,
    *,
    # depth params
    num_new_layers: int = 4,
    insert_strategy: str = "uniform",
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
        )


def _build_expander(
    method: str,
    profile: ModelProfile,
    num_new_layers: int,
    insert_strategy: str,
    expand_factor: int,
    noise_scale: float,
    ffn_size_expansion: int,
):
    # ── depth ──────────────────────────────────────────────────────────────
    if method == "depth":
        return _build_depth_expander(profile, num_new_layers, insert_strategy)

    # ── expert ─────────────────────────────────────────────────────────────
    if method == "expert":
        if not profile.is_moe:
            raise ValueError(
                f"method='expert' requires a MoE model, but detected "
                f"family='{profile.family}' (Dense). "
                f"Use method='depth' or method='width' for Dense models."
            )
        return _build_expert_expander(profile, expand_factor, noise_scale)

    # ── width ──────────────────────────────────────────────────────────────
    if method == "width":
        if profile.is_moe:
            raise NotImplementedError(
                "method='width' (FFN size expansion) is not yet "
                "supported for MoE models. "
                "Use method='expert' to increase expert count."
            )
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

    raise ValueError(
        f"Unknown method: {method!r}. Choose 'depth', 'expert', or 'width'."
    )


def _build_depth_expander(profile: ModelProfile, num_new_layers: int, strategy: str):
    """Select and configure the correct depth expander for this architecture."""

    if profile.has_dual_attn:
        # LongCat-Flash: dual attention + dual MLP + 512 experts
        from llm_grow.safetensor.models.longcat import LongcatDepthConfig, LongcatDepthExpander

        return LongcatDepthExpander(
            LongcatDepthConfig(
                num_new_layers=num_new_layers,
                insert_strategy=strategy,
            )
        )

    if profile.is_moe:
        # Standard MoE or DeepSeek-style MoE
        from llm_grow.safetensor.models.moe_generic import (
            GenericMoEDepthConfig,
            GenericMoEDepthExpander,
        )

        return GenericMoEDepthExpander(
            GenericMoEDepthConfig(
                num_new_layers=num_new_layers,
                insert_strategy=strategy,
                extra_attn_zero_suffixes=profile.attn_zero_suffixes,
                dense_mlp_zero_suffixes=profile.dense_mlp_zero_suffixes,
                zero_shared_expert_down=profile.has_shared_expert,
            )
        )

    # Pure dense model
    from llm_grow.safetensor.methods.zero_block_insert import (
        ZeroBlockInsertSafetensorConfig,
        ZeroBlockInsertSafetensorExpander,
    )

    return ZeroBlockInsertSafetensorExpander(
        ZeroBlockInsertSafetensorConfig(
            num_new_blocks=num_new_layers,
            insert_strategy=strategy,
            attn_zero_suffixes=profile.attn_zero_suffixes,
            mlp_zero_suffixes=profile.dense_mlp_zero_suffixes,
        )
    )


def _build_expert_expander(
    profile: ModelProfile, expand_factor: int, noise_scale: float
):
    """Select and configure the correct expert clone expander."""

    if profile.has_dual_attn:
        # LongCat-Flash (special router structure)
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

    # Generic: Qwen3MoE, DeepSeek-V2/V3, KimiK2, Mixtral, …
    from llm_grow.safetensor.moe_generic import (
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
