"""LongCat-Flash safetensor expanders.

Supports two expansion modes for the LongcatFlash architecture:

1. **LongcatExpertCloneExpander** — Expert count expansion (e.g. 512 → 1024).
   Uses mmap streaming: processes one shard at a time, never loads all weights.
   Memory peak ≈ largest shard × 2.

2. **LongcatDepthExpander** — Depth expansion via identity block insertion
   (ZeroBlockInsert style), adapted for LongCat's dual-attention + MoE structure.

LongCat-Flash architecture specifics
-------------------------------------
- 28 layers, each with:
  * Two MLA attention heads:  ``self_attn.0.*``, ``self_attn.1.*``
  * 512 routed experts:       ``mlp.experts.{i}.{gate/up/down}_proj.weight``
  * Router:                   ``mlp.router.classifier.weight [N_experts, hidden]``
                              ``mlp.router.e_score_correction_bias [N_experts]``
  * Two dense MLPs:           ``mlps.{0,1}.{gate/up/down}_proj.weight``
  * Two layernorms each:      ``input_layernorm.{0,1}``,
                              ``post_attention_layernorm.{0,1}``
- MTP (Multi-Token Prediction) module at ``model.mtp.*`` — always passed through.
- config keys:  ``n_routed_experts``, ``zero_expert_num``, ``moe_topk``
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_grow.configs.base import BaseMoEDepthConfig, ExpansionConfig
from llm_grow.safetensor.base import ExpansionPlan, SafetensorExpanderBase, TensorRecipe
from llm_grow.safetensor.utils import (
    ShardIndex,
    expert_idx,
    expert_key_offset,
    is_expert_key,
    peek_model_config,
)
from llm_grow.utils.insertion import build_layer_sequence

# ── ExpertClone ──────────────────────────────────────────────────────────


@dataclass
class LongcatExpertCloneConfig(ExpansionConfig):
    expand_factor: int = 2
    """Expert count multiplier (e.g. 2 → 512 experts × 2 = 1024)."""

    noise_scale: float = 1e-6
    """Noise std for router classifier rows (relative to tensor std).
    Applied only to real expert rows; zero expert rows always use noise=0."""

    double_zero_experts: bool = True
    """Also double ``zero_expert_num`` in config (default True)."""

    scale_moe_topk: bool = False
    """Whether to scale ``moe_topk`` by ``expand_factor``.

    Default **False** to match the original ``expand_experts.py`` behaviour
    (topk unchanged unless explicitly requested).  Set to True if you want to
    maintain the same activation *ratio* after expert doubling.
    """

    use_group_routing: bool = False
    """Add ``use_group_routing=True`` and ``expert_expansion_factor`` to
    config instead of scaling topk.  Mutually exclusive with ``scale_moe_topk``.
    Mirrors the ``--use_group_routing`` flag in the original script.
    """


class LongcatExpertCloneExpander(SafetensorExpanderBase):
    """Expand LongCat-Flash expert count using mmap streaming.

    Behavioural alignment with the bundled ``expand_experts.py``
    ---------------------------------------------------------------
    For ``expand_factor=2`` (the common case) this expander produces
    **identical tensor values** to ``expand_experts.py``:

    * Real expert weights copied as-is; copies get small Gaussian noise
      (``noise_scale`` relative to per-tensor std) to break symmetry.
    * Zero expert weights (``zero_expert_num``) are always copied exactly
      — no noise — because they are identity-initialised blocks that should
      stay semantically neutral.
    * Router ``classifier.weight`` layout: ``[real×k | zero×k, hidden]``
      where ``k = expand_factor``, real copies have noise, zero copies do not.
    * Router ``e_score_correction_bias`` layout: same structure, no noise.

    Differences from ``expand_experts.py``
    ----------------------------------------
    * **Memory-efficient**: processes one shard at a time via mmap;
      ``expand_experts.py`` loads entire shards with ``load_file()``.
    * **scale_moe_topk** defaults to *False* (matches original script default
      which leaves topk unchanged unless ``--target_topk`` is given).
    * **use_group_routing** maps to ``--use_group_routing`` in the original.
    * **expand_factor > 2** for the router still raises ``NotImplementedError``
      (same workaround: use ``expand_experts.py`` for arbitrary factors).

    Example::

        cfg = LongcatExpertCloneConfig(
            expand_factor=2,
            noise_scale=1e-6,
            scale_moe_topk=True,   # 12 → 24 (same activation ratio)
        )
        LongcatExpertCloneExpander(cfg).expand(
            src_dir="LongCat-Flash-Chat",
            dst_dir="LongCat-Flash-Chat-2x-experts",
        )

        # group-routing variant (topk unchanged, add routing flags)
        cfg2 = LongcatExpertCloneConfig(expand_factor=2, use_group_routing=True)
        LongcatExpertCloneExpander(cfg2).dry_run("LongCat-Flash-Chat")
    """

    def __init__(self, config: LongcatExpertCloneConfig | None = None) -> None:
        self.config = config or LongcatExpertCloneConfig()

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        cfg = self.config
        wmap = src_index.weight_map

        orig_n_experts = self._count_experts_per_layer(wmap)
        new_n_experts = orig_n_experts * cfg.expand_factor

        # Config patches
        orig_cfg = peek_model_config(src_index.model_dir)
        zero_expert_num: int = orig_cfg.get("zero_expert_num") or 0

        if cfg.use_group_routing and cfg.scale_moe_topk:
            raise ValueError(
                "use_group_routing and scale_moe_topk are mutually exclusive. "
                "use_group_routing keeps moe_topk unchanged by design."
            )

        # Config patches — mirrors expand_experts.py's expand_config()
        patches: dict = {"n_routed_experts": new_n_experts}
        if cfg.double_zero_experts and zero_expert_num > 0:
            patches["zero_expert_num"] = zero_expert_num * cfg.expand_factor
        if cfg.scale_moe_topk and "moe_topk" in orig_cfg:
            # scale topk to maintain the same activation ratio
            patches["moe_topk"] = orig_cfg["moe_topk"] * cfg.expand_factor
        if cfg.use_group_routing:
            # group routing: keep topk unchanged, add routing flags to config
            patches["use_group_routing"] = True
            patches["expert_expansion_factor"] = cfg.expand_factor

        plan = ExpansionPlan(
            new_num_hidden_layers=src_index.num_hidden_layers(),
            config_patches=patches,
        )

        # router_split: rows [0:orig_n_experts] = real experts (get noise),
        #               rows [orig_n_experts:]  = zero experts (no noise)
        router_split = orig_n_experts if zero_expert_num > 0 else 0

        for key, shard in wmap.items():
            if is_expert_key(key):
                # Keep original; create (expand_factor - 1) copies with index offset
                plan.passthrough(key, shard)
                for copy_idx in range(1, cfg.expand_factor):
                    offset = orig_n_experts * copy_idx
                    new_key = expert_key_offset(key, offset)
                    plan.add(new_key, TensorRecipe(src_shard=shard, src_key=key))

            elif key.endswith("mlp.router.classifier.weight"):
                # Router duplication is only implemented for expand_factor=2.
                # Higher factors require multi-pass duplication and are not
                # supported by the current recipe model.  Use the standalone
                # expand_moe_experts.py for arbitrary factors.
                if cfg.expand_factor > 2:
                    raise NotImplementedError(
                        f"expand_factor={cfg.expand_factor} > 2 "
                        "for router classifier requires "
                        "multi-pass duplication; only "
                        "expand_factor=2 is supported. "
                        "Use the standalone "
                        "expand_moe_experts.py for "
                        "arbitrary factors."
                    )
                plan.add(
                    key,
                    TensorRecipe(
                        src_shard=shard,
                        src_key=key,
                        dup_rows=True,
                        dup_rows_noise_scale=cfg.noise_scale,
                        router_split=router_split,  # ← real/zero separation
                    ),
                )

            elif key.endswith("mlp.router.e_score_correction_bias"):
                # Bias: exact copies, no noise, but still split real/zero
                plan.add(
                    key,
                    TensorRecipe(
                        src_shard=shard,
                        src_key=key,
                        dup_rows=True,
                        dup_rows_noise_scale=0.0,
                        router_split=router_split,
                    ),
                )

            else:
                plan.passthrough(key, shard)

        return plan

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _count_experts_per_layer(wmap: dict[str, str]) -> int:
        """Count distinct expert indices in layer 0."""
        indices = {
            expert_idx(k)
            for k in wmap
            if k.startswith("model.layers.0.") and is_expert_key(k)
        }
        return len(indices)


# ── Depth Expansion ───────────────────────────────────────────────────────────


@dataclass
class LongcatDepthConfig(BaseMoEDepthConfig):
    """LongCat depth expansion configuration."""

    zero_suffixes: list[str] = field(
        default_factory=lambda: [
            "self_attn.0.o_proj.weight",
            "self_attn.1.o_proj.weight",
            "mlps.0.down_proj.weight",
            "mlps.1.down_proj.weight",
        ]
    )
    """Exact layer-suffixes to zero in identity blocks."""

    noise_scale: float = 0.0
    """Optional noise for non-zero tensors in identity blocks
    (default 0 = exact copy)."""


class LongcatDepthExpander(SafetensorExpanderBase):
    """Insert ZeroBlockInsert-style identity layers into LongCat-Flash.

    An identity layer zeros attention output projections, dense MLP output
    projections, and all MoE expert down_proj weights.  Residual connections
    ensure ``Layer(x) = x``.

    Example::

        LongcatDepthExpander(LongcatDepthConfig(num_new_layers=4)).dry_run(
            "LongCat-Flash-Chat"
        )
    """

    def __init__(self, config: LongcatDepthConfig | None = None) -> None:
        self.config = config or LongcatDepthConfig()

    def _should_zero(self, suf: str) -> bool:
        """Return True if this tensor suffix should be zeroed in identity blocks."""
        cfg = self.config
        if suf in cfg.zero_suffixes:
            return True
        if suf.endswith(".down_proj.weight") and "mlp.experts." in suf:
            return cfg.zero_expert_down
        return bool(
            cfg.zero_shared_expert_down and suf == "mlp.shared_experts.down_proj.weight"
        )

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        from llm_grow.safetensor.utils import insert_positions

        cfg = self.config
        num_orig = src_index.num_hidden_layers()

        positions = set(
            insert_positions(num_orig, cfg.num_new_layers, cfg.insert_strategy)
        )
        sequence = build_layer_sequence(num_orig, positions)

        return self._build_layer_plan(src_index, layer_sequence=sequence)
