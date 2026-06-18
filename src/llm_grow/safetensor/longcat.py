"""LongCat-Flash safetensor expanders.

Supports two expansion modes for the LongcatFlash architecture:

1. **LongcatExpertUpcyclingExpander** — Expert count expansion (e.g. 512 → 1024).
   Uses mmap streaming: processes one shard at a time, never loads all weights.
   Memory peak ≈ largest shard × 2.

2. **LongcatDepthExpander** — Depth expansion via identity block insertion
   (LLaMA-Pro style), adapted for LongCat's dual-attention + MoE structure.

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

import re
from dataclasses import dataclass

from llm_grow.safetensor.base import ExpansionPlan, SafetensorExpanderBase, TensorRecipe
from llm_grow.safetensor.utils import ShardIndex, parse_layer_idx

# ── regex helpers ─────────────────────────────────────────────────────────────

_EXPERT_RE = re.compile(r"^(.*\.mlp\.experts\.)(\d+)(\..*)$")


def _is_expert_key(key: str) -> bool:
    return bool(_EXPERT_RE.match(key))


def _expert_idx(key: str) -> int:
    m = _EXPERT_RE.match(key)
    return int(m.group(2)) if m else -1


def _expert_key_offset(key: str, offset: int) -> str:
    """Rename expert index: experts.{i}.* → experts.{i+offset}.*"""
    m = _EXPERT_RE.match(key)
    return f"{m.group(1)}{int(m.group(2)) + offset}{m.group(3)}"


# ── Expert Upcycling ──────────────────────────────────────────────────────────


@dataclass
class LongcatExpertUpcyclingConfig:
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


class LongcatExpertUpcyclingExpander(SafetensorExpanderBase):
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

        cfg = LongcatExpertUpcyclingConfig(
            expand_factor=2,
            noise_scale=1e-6,
            scale_moe_topk=True,   # 12 → 24 (same activation ratio)
        )
        LongcatExpertUpcyclingExpander(cfg).expand(
            src_dir="LongCat-Flash-Chat",
            dst_dir="LongCat-Flash-Chat-2x-experts",
        )

        # group-routing variant (topk unchanged, add routing flags)
        cfg2 = LongcatExpertUpcyclingConfig(expand_factor=2, use_group_routing=True)
        LongcatExpertUpcyclingExpander(cfg2).dry_run("LongCat-Flash-Chat")
    """

    def __init__(self, config: LongcatExpertUpcyclingConfig | None = None) -> None:
        self.config = config or LongcatExpertUpcyclingConfig()

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        cfg = self.config
        wmap = src_index.weight_map

        orig_n_experts = self._count_experts_per_layer(wmap)
        new_n_experts = orig_n_experts * cfg.expand_factor

        # Config patches
        orig_cfg = self._peek_config(src_index.model_dir)
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
            if _is_expert_key(key):
                # Keep original; create (expand_factor - 1) copies with index offset
                plan.passthrough(key, shard)
                for copy_idx in range(1, cfg.expand_factor):
                    offset = orig_n_experts * copy_idx
                    new_key = _expert_key_offset(key, offset)
                    plan.add(new_key, TensorRecipe(src_shard=shard, src_key=key))

            elif key.endswith("mlp.router.classifier.weight"):
                # Expand factor ≥ 2 supported via repeated plan entries.
                # Each step doubles: [N]→[2N]→[4N]→…
                # For expand_factor > 2, apply the recipe to successive results.
                # For simplicity in the plan model, we only do ×2 at recipe level
                # and generate additional copy keys for higher factors.
                # Current recipe model supports ×2 natively (dup_rows).
                # For expand_factor > 2, log a warning but proceed with ×2 semantics.
                if cfg.expand_factor > 2:
                    raise NotImplementedError(
                        f"expand_factor={cfg.expand_factor} > 2 for router classifier "
                        "requires multi-pass duplication; only expand_factor=2 is supported. "
                        "Use the standalone expand_moe_experts.py for arbitrary factors."
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
            _expert_idx(k)
            for k in wmap
            if k.startswith("model.layers.0.") and _is_expert_key(k)
        }
        return len(indices)

    @staticmethod
    def _peek_config(model_dir) -> dict:
        import json

        cfg_path = model_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return json.load(f)
        return {}


# ── Depth Expansion ───────────────────────────────────────────────────────────


@dataclass
class LongcatDepthConfig:
    num_new_layers: int = 4
    """Number of identity layers to insert."""

    insert_strategy: str = "uniform"
    """'uniform' | 'front' | 'rear'"""

    noise_scale: float = 0.0
    """Optional noise for non-zero tensors in identity blocks (default 0 = exact copy)."""


class LongcatDepthExpander(SafetensorExpanderBase):
    """Insert LLaMA-Pro–style identity layers into LongCat-Flash.

    An identity layer zeros:
    * ``self_attn.{0,1}.o_proj.weight``         — attention output projections
    * ``mlps.{0,1}.down_proj.weight``            — dense MLP output projections
    * ``mlp.experts.{ALL}.down_proj.weight``     — all 512 MoE expert outputs

    Residual connections ensure ``Layer(x) = x``.

    Example::

        LongcatDepthExpander(LongcatDepthConfig(num_new_layers=4)).dry_run(
            "LongCat-Flash-Chat"
        )
    """

    # Exact-suffix matches for non-expert identity zeroing
    _ATTN_ZERO = frozenset(
        {
            "self_attn.0.o_proj.weight",
            "self_attn.1.o_proj.weight",
        }
    )
    _DENSE_MLP_ZERO = frozenset(
        {
            "mlps.0.down_proj.weight",
            "mlps.1.down_proj.weight",
        }
    )

    def __init__(self, config: LongcatDepthConfig | None = None) -> None:
        self.config = config or LongcatDepthConfig()

    def _should_zero(self, suf: str) -> bool:
        """Return True if this tensor suffix should be zeroed in identity blocks."""
        if suf in self._ATTN_ZERO or suf in self._DENSE_MLP_ZERO:
            return True
        # Zero ALL expert down_proj weights
        return bool(suf.endswith(".down_proj.weight") and "mlp.experts." in suf)

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        from llm_grow.safetensor.llama_pro import _insert_positions

        cfg = self.config
        num_orig = src_index.num_hidden_layers()
        wmap = src_index.weight_map
        suffixes = src_index.layer_suffixes()

        positions = set(
            _insert_positions(num_orig, cfg.num_new_layers, cfg.insert_strategy)
        )
        sequence: list[tuple[int, bool]] = []
        for i in range(num_orig):
            sequence.append((i, False))
            if i in positions:
                sequence.append((i, True))  # identity copy

        plan = ExpansionPlan(new_num_hidden_layers=len(sequence))

        for new_idx, (src_idx, is_identity) in enumerate(sequence):
            for suf in suffixes:
                src_key = f"model.layers.{src_idx}.{suf}"
                if src_key not in wmap:
                    continue
                new_key = f"model.layers.{new_idx}.{suf}"
                plan.add(
                    new_key,
                    TensorRecipe(
                        src_shard=wmap[src_key],
                        src_key=src_key,
                        zero_out=is_identity and self._should_zero(suf),
                    ),
                )

        # Non-layer tensors (embed, norm, mtp.*) pass through unchanged
        for key, shard in wmap.items():
            if parse_layer_idx(key) is None:
                plan.passthrough(key, shard)

        return plan
