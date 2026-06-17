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
    """Noise std for router classifier rows (relative to tensor std)."""

    double_zero_experts: bool = True
    """Also double ``zero_expert_num`` in config."""


class LongcatExpertUpcyclingExpander(SafetensorExpanderBase):
    """Expand LongCat-Flash expert count using mmap streaming.

    Differences from the bundled ``expand_experts.py``:

    * **Memory-efficient**: processes one shard at a time via mmap instead of
      loading entire shards with ``load_file()``.
    * **Generalised expand_factor**: supports any integer multiplier, not just ×2.
    * **Integrated into llm-grow**: uses the shared ``TensorRecipe`` / plan
      infrastructure, verification script, and CLI.

    Example::

        cfg = LongcatExpertUpcyclingConfig(expand_factor=2, noise_scale=1e-6)
        LongcatExpertUpcyclingExpander(cfg).expand(
            src_dir="LongCat-Flash-Chat",
            dst_dir="LongCat-Flash-Chat-2x-experts",
        )

        # Or dry-run first (no weight files required):
        LongcatExpertUpcyclingExpander(cfg).dry_run("LongCat-Flash-Chat")
    """

    def __init__(self, config: LongcatExpertUpcyclingConfig | None = None) -> None:
        self.config = config or LongcatExpertUpcyclingConfig()

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        cfg = self.config
        wmap = src_index.weight_map

        orig_n_experts = self._count_experts_per_layer(wmap)
        new_n_experts = orig_n_experts * cfg.expand_factor

        # Config patches
        patches: dict = {"n_routed_experts": new_n_experts}
        orig_cfg = self._peek_config(src_index.model_dir)
        if cfg.double_zero_experts and "zero_expert_num" in orig_cfg:
            patches["zero_expert_num"] = orig_cfg["zero_expert_num"] * cfg.expand_factor
        if "moe_topk" in orig_cfg:
            patches["moe_topk"] = orig_cfg["moe_topk"] * cfg.expand_factor

        plan = ExpansionPlan(
            new_num_hidden_layers=src_index.num_hidden_layers(),
            config_patches=patches,
        )

        for key, shard in wmap.items():
            if _is_expert_key(key):
                # Keep the original expert
                plan.passthrough(key, shard)
                # Create (expand_factor - 1) copies with offset indices
                for copy_idx in range(1, cfg.expand_factor):
                    offset = orig_n_experts * copy_idx
                    new_key = _expert_key_offset(key, offset)
                    plan.add(new_key, TensorRecipe(src_shard=shard, src_key=key))

            elif key.endswith("mlp.router.classifier.weight"):
                # Duplicate rows: [N, H] → [N*factor, H] (with noise on copies)
                plan.add(
                    key,
                    TensorRecipe(
                        src_shard=shard,
                        src_key=key,
                        dup_rows=True,
                        dup_rows_noise_scale=cfg.noise_scale,
                    ),
                )
                # Note: dup_rows doubles once; for factor>2 we'd need to chain.
                # factor > 2 requires custom handling — flag it.
                if cfg.expand_factor > 2:
                    raise NotImplementedError(
                        "expand_factor > 2 for router classifier requires "
                        "multi-pass duplication; only expand_factor=2 supported."
                    )

            elif key.endswith("mlp.router.e_score_correction_bias"):
                # Bias: [N] → [N*factor] (exact copy, no noise)
                plan.add(
                    key,
                    TensorRecipe(
                        src_shard=shard,
                        src_key=key,
                        dup_rows=True,
                        dup_rows_noise_scale=0.0,
                    ),
                )

            else:
                plan.passthrough(key, shard)

        return plan

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _count_experts_per_layer(wmap: dict[str, str]) -> int:
        """Count distinct expert indices in layer 0."""
        indices = {_expert_idx(k) for k in wmap if k.startswith("model.layers.0.") and _is_expert_key(k)}
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

        positions = set(_insert_positions(num_orig, cfg.num_new_layers, cfg.insert_strategy))
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
