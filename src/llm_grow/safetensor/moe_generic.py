"""Generic MoE safetensor expanders for standard ``mlp.experts.{i}.*`` architectures.

Pre-configured for:
  - **Qwen3MoE**   (Qwen3-30B-A3B): 128 experts/layer, GQA attn, no fp8
  - **KimiK2**     (Kimi-K2-Base) : 384 experts/layer, MLA attn, fp8 weights,
                                     layer-0 dense, shared expert per MoE layer

Both are backed by a single ``GenericMoEExpertUpcyclingExpander`` and
``GenericMoEDepthExpander`` that handle any model following the
``model.layers.{i}.mlp.experts.{j}.*`` naming convention.

FP8 note
--------
Kimi-K2 stores fp8 weights as ``{name}.weight`` + ``{name}.weight_scale_inv``.
When duplicating experts we copy both the weight and its scale tensor unchanged
(scale tensors are fine to copy since the corresponding weight will be identical).
For identity blocks we only zero the ``down_proj.weight``; scale tensors are left
intact (they are multiplied by zero weights, so the output is still zero).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_grow.safetensor.base import ExpansionPlan, SafetensorExpanderBase, TensorRecipe
from llm_grow.safetensor.llama_pro import _insert_positions
from llm_grow.safetensor.longcat import (
    _expert_idx,
    _expert_key_offset,
    _is_expert_key,
)
from llm_grow.safetensor.utils import ShardIndex, parse_layer_idx

# ── config dataclasses ────────────────────────────────────────────────────────


@dataclass
class GenericMoEUpcyclingConfig:
    expand_factor: int = 2
    """Expert count multiplier."""

    noise_scale: float = 1e-6
    """Noise std for router weight rows (relative to tensor std)."""

    router_weight_suffixes: list[str] = field(
        default_factory=lambda: ["mlp.gate.weight"]
    )
    """Layer-suffixes for router weight matrices → duplicated rows with noise."""

    router_bias_suffixes: list[str] = field(default_factory=list)
    """Layer-suffixes for router bias vectors → duplicated rows without noise."""

    config_expert_count_key: str = "num_experts"
    """config.json key that stores the per-layer expert count."""

    config_topk_key: str = "num_experts_per_tok"
    """config.json key for top-k. If set, doubled in output config."""

    scale_topk: bool = True
    """Whether to multiply moe_topk / num_experts_per_tok by expand_factor."""


@dataclass
class GenericMoEDepthConfig:
    num_new_layers: int = 4
    """Number of identity blocks to insert."""

    insert_strategy: str = "uniform"
    """'uniform' | 'front' | 'rear'"""

    extra_attn_zero_suffixes: list[str] = field(
        default_factory=lambda: ["self_attn.o_proj.weight"]
    )
    """Exact layer-suffixes for attention output projections to zero."""

    dense_mlp_zero_suffixes: list[str] = field(
        default_factory=lambda: ["mlp.down_proj.weight"]
    )
    """Exact layer-suffixes for dense MLP outputs to zero (non-MoE layers)."""

    zero_shared_expert_down: bool = True
    """Zero mlp.shared_experts.down_proj.weight in identity blocks."""


# ── Expert Upcycling ──────────────────────────────────────────────────────────


class GenericMoEExpertUpcyclingExpander(SafetensorExpanderBase):
    """Expert upcycling for any model using ``mlp.experts.{i}.*`` keys.

    Works for Qwen3MoE, DeepSeek-V2/V3, Kimi-K2, Mixtral, etc.
    Handles fp8 ``weight_scale_inv`` tensors transparently.

    Usage::

        cfg = GenericMoEUpcyclingConfig(
            expand_factor=2,
            router_weight_suffixes=["mlp.gate.weight"],
            router_bias_suffixes=["mlp.gate.e_score_correction_bias"],
            config_expert_count_key="n_routed_experts",
        )
        GenericMoEExpertUpcyclingExpander(cfg).dry_run("path/to/model")
    """

    def __init__(self, config: GenericMoEUpcyclingConfig | None = None) -> None:
        self.config = config or GenericMoEUpcyclingConfig()

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        cfg = self.config
        wmap = src_index.weight_map

        orig_n = self._count_experts_per_moe_layer(wmap)
        if orig_n == 0:
            raise ValueError("No mlp.experts.* tensors found in weight map.")

        # Config patches
        orig_model_cfg = self._peek_config(src_index.model_dir)
        zero_expert_num: int = orig_model_cfg.get("zero_expert_num") or 0
        patches: dict[str, Any] = {}
        if cfg.config_expert_count_key:
            patches[cfg.config_expert_count_key] = orig_n * cfg.expand_factor
        if (
            cfg.scale_topk
            and cfg.config_topk_key
            and cfg.config_topk_key in orig_model_cfg
        ):
            patches[cfg.config_topk_key] = (
                orig_model_cfg[cfg.config_topk_key] * cfg.expand_factor
            )

        plan = ExpansionPlan(
            new_num_hidden_layers=src_index.num_hidden_layers(),
            config_patches=patches,
        )

        # router_split: real experts end at orig_n; zero experts follow
        router_split = orig_n if zero_expert_num > 0 else 0

        router_w_set = set(cfg.router_weight_suffixes)
        router_b_set = set(cfg.router_bias_suffixes)

        for key, shard in wmap.items():
            layer_idx = parse_layer_idx(key)
            suf = (
                key[key.index(".", len("model.layers.")) + 1 :]
                if layer_idx is not None
                else ""
            )

            if _is_expert_key(key):
                plan.passthrough(key, shard)
                for copy_i in range(1, cfg.expand_factor):
                    new_key = _expert_key_offset(key, orig_n * copy_i)
                    plan.add(new_key, TensorRecipe(src_shard=shard, src_key=key))

            elif layer_idx is not None and suf in router_w_set:
                plan.add(
                    key,
                    TensorRecipe(
                        src_shard=shard,
                        src_key=key,
                        dup_rows=True,
                        dup_rows_noise_scale=cfg.noise_scale,
                        router_split=router_split,
                    ),
                )

            elif layer_idx is not None and suf in router_b_set:
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
    def _count_experts_per_moe_layer(wmap: dict[str, str]) -> int:
        """Find the first MoE layer and count distinct expert indices."""
        num_layers = (
            max(
                (parse_layer_idx(k) or 0)
                for k in wmap
                if parse_layer_idx(k) is not None
            )
            + 1
        )
        for layer_id in range(num_layers):
            prefix = f"model.layers.{layer_id}."
            indices = {
                _expert_idx(k)
                for k in wmap
                if k.startswith(prefix) and _is_expert_key(k)
            }
            if indices:
                return len(indices)
        return 0

    @staticmethod
    def _peek_config(model_dir) -> dict:
        import json

        p = model_dir / "config.json"
        if not p.exists():
            return {}
        with open(p) as f:
            return json.load(f)


# ── Depth Expansion ───────────────────────────────────────────────────────────


class GenericMoEDepthExpander(SafetensorExpanderBase):
    """Insert LLaMA-Pro–style identity layers into any MoE model.

    Zeros per identity block:
    * ``self_attn.o_proj.weight``                  — attention output
    * ``mlp.experts.{ALL}.down_proj.weight``        — all expert outputs
    * ``mlp.shared_experts.down_proj.weight``       — shared expert (if present)
    * ``mlp.down_proj.weight`` in dense-only layers — (e.g. Kimi layer 0)

    FP8 ``weight_scale_inv`` tensors are **never zeroed** (output is still zero
    because the weight is zero; scale tensor is irrelevant).
    """

    def __init__(self, config: GenericMoEDepthConfig | None = None) -> None:
        self.config = config or GenericMoEDepthConfig()

    def _should_zero(self, suf: str) -> bool:
        cfg = self.config
        if suf in cfg.extra_attn_zero_suffixes:
            return True
        if suf in cfg.dense_mlp_zero_suffixes:
            return True
        # All expert down_proj (but NOT weight_scale_inv — see FP8 note above)
        if suf.endswith(".down_proj.weight") and "mlp.experts." in suf:
            return True
        # Shared expert down_proj
        return bool(
            cfg.zero_shared_expert_down and suf == "mlp.shared_experts.down_proj.weight"
        )

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
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
                sequence.append((i, True))

        plan = ExpansionPlan(new_num_hidden_layers=len(sequence))

        for new_idx, (src_idx, is_identity) in enumerate(sequence):
            for suf in suffixes:
                src_key = f"model.layers.{src_idx}.{suf}"
                if src_key not in wmap:
                    continue  # suffix absent for this layer (e.g. expert suf in dense layer 0)
                new_key = f"model.layers.{new_idx}.{suf}"
                plan.add(
                    new_key,
                    TensorRecipe(
                        src_shard=wmap[src_key],
                        src_key=src_key,
                        zero_out=is_identity and self._should_zero(suf),
                    ),
                )

        for key, shard in wmap.items():
            if parse_layer_idx(key) is None:
                plan.passthrough(key, shard)

        return plan


# ── pre-configured instances ──────────────────────────────────────────────────


def make_qwen3moe_upcycling(expand_factor: int = 2, noise_scale: float = 1e-6):
    """Expert upcycling for Qwen3MoeForCausalLM (Qwen3-30B-A3B, etc.)."""
    return GenericMoEExpertUpcyclingExpander(
        GenericMoEUpcyclingConfig(
            expand_factor=expand_factor,
            noise_scale=noise_scale,
            router_weight_suffixes=["mlp.gate.weight"],
            router_bias_suffixes=[],
            config_expert_count_key="num_experts",
            config_topk_key="num_experts_per_tok",
        )
    )


def make_qwen3moe_depth(num_new_layers: int = 4, strategy: str = "uniform"):
    """LLaMA-Pro depth expansion for Qwen3MoeForCausalLM."""
    return GenericMoEDepthExpander(
        GenericMoEDepthConfig(
            num_new_layers=num_new_layers,
            insert_strategy=strategy,
            extra_attn_zero_suffixes=["self_attn.o_proj.weight"],
            dense_mlp_zero_suffixes=[],  # pure MoE, no dense MLP
            zero_shared_expert_down=False,
        )
    )


def make_kimik2_upcycling(expand_factor: int = 2, noise_scale: float = 1e-6):
    """Expert upcycling for Kimi-K2-Base (DeepseekV3ForCausalLM variant)."""
    return GenericMoEExpertUpcyclingExpander(
        GenericMoEUpcyclingConfig(
            expand_factor=expand_factor,
            noise_scale=noise_scale,
            router_weight_suffixes=["mlp.gate.weight"],
            router_bias_suffixes=["mlp.gate.e_score_correction_bias"],
            config_expert_count_key="n_routed_experts",
            config_topk_key="num_experts_per_tok",
        )
    )


def make_kimik2_depth(num_new_layers: int = 4, strategy: str = "uniform"):
    """LLaMA-Pro depth expansion for Kimi-K2-Base."""
    return GenericMoEDepthExpander(
        GenericMoEDepthConfig(
            num_new_layers=num_new_layers,
            insert_strategy=strategy,
            extra_attn_zero_suffixes=["self_attn.o_proj.weight"],
            # Layer 0 is dense (no experts): zero its down_proj too
            dense_mlp_zero_suffixes=["mlp.down_proj.weight"],
            zero_shared_expert_down=True,
        )
    )
