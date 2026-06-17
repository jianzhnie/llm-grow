"""MSG safetensor expander: depth + FFN-width masked structural growth.

Supports two orthogonal expansion axes:
  - depth_expansion   : identity block insertion (same as LLaMA-Pro)
  - ffn_size_expansion: zero-pad gate_proj/up_proj rows and down_proj cols

Both are function-preserving: all new parameters start at zero.
hidden_size / attention-head expansion is NOT implemented here because it
requires touching every linear projection + embedding, making the recipe
much more complex; use the in-memory MSGExpander for that case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_grow.safetensor.base import ExpansionPlan, SafetensorExpanderBase, TensorRecipe
from llm_grow.safetensor.llama_pro import _insert_positions
from llm_grow.safetensor.utils import ShardIndex, parse_layer_idx

# Suffixes whose output dimension (rows) expands with intermediate_size
_FFN_OUT_SUFFIXES = frozenset({"mlp.gate_proj.weight", "mlp.up_proj.weight"})
# Suffix whose input dimension (cols) expands with intermediate_size
_FFN_IN_SUFFIXES = frozenset({"mlp.down_proj.weight"})


@dataclass
class MSGSafetensorConfig:
    # ── depth ────────────────────────────────────────────────────────────────
    depth_expansion: int = 0
    """Number of identity blocks to insert (0 = depth only disabled)."""

    insert_strategy: str = "uniform"
    """Insertion strategy: 'uniform' | 'front' | 'rear'."""

    # ── FFN width ────────────────────────────────────────────────────────────
    ffn_size_expansion: int = 0
    """Amount to increase intermediate_size (FFN hidden dim) per layer."""

    # ── zero suffixes (identity blocks) ─────────────────────────────────────
    attn_zero_suffixes: list[str] = field(default_factory=lambda: ["self_attn.o_proj.weight"])
    mlp_zero_suffixes: list[str] = field(default_factory=lambda: ["mlp.down_proj.weight"])


class MSGSafetensorExpander(SafetensorExpanderBase):
    """MSG-style safetensor expander combining depth and FFN-width growth.

    Example — depth only::

        MSGSafetensorExpander(MSGSafetensorConfig(depth_expansion=4)).expand(...)

    Example — depth + wider FFN::

        MSGSafetensorExpander(MSGSafetensorConfig(
            depth_expansion=4, ffn_size_expansion=1024,
        )).expand(...)
    """

    def __init__(self, config: MSGSafetensorConfig | None = None) -> None:
        self.config = config or MSGSafetensorConfig()
        self.IDENTITY_ZERO_SUFFIXES = frozenset(
            self.config.attn_zero_suffixes + self.config.mlp_zero_suffixes
        )

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        cfg = self.config
        num_orig = src_index.num_hidden_layers()
        wmap = src_index.weight_map
        suffixes = src_index.layer_suffixes()

        # ── depth: build layer sequence ──────────────────────────────────────
        if cfg.depth_expansion > 0:
            positions = set(_insert_positions(num_orig, cfg.depth_expansion, cfg.insert_strategy))
        else:
            positions = set()

        sequence: list[tuple[int, bool]] = []
        for i in range(num_orig):
            sequence.append((i, False))
            if i in positions:
                sequence.append((i, True))

        plan = ExpansionPlan(
            new_num_hidden_layers=len(sequence),
            config_patches=(
                {"intermediate_size": _get_intermediate_size(src_index) + cfg.ffn_size_expansion}
                if cfg.ffn_size_expansion > 0
                else {}
            ),
        )

        # ── per-layer tensors ─────────────────────────────────────────────────
        for new_idx, (src_idx, is_identity) in enumerate(sequence):
            for suf in suffixes:
                src_key = f"model.layers.{src_idx}.{suf}"
                new_key = f"model.layers.{new_idx}.{suf}"

                zero = is_identity and suf in self.IDENTITY_ZERO_SUFFIXES

                # FFN width padding — applied to ALL layers (incl. identity blocks)
                # so their tensors match the expanded intermediate_size.
                # zero_out handles zeroing *after* padding.
                pad_r = pad_c = 0
                if cfg.ffn_size_expansion > 0:
                    if suf in _FFN_OUT_SUFFIXES:
                        pad_r = cfg.ffn_size_expansion
                    elif suf in _FFN_IN_SUFFIXES:
                        pad_c = cfg.ffn_size_expansion

                plan.add(
                    new_key,
                    TensorRecipe(
                        src_shard=wmap[src_key],
                        src_key=src_key,
                        zero_out=zero,
                        pad_rows=pad_r,
                        pad_cols=pad_c,
                    ),
                )

        # ── non-layer tensors pass through ───────────────────────────────────
        for key, shard in wmap.items():
            if parse_layer_idx(key) is None:
                plan.passthrough(key, shard)

        return plan


def _get_intermediate_size(src_index: ShardIndex) -> int:
    """Infer intermediate_size from gate_proj shape in layer 0."""
    for key in src_index.weight_map:
        if key.endswith("mlp.gate_proj.weight") and key.startswith("model.layers.0."):
            # We need to peek at the tensor shape without loading full model
            # Use safe_open mmap for a single tensor
            from safetensors import safe_open

            shard = src_index.model_dir / src_index.weight_map[key]
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                return f.get_tensor(key).shape[0]
    return 0
