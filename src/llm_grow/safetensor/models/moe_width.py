"""MoE width expansion: expert FFN width (M3) + hidden_size/Attention (M4).

Supports two orthogonal width axes for MoE models:

  M3 — **Expert FFN width** (``ffn_size_expansion``):
       Zero-pad each expert's ``gate_proj``/``up_proj`` rows and ``down_proj``
       cols to increase ``intermediate_size``.  Router is untouched (it only
       sees ``hidden_size``).  Shared experts are padded identically.

  M4 — **Hidden size / Attention** (``hidden_size_expansion``):
       Zero-pad the ``hidden_size`` dimension across *every* linear layer:
       attention Q/K/V/O projections, all expert FFN projections (input/output
       dim), router weight cols, shared experts, embeddings, lm_head, and
       layernorms.

Both are function-preserving: new dimensions are zero-initialized so
forward output is unchanged.  Can be combined with depth expansion
(``num_new_layers``) for M2+M3+M4 in a single pass.

FP8 note: ``weight_scale_inv`` tensors are 1-D per-channel scales; they are
padded with zeros (scale for the new zero-weight channels is irrelevant).
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_grow.configs.base import BaseMoEDepthConfig, BaseWidthConfig
from llm_grow.safetensor.base import ExpansionPlan, SafetensorExpanderBase, TensorRecipe
from llm_grow.safetensor.utils import (
    ShardIndex,
    get_hidden_size_from_index,
    insert_positions,
    parse_layer_idx,
    read_safetensors_header,
)
from llm_grow.utils.insertion import build_layer_sequence


@dataclass
class MoEWidthConfig(BaseMoEDepthConfig, BaseWidthConfig):
    """MoE width expansion configuration (M3 + M4)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ffn_size_expansion < 0:
            raise ValueError(
                f"ffn_size_expansion must be >= 0, got {self.ffn_size_expansion}"
            )
        if self.hidden_size_expansion < 0:
            raise ValueError(
                f"hidden_size_expansion must be >= 0, got {self.hidden_size_expansion}"
            )


class MoEWidthExpander(SafetensorExpanderBase):
    """Width expansion for MoE models (M3 + M4).

    Example — M3 only (wider experts)::

        MoEWidthExpander(MoEWidthConfig(ffn_size_expansion=1024)).expand(...)

    Example — M4 only (wider hidden)::

        MoEWidthExpander(MoEWidthConfig(hidden_size_expansion=256)).expand(...)

    Example — M3 + M4 + M2 combined::

        MoEWidthExpander(MoEWidthConfig(
            ffn_size_expansion=1024,
            hidden_size_expansion=256,
            num_new_layers=4,
        )).expand(...)
    """

    def __init__(self, config: MoEWidthConfig | None = None) -> None:
        self.config = config or MoEWidthConfig()

    def _should_zero(self, suf: str) -> bool:
        """Determine if a tensor suffix should be zeroed in an identity block."""
        cfg = self.config
        return self._should_zero_moe(
            suf,
            zero_suffixes=cfg.zero_suffixes,
            zero_expert_down=cfg.zero_expert_down,
            zero_shared_expert_down=cfg.zero_shared_expert_down,
        )

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        cfg = self.config
        num_orig = src_index.num_hidden_layers()
        wmap = src_index.weight_map
        suffixes = src_index.layer_suffixes()

        if cfg.num_new_layers > 0:
            pos_set = set(
                insert_positions(num_orig, cfg.num_new_layers, cfg.insert_strategy)
            )
        else:
            pos_set = set()

        sequence = build_layer_sequence(num_orig, pos_set)

        plan = ExpansionPlan(
            new_num_hidden_layers=len(sequence),
            config_patches=self._build_config_patches(src_index),
        )

        for new_idx, (src_idx, is_identity) in enumerate(sequence):
            for suf in suffixes:
                src_key = f"model.layers.{src_idx}.{suf}"
                if src_key not in wmap:
                    continue
                new_key = f"model.layers.{new_idx}.{suf}"

                zero = is_identity and self._should_zero(suf)
                pad_r, pad_c = self._compute_padding(suf)

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

        self._build_non_layer_plan(plan, wmap)
        return plan

    def _compute_padding(self, suf: str) -> tuple[int, int]:
        """Return (pad_rows, pad_cols) for a layer-level tensor suffix."""
        cfg = self.config
        pad_r = pad_c = 0

        is_expert = "mlp.experts." in suf or "mlp.shared_experts." in suf
        is_scale = suf.endswith("weight_scale_inv")

        if cfg.ffn_size_expansion > 0 and is_expert and not is_scale:
            proj_name = suf.split(".")[-2] if suf.count(".") >= 2 else ""
            if proj_name in ("gate_proj", "up_proj"):
                pad_r += cfg.ffn_size_expansion
            elif proj_name == "down_proj":
                pad_c += cfg.ffn_size_expansion

        if cfg.ffn_size_expansion > 0 and is_expert and is_scale:
            proj_name = suf.split(".")[-3] if suf.count(".") >= 3 else ""
            if proj_name in ("gate_proj", "up_proj"):
                pad_r += cfg.ffn_size_expansion

        if cfg.hidden_size_expansion > 0:
            if is_expert and not is_scale:
                proj_name = suf.split(".")[-2] if suf.count(".") >= 2 else ""
                if proj_name in ("gate_proj", "up_proj"):
                    pad_c += cfg.hidden_size_expansion
                elif proj_name == "down_proj":
                    pad_r += cfg.hidden_size_expansion
            elif is_expert and is_scale:
                proj_name = suf.split(".")[-3] if suf.count(".") >= 3 else ""
                if proj_name == "down_proj":
                    pad_r += cfg.hidden_size_expansion
            elif suf.endswith("mlp.gate.weight"):
                pad_c += cfg.hidden_size_expansion
            elif suf.startswith("self_attn.") and suf.endswith(".weight"):
                proj = suf.split(".")[-2]
                if proj in ("q_proj", "k_proj", "v_proj"):
                    pad_c += cfg.hidden_size_expansion
                elif proj == "o_proj":
                    pad_r += cfg.hidden_size_expansion
            elif "norm" in suf and suf.endswith(".weight"):
                pad_r += cfg.hidden_size_expansion

        return pad_r, pad_c

    def _build_non_layer_plan(self, plan: ExpansionPlan, wmap: dict[str, str]) -> None:
        cfg = self.config
        for key, shard in wmap.items():
            if parse_layer_idx(key) is not None:
                continue
            if cfg.hidden_size_expansion > 0:
                pad_r = pad_c = 0
                if "embed_tokens" in key or "lm_head" in key:
                    pad_c = cfg.hidden_size_expansion
                elif "norm" in key and ".weight" in key:
                    pad_r = cfg.hidden_size_expansion

                if pad_r or pad_c:
                    plan.add(
                        key,
                        TensorRecipe(
                            src_shard=shard,
                            src_key=key,
                            pad_rows=pad_r,
                            pad_cols=pad_c,
                        ),
                    )
                else:
                    plan.passthrough(key, shard)
            else:
                plan.passthrough(key, shard)

    def _build_config_patches(self, src_index: ShardIndex) -> dict:
        cfg = self.config
        patches: dict = {}
        if cfg.ffn_size_expansion > 0:
            orig = _get_expert_intermediate_size(src_index)
            if orig > 0:
                patches["moe_intermediate_size"] = orig + cfg.ffn_size_expansion
        if cfg.hidden_size_expansion > 0:
            orig = get_hidden_size_from_index(src_index)
            if orig > 0:
                patches["hidden_size"] = orig + cfg.hidden_size_expansion
        return patches


def _get_expert_intermediate_size(src_index: ShardIndex) -> int:
    """Infer expert intermediate_size from expert 0 gate_proj shape."""
    for key in src_index.weight_map:
        if "mlp.experts.0.gate_proj.weight" in key:
            shard_path = src_index.model_dir / src_index.weight_map[key]
            header = read_safetensors_header(shard_path)
            if key in header:
                _dtype, shape = header[key]
                return shape[0]
    return 0
