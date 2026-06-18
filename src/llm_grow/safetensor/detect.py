"""Model architecture auto-detection for safetensor expansion.

Reads config.json + weight-map index to build a ``ModelProfile`` that
describes every architecture-specific property needed to select and
configure the right expander — without loading any weights.

Design principle
----------------
Detection is purely structural: we look at config.json fields and
tensor-name patterns in the weight map index.  No tensors are loaded.

Supported architectures
-----------------------
Dense:
  qwen3, llama, mistral, gemma, ...   (any model with fused mlp.down_proj)

MoE-standard:
  qwen3_moe  (Qwen3-30B-A3B, etc.)   num_experts / mlp.gate.weight
  kimi_k2 / deepseek_v3              n_routed_experts + shared expert +
                                      fp8 + MLA + optional dense layer 0

MoE-longcat:
  longcat_flash                       dual-attn + dual-MLP + 512 experts +
                                      mlp.router.classifier.weight
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from llm_grow.safetensor.longcat import _expert_idx, _is_expert_key

# ── data class ────────────────────────────────────────────────────────────────


@dataclass
class ModelProfile:
    """Full architecture description derived from config + weight-map index."""

    # ── basic ─────────────────────────────────────────────────────────────────
    src_dir: Path
    model_type: str  # value of config.json "model_type"
    arch_class: str  # config.json "architectures"[0]
    num_hidden_layers: int

    # ── MoE topology ─────────────────────────────────────────────────────────
    is_moe: bool = False
    experts_per_moe_layer: int = 0  # 0 for dense
    has_shared_expert: bool = False  # mlp.shared_experts.*
    dense_only_layers: list[int] = field(default_factory=list)
    """Layer indices that carry NO expert tensors (e.g. layer 0 in KimiK2)."""

    # ── router ────────────────────────────────────────────────────────────────
    router_weight_suffix: str = "mlp.gate.weight"
    router_bias_suffix: str | None = None
    expert_count_config_key: str = "num_experts"
    topk_config_key: str = "num_experts_per_tok"

    # ── special architecture features ─────────────────────────────────────────
    has_fp8: bool = False  # weight_scale_inv tensors present
    has_mla_attn: bool = False  # DeepSeek/Kimi-K2 MLA (q_a_proj, kv_a_proj)
    has_dual_attn: bool = False  # LongCat self_attn.0 / self_attn.1
    has_dual_dense_mlp: bool = False  # LongCat mlps.0 / mlps.1

    # ── identity-block zero suffixes ─────────────────────────────────────────
    attn_zero_suffixes: list[str] = field(
        default_factory=lambda: ["self_attn.o_proj.weight"]
    )
    dense_mlp_zero_suffixes: list[str] = field(
        default_factory=lambda: ["mlp.down_proj.weight"]
    )

    @property
    def family(self) -> str:
        """Human-readable model family string."""
        if self.has_dual_attn:
            return "longcat"
        if self.is_moe and self.has_mla_attn:
            return "deepseek_moe"
        if self.is_moe:
            return "standard_moe"
        return "dense"

    def summary(self) -> str:
        lines = [
            f"ModelProfile  [{self.family}]  {self.src_dir.name}",
            f"  arch_class      : {self.arch_class}",
            f"  model_type      : {self.model_type}",
            f"  num_layers      : {self.num_hidden_layers}",
        ]
        if self.is_moe:
            lines += [
                "  is_moe          : True",
                f"  experts/layer   : {self.experts_per_moe_layer}",
                f"  shared_expert   : {self.has_shared_expert}",
                f"  dense_layers    : {self.dense_only_layers}",
                f"  router_weight   : {self.router_weight_suffix}",
                f"  router_bias     : {self.router_bias_suffix or '(none)'}",
                f"  expert_cfg_key  : {self.expert_count_config_key}",
                f"  topk_cfg_key    : {self.topk_config_key}",
            ]
        else:
            lines.append("  is_moe          : False (dense)")
        lines += [
            f"  has_fp8         : {self.has_fp8}",
            f"  has_mla_attn    : {self.has_mla_attn}",
            f"  has_dual_attn   : {self.has_dual_attn}",
            f"  attn_zero_suf   : {self.attn_zero_suffixes}",
            f"  dense_zero_suf  : {self.dense_mlp_zero_suffixes}",
        ]
        return "\n".join(lines)


# ── detection entry point ─────────────────────────────────────────────────────


def detect_model(src_dir: str | Path) -> ModelProfile:
    """Detect model architecture from config.json + safetensors index.

    No weights are loaded.

    Args:
        src_dir: Model directory containing config.json and either
                 model.safetensors or model.safetensors.index.json.

    Returns:
        A fully-populated ``ModelProfile``.
    """
    src_dir = Path(src_dir)
    cfg = _load_config(src_dir)
    wmap = _load_weight_map(src_dir)

    arch_class = (cfg.get("architectures") or ["Unknown"])[0]
    model_type = cfg.get("model_type", "")
    num_layers = cfg.get("num_hidden_layers")
    if num_layers is None:
        num_layers = cfg.get("num_layers")
    if num_layers is None:
        num_layers = 0

    # ── structural probes ────────────────────────────────────────────────────
    has_experts = any(_is_expert_key(k) for k in wmap)
    has_fp8 = any(k.endswith("weight_scale_inv") for k in wmap)
    has_mla = any("q_a_proj" in k or "kv_a_proj" in k for k in wmap)
    has_dual_attn = any("self_attn.0." in k for k in wmap)
    has_dual_mlp = any("mlps.0." in k for k in wmap)
    has_shared = any("shared_experts" in k for k in wmap)

    # ── router keys ──────────────────────────────────────────────────────────
    router_w_suf, router_b_suf = _detect_router(wmap)

    # ── expert count ─────────────────────────────────────────────────────────
    n_experts_per_moe, dense_layers = _count_experts(wmap, num_layers)

    # ── config keys ──────────────────────────────────────────────────────────
    exp_key, topk_key = _detect_config_keys(cfg)

    # ── identity-block zero suffixes ─────────────────────────────────────────
    attn_zero, dense_zero = _derive_zero_suffixes(
        has_dual_attn=has_dual_attn,
        has_dual_mlp=has_dual_mlp,
        is_moe=has_experts,
        dense_layers=dense_layers,
    )

    return ModelProfile(
        src_dir=src_dir,
        model_type=model_type,
        arch_class=arch_class,
        num_hidden_layers=num_layers,
        is_moe=has_experts,
        experts_per_moe_layer=n_experts_per_moe,
        has_shared_expert=has_shared,
        dense_only_layers=dense_layers,
        router_weight_suffix=router_w_suf,
        router_bias_suffix=router_b_suf,
        expert_count_config_key=exp_key,
        topk_config_key=topk_key,
        has_fp8=has_fp8,
        has_mla_attn=has_mla,
        has_dual_attn=has_dual_attn,
        has_dual_dense_mlp=has_dual_mlp,
        attn_zero_suffixes=attn_zero,
        dense_mlp_zero_suffixes=dense_zero,
    )


# ── internal helpers ──────────────────────────────────────────────────────────


def _load_config(src_dir: Path) -> dict:
    p = src_dir / "config.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _load_weight_map(src_dir: Path) -> dict[str, str]:
    index_path = src_dir / "model.safetensors.index.json"
    single_path = src_dir / "model.safetensors"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    if single_path.exists():
        from safetensors import safe_open

        with safe_open(str(single_path), framework="pt", device="cpu") as f:
            return dict.fromkeys(f.keys(), "model.safetensors")
    raise FileNotFoundError(f"No safetensor files in {src_dir}")


def _detect_router(wmap: dict) -> tuple[str, str | None]:
    """Return (router_weight_suffix, router_bias_suffix_or_None)."""
    # Priority: check layer 0 or layer 1 for router keys
    router_w_candidates = [
        "mlp.gate.weight",
        "mlp.router.classifier.weight",
    ]
    router_b_candidates = [
        "mlp.gate.e_score_correction_bias",
        "mlp.router.e_score_correction_bias",
    ]
    found_w = found_b = None
    for layer in range(5):
        prefix = f"model.layers.{layer}."
        for suf in router_w_candidates:
            if f"{prefix}{suf}" in wmap:
                found_w = suf
                break
        for suf in router_b_candidates:
            if f"{prefix}{suf}" in wmap:
                found_b = suf
                break
        if found_w:
            break
    return found_w or "mlp.gate.weight", found_b


def _count_experts(wmap: dict, num_layers: int) -> tuple[int, list[int]]:
    """Return (max_experts_per_moe_layer, dense_only_layer_indices)."""
    max_experts = 0
    dense_layers: list[int] = []
    for layer_id in range(num_layers):
        prefix = f"model.layers.{layer_id}."
        idxs = {
            _expert_idx(k) for k in wmap if k.startswith(prefix) and _is_expert_key(k)
        }
        if idxs:
            max_experts = max(max_experts, len(idxs))
        else:
            if any(k.startswith(prefix) for k in wmap):
                dense_layers.append(layer_id)
    return max_experts, dense_layers


def _detect_config_keys(cfg: dict) -> tuple[str, str]:
    """Return (expert_count_key, topk_key) that actually exist in config."""
    expert_key = next(
        (
            k
            for k in ("num_experts", "n_routed_experts", "num_routed_experts")
            if k in cfg
        ),
        "num_experts",
    )
    topk_key = next(
        (
            k
            for k in ("num_experts_per_tok", "moe_topk", "num_experts_per_token")
            if k in cfg
        ),
        "num_experts_per_tok",
    )
    return expert_key, topk_key


def _derive_zero_suffixes(
    *,
    has_dual_attn: bool,
    has_dual_mlp: bool,
    is_moe: bool,
    dense_layers: list[int],
) -> tuple[list[str], list[str]]:
    """Compute which layer-suffixes to zero in LLaMA-Pro identity blocks."""
    if has_dual_attn:
        attn_zero = ["self_attn.0.o_proj.weight", "self_attn.1.o_proj.weight"]
    else:
        attn_zero = ["self_attn.o_proj.weight"]

    dense_zero: list[str] = []
    if has_dual_mlp:
        dense_zero = ["mlps.0.down_proj.weight", "mlps.1.down_proj.weight"]
    elif not is_moe:
        # Pure dense model: zero the single fused down_proj
        dense_zero = ["mlp.down_proj.weight"]
    elif dense_layers:
        # Mixed: some layers are dense (e.g. layer 0 in KimiK2)
        dense_zero = ["mlp.down_proj.weight"]
    # Pure MoE (no dense layers): dense_zero stays []
    # (all expert down_proj handled via pattern match in expander)

    return attn_zero, dense_zero
