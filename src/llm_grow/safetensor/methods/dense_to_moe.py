"""DenseToMoE safetensor expander: Dense FFN → Sparse MoE (arXiv:2212.05055).

Converts each dense FFN layer into a MoE layer by:
  - Copying gate_proj/up_proj/down_proj as ``mlp.experts.{0..N-1}.*``
  - Creating a zero-initialized router ``mlp.gate.weight``
  - Adding noise to expert copies 1..N-1 for symmetry breaking

Non-FP: router is randomly initialized, requires CPT after expansion.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_grow.safetensor.base import ExpansionPlan, SafetensorExpanderBase, TensorRecipe
from llm_grow.safetensor.utils import ShardIndex, parse_layer_idx, read_safetensors_header

_FFN_SUFFIXES = frozenset({
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.bias",
    "mlp.up_proj.bias",
    "mlp.down_proj.bias",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    "mlp.fc1.bias",
    "mlp.fc2.bias",
})


@dataclass
class DenseToMoESafetensorConfig:
    num_experts: int = 8
    """Number of experts (FFN copies per layer)."""

    top_k: int = 2
    """Number of experts activated per token."""

    noise_std: float = 0.01
    """Noise std for symmetry breaking on expert copies 1..N-1."""

    router_suffix: str = "mlp.gate.weight"
    """Layer-suffix for the new router weight tensor."""

    config_expert_count_key: str = "num_experts"
    """config.json key for expert count."""

    config_topk_key: str = "num_experts_per_tok"
    """config.json key for top-k."""


class DenseToMoESafetensorExpander(SafetensorExpanderBase):
    """Convert dense FFN to MoE at the safetensor level.

    Example::

        from llm_grow.safetensor.methods.dense_to_moe import (
            DenseToMoESafetensorConfig, DenseToMoESafetensorExpander,
        )
        cfg = DenseToMoESafetensorConfig(num_experts=8, top_k=2)
        DenseToMoESafetensorExpander(cfg).expand(
            src_dir="Qwen/Qwen3-8B",
            dst_dir="./outputs/qwen3_moe",
        )
    """

    def __init__(self, config: DenseToMoESafetensorConfig | None = None) -> None:
        self.config = config or DenseToMoESafetensorConfig()

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        cfg = self.config
        wmap = src_index.weight_map
        suffixes = src_index.layer_suffixes()
        num_layers = src_index.num_hidden_layers()

        hidden_size = _get_hidden_size(src_index)

        plan = ExpansionPlan(
            new_num_hidden_layers=num_layers,
            config_patches={
                cfg.config_expert_count_key: cfg.num_experts,
                cfg.config_topk_key: cfg.top_k,
            },
        )

        ffn_suffixes = _FFN_SUFFIXES & set(suffixes)

        for new_idx in range(num_layers):
            for suf in suffixes:
                src_key = f"model.layers.{new_idx}.{suf}"
                if src_key not in wmap:
                    continue

                if suf in ffn_suffixes:
                    expert_suf = suf.replace("mlp.", "mlp.experts.{i}.", 1)
                    for i in range(cfg.num_experts):
                        expert_key = f"model.layers.{new_idx}.{expert_suf.format(i=i)}"
                        plan.add(
                            expert_key,
                            TensorRecipe(
                                src_shard=wmap[src_key],
                                src_key=src_key,
                                add_noise_std=cfg.noise_std if i > 0 else 0.0,
                            ),
                        )
                else:
                    plan.passthrough(src_key, wmap[src_key])

            router_key = f"model.layers.{new_idx}.{cfg.router_suffix}"
            plan.add(
                router_key,
                TensorRecipe(
                    src_shard="",
                    src_key="",
                    create_shape=(cfg.num_experts, hidden_size),
                    create_dtype="BF16",
                ),
            )

        self._passthrough_non_layer_keys(plan, wmap)
        return plan


def _get_hidden_size(src_index: ShardIndex) -> int:
    """Infer hidden_size from q_proj or embed shape in layer 0."""
    for key in src_index.weight_map:
        if key.endswith("self_attn.q_proj.weight") and key.startswith("model.layers.0."):
            shard_path = src_index.model_dir / src_index.weight_map[key]
            header = read_safetensors_header(shard_path)
            if key in header:
                _dtype, shape = header[key]
                return shape[1]
    for key in src_index.weight_map:
        if key == "model.embed_tokens.weight":
            shard_path = src_index.model_dir / src_index.weight_map[key]
            header = read_safetensors_header(shard_path)
            if key in header:
                _dtype, shape = header[key]
                return shape[1]
    return 4096
