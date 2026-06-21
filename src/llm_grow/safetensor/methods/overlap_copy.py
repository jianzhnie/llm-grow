"""OverlapCopy safetensor expander: depth up-scaling via layer overlap-copy.

Non-FP method.  Requires 100B+ continued pretraining to recover accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_grow.configs.base import ExpansionConfig
from llm_grow.safetensor.base import ExpansionPlan, SafetensorExpanderBase
from llm_grow.safetensor.utils import ShardIndex
from llm_grow.utils.expansion_rules import build_overlap_sequence


@dataclass
class OverlapCopySafetensorConfig(ExpansionConfig):
    num_overlap: int = 8
    """Number of overlapping layers.

    Upper copy  = layers [0  ..  L-overlap-1]
    Lower copy  = layers [overlap .. L-1]
    Result size = 2 * (L - overlap) layers
    """


class OverlapCopySafetensorExpander(SafetensorExpanderBase):
    """OverlapCopy depth up-scaling directly on safetensor files.

    Example::

        OverlapCopySafetensorExpander(OverlapCopySafetensorConfig(num_overlap=8)).expand(
            src_dir="Qwen/Qwen3-8B",
            dst_dir="./outputs/qwen3_overlap_copy",
        )
    """

    def __init__(self, config: OverlapCopySafetensorConfig | None = None) -> None:
        self.config = config or OverlapCopySafetensorConfig()

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        num_orig = src_index.num_hidden_layers()
        overlap = self.config.num_overlap

        sequence = build_overlap_sequence(num_orig, overlap)
        return self._build_layer_plan(src_index, layer_sequence=sequence)
