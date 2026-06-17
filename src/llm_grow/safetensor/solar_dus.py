"""SOLAR DUS safetensor expander: depth up-scaling via layer overlap-copy.

Non-FP method.  Requires 100B+ continued pretraining to recover accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_grow.safetensor.base import ExpansionPlan, SafetensorExpanderBase
from llm_grow.safetensor.utils import ShardIndex


@dataclass
class SolarDUSSafetensorConfig:
    num_overlap: int = 8
    """Number of overlapping layers.

    Upper copy  = layers [0  ..  L-overlap-1]
    Lower copy  = layers [overlap .. L-1]
    Result size = 2 * (L - overlap) layers
    """


class SolarDUSSafetensorExpander(SafetensorExpanderBase):
    """SOLAR DUS depth up-scaling directly on safetensor files.

    Example::

        SolarDUSSafetensorExpander(SolarDUSSafetensorConfig(num_overlap=8)).expand(
            src_dir="Qwen/Qwen3-8B",
            dst_dir="./outputs/qwen3_solar_dus",
        )
    """

    def __init__(self, config: SolarDUSSafetensorConfig | None = None) -> None:
        self.config = config or SolarDUSSafetensorConfig()

    def _build_plan(self, src_index: ShardIndex) -> ExpansionPlan:
        num_orig = src_index.num_hidden_layers()
        overlap = self.config.num_overlap

        if overlap >= num_orig:
            raise ValueError(f"num_overlap ({overlap}) must be < num_hidden_layers ({num_orig}).")

        upper_end = num_orig - overlap  # upper copy: layers 0 .. upper_end-1
        lower_start = overlap  # lower copy: layers lower_start .. num_orig-1

        sequence: list[tuple[int, bool]] = [(i, False) for i in range(upper_end)] + [
            (i, False) for i in range(lower_start, num_orig)
        ]

        return self._build_layer_plan(src_index, layer_sequence=sequence)
