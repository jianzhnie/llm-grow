"""Shared utilities for computing layer insertion positions and sequences."""

from __future__ import annotations

from typing import Literal

from llm_grow.core.markers import DECODER_LAYER_ATTRS, NEW_GROWTH_ATTR

InsertStrategy = Literal["uniform", "front", "rear"]
"""Layer insertion strategy for depth expansion."""

__all__ = [
    "DECODER_LAYER_ATTRS",
    "NEW_GROWTH_ATTR",
    "InsertStrategy",
    "build_layer_sequence",
    "insert_positions",
]


def build_layer_sequence(num_orig: int, insert_pos: set[int]) -> list[tuple[int, bool]]:
    """Build an ordered layer sequence with identity block markers.

    Returns:
        List of ``(src_layer_idx, is_identity)`` tuples.
    """
    invalid = {p for p in insert_pos if p < 0 or p >= num_orig}
    if invalid:
        raise ValueError(
            f"Insert positions out of range [0, {num_orig}): {sorted(invalid)}"
        )
    sequence: list[tuple[int, bool]] = []
    for i in range(num_orig):
        sequence.append((i, False))
        if i in insert_pos:
            sequence.append((i, True))
    return sequence


def insert_positions(
    num_orig: int, num_new: int, strategy: InsertStrategy
) -> list[int]:
    """Compute layer insertion positions for depth expansion.

    Strategies:
      - 'uniform': evenly spaced (best general-purpose choice)
      - 'front':   insert at the beginning
      - 'rear':    insert at the end
    """
    if num_new <= 0:
        return []
    if num_new > num_orig:
        raise ValueError(
            f"num_new_layers ({num_new}) cannot exceed num_orig_layers ({num_orig})."
        )
    if strategy == "uniform":
        if num_new >= num_orig:
            return list(range(num_orig))
        positions = []
        for i in range(num_new):
            pos = int((i + 0.5) * num_orig / num_new)
            positions.append(pos)

        # Resolve rare rounding collisions by shifting forward.
        seen: set[int] = set()
        result: list[int] = []
        for p in positions:
            while p in seen:
                p += 1
            if p >= num_orig:
                raise RuntimeError(
                    "Uniform insertion position overflow. "
                    "This should not happen when num_new < num_orig."
                )
            seen.add(p)
            result.append(p)
        return sorted(result)
    if strategy == "front":
        return list(range(num_new))
    if strategy == "rear":
        return list(range(num_orig - num_new, num_orig))
    raise ValueError(f"Unknown insert strategy: {strategy!r}")
