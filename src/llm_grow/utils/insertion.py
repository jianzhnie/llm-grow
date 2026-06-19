"""Shared utilities for computing layer insertion positions and sequences."""

from __future__ import annotations

NEW_GROWTH_ATTR = "_is_new_growth"
"""Canonical attribute name used to tag newly-grown parameters."""

DECODER_LAYER_ATTRS = ("layers", "model.layers", "transformer.h", "decoder.layers")
"""Common attribute paths for decoder layer lists in HuggingFace models."""


def build_layer_sequence(num_orig: int, insert_pos: set[int]) -> list[tuple[int, bool]]:
    """Build an ordered layer sequence with identity block markers.

    Returns:
        List of ``(src_layer_idx, is_identity)`` tuples.
    """
    sequence: list[tuple[int, bool]] = []
    for i in range(num_orig):
        sequence.append((i, False))
        if i in insert_pos:
            sequence.append((i, True))
    return sequence


def insert_positions(num_orig: int, num_new: int, strategy: str) -> list[int]:
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
        step = num_orig / (num_new + 1)
        positions = sorted({round(step * (i + 1)) - 1 for i in range(num_new)})
        if len(positions) < num_new:
            import warnings

            warnings.warn(
                f"Uniform insertion produced {len(positions)} unique positions "
                f"(requested {num_new}). Consider reducing num_new_layers.",
                stacklevel=2,
            )
        return positions
    if strategy == "front":
        return list(range(num_new))
    if strategy == "rear":
        return list(range(num_orig - num_new, num_orig))
    raise ValueError(f"Unknown insert strategy: {strategy!r}")
