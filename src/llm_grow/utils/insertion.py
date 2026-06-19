"""Shared utilities for computing layer insertion positions."""

from __future__ import annotations


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
