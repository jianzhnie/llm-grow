"""Cross-layer markers and constants used by expanders and training.

These values are part of the public contract between model expansion and
subsequent training/fine-tuning.  Keeping them in a dedicated core module
prevents the training layer from depending on expansion utilities.
"""

from __future__ import annotations

NEW_GROWTH_ATTR = "_is_new_growth"
"""Canonical attribute name used to tag newly-grown parameters."""

DECODER_LAYER_ATTRS = ("layers", "model.layers", "transformer.h", "decoder.layers")
"""Common attribute paths for decoder layer lists in HuggingFace models."""
