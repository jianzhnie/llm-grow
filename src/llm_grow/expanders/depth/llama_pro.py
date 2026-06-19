"""Backward-compatible re-export.

This module has been renamed to ``identity_graft``.
All public names are re-exported here for backward compatibility.

See Also:
    :mod:`llm_grow.expanders.depth.identity_graft`
"""

from llm_grow.expanders.depth.identity_graft import *  # noqa: F401,F403
from llm_grow.expanders.depth.identity_graft import (  # noqa: F401
    _compute_insert_positions,
    _get_decoder_layers,
    _make_identity_block,
    _set_decoder_layers,
    _update_num_hidden_layers,
)
