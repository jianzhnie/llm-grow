"""Safetensor implementation of the ModelInspector abstraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from safetensors import safe_open

from llm_grow.core.inspector import ModelInspector
from llm_grow.safetensor.utils import ShardIndex, peek_model_config


class SafetensorModelInspector(ModelInspector):
    """Inspect a HuggingFace-style safetensor model without loading it fully."""

    def __init__(self, model_dir: str | Path) -> None:
        self._model_dir = Path(model_dir)
        self._index = ShardIndex.load(self._model_dir)
        self._config: dict[str, Any] | None = None
        self._handles: dict[str, Any] = {}

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def peek_config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = peek_model_config(self._model_dir)
        return self._config

    def num_hidden_layers(self) -> int:
        return self._index.num_hidden_layers()

    def layer_suffixes(self) -> list[str]:
        return self._index.layer_suffixes()

    def all_keys(self) -> list[str]:
        return list(self._index.all_keys)

    def weight_map(self) -> dict[str, str]:
        return dict(self._index.weight_map)

    def get_tensor(self, key: str) -> Any:
        shard = self._index.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                str(self._model_dir / shard), framework="pt", device="cpu"
            )
        return self._handles[shard].get_tensor(key)

    def close(self) -> None:
        for handle in list(self._handles.values()):
            handle.__exit__(None, None, None)
        self._handles.clear()
