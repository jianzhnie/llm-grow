"""Abstract model inspector interface for verification and tooling.

The inspector abstracts away storage-format details (safetensors, PyTorch
state dicts, GGUF, etc.) so that higher-level code in ``llm_grow.eval`` can
operate on a model without loading it fully into RAM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any


class ModelInspector(ABC):
    """Read-only view of a serialized model.

    Implementations are responsible for opening files lazily and releasing
    resources when ``close()`` is called or the context manager exits.
    """

    @property
    @abstractmethod
    def model_dir(self) -> Path:
        """Directory containing the serialized model."""

    @abstractmethod
    def peek_config(self) -> dict[str, Any]:
        """Return the model's config dict (e.g. ``config.json``)."""

    @abstractmethod
    def num_hidden_layers(self) -> int:
        """Return the number of hidden/decoder layers."""

    @abstractmethod
    def layer_suffixes(self) -> list[str]:
        """Return the per-layer tensor suffixes present in the model."""

    @abstractmethod
    def all_keys(self) -> list[str]:
        """Return all tensor keys in the model."""

    @abstractmethod
    def weight_map(self) -> dict[str, str]:
        """Return mapping from tensor key to shard/file name."""

    @abstractmethod
    def get_tensor(self, key: str) -> Any:
        """Return the tensor associated with ``key``.

        The concrete return type is implementation-specific (typically
        ``torch.Tensor``), so the abstract signature uses ``Any``.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any open file handles or resources."""

    def __enter__(self) -> ModelInspector:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        self.close()
