"""ShardIndex and tensor key utilities for safetensor-level expansion."""

from __future__ import annotations

import json
import os
import re
import shutil
from functools import cached_property
from pathlib import Path

from safetensors import safe_open

from llm_grow.configs.constants import DEFAULT_TARGET_SHARD_BYTES
from llm_grow.utils.insertion import insert_positions  # Re-exported for compat
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "DTYPE_SIZES",
    "ShardIndex",
    "auto_detect_shard_size",
    "expert_idx",
    "expert_key_offset",
    "get_hidden_size_from_index",
    "insert_positions",
    "is_expert_key",
    "layer_suffix",
    "nbytes_from_header",
    "parse_layer_idx",
    "peek_model_config",
    "read_safetensors_header",
    "rename_layer_idx",
]

# ── dtype → bytes per element (mirrors safetensors dtype strings) ─────────────
DTYPE_SIZES: dict[str, int] = {
    "F64": 8,
    "I64": 8,
    "F32": 4,
    "I32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

# ── tensor key helpers ──────────────────────────────────────────────────────

_LAYER_RE = re.compile(
    r"^((?:model\.layers|transformer\.h|decoder\.layers)\.)(\d+)(\..*)"
)


def parse_layer_idx(key: str) -> int | None:
    """Return layer index from common decoder layer keys, else None.

    Supports ``model.layers.{i}.xxx``, ``transformer.h.{i}.xxx``,
    and ``decoder.layers.{i}.xxx``.
    """
    m = _LAYER_RE.match(key)
    return int(m.group(2)) if m else None


def rename_layer_idx(key: str, new_idx: int) -> str:
    """Replace the layer index in a tensor key while preserving the prefix."""
    return _LAYER_RE.sub(lambda m: f"{m.group(1)}{new_idx}{m.group(3)}", key)


def layer_suffix(key: str) -> str | None:
    """Return the part after the layer index prefix, or None for non-layer keys."""
    m = _LAYER_RE.match(key)
    return m.group(3)[1:] if m else None  # strip leading dot


# ── ShardIndex ───────────────────────────────────────────────────────────────


class ShardIndex:
    """Abstracts single-file and multi-shard safetensor models.

    Provides memory-mapped access to tensors: only the bytes you request
    are loaded from disk, making it safe to use with 100B+ models.
    """

    INDEX_FILENAME = "model.safetensors.index.json"
    SINGLE_FILENAME = "model.safetensors"

    def __init__(self, model_dir: Path, weight_map: dict[str, str]) -> None:
        self.model_dir = Path(model_dir)
        # tensor_name -> shard_filename (basename only)
        self.weight_map: dict[str, str] = weight_map

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def load(cls, model_dir: str | Path) -> ShardIndex:
        """Load from model directory (auto-detects single vs. sharded)."""
        model_dir = Path(model_dir)
        index_path = model_dir / cls.INDEX_FILENAME
        single_path = model_dir / cls.SINGLE_FILENAME

        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            return cls(model_dir, data["weight_map"])

        if single_path.exists():
            with safe_open(str(single_path), framework="pt", device="cpu") as f:
                weight_map = dict.fromkeys(f.keys(), cls.SINGLE_FILENAME)
            return cls(model_dir, weight_map)

        raise FileNotFoundError(
            f"No safetensor files found in {model_dir}. "
            f"Expected {cls.SINGLE_FILENAME} or {cls.INDEX_FILENAME}."
        )

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def all_keys(self) -> list[str]:
        return list(self.weight_map.keys())

    @property
    def shard_files(self) -> list[str]:
        return sorted(set(self.weight_map.values()))

    @property
    def is_single_shard(self) -> bool:
        return (
            len(self.shard_files) == 1 and self.shard_files[0] == self.SINGLE_FILENAME
        )

    def total_size_bytes(self) -> int:
        return sum((self.model_dir / sf).stat().st_size for sf in self.shard_files)

    # ── access ───────────────────────────────────────────────────────────────

    def open_all_shards(self) -> dict[str, safe_open]:
        """Open all shards with mmap.  Caller responsible for resource lifecycle."""
        return {
            sf: safe_open(str(self.model_dir / sf), framework="pt", device="cpu")
            for sf in self.shard_files
        }

    def layer_suffixes(self) -> list[str]:
        """Return sorted list of per-layer tensor suffixes (from any layer)."""
        seen: set[str] = set()
        for key in self.weight_map:
            suf = layer_suffix(key)
            if suf is not None:
                seen.add(suf)
        return sorted(seen)

    @cached_property
    def _num_hidden_layers(self) -> int:
        """Infer and cache num_hidden_layers from tensor keys.

        Counts unique layer indices rather than assuming contiguity
        (``max(idx) + 1``), so models with non-consecutive layer
        numbering are handled correctly.
        """
        indices: set[int] = set()
        for k in self.weight_map:
            idx = parse_layer_idx(k)
            if idx is not None:
                indices.add(idx)
        return len(indices)

    def num_hidden_layers(self) -> int:
        """Return num_hidden_layers (cached after first call)."""
        return self._num_hidden_layers

    # ── write helpers ────────────────────────────────────────────────────────

    def write_index_json(self, dst_dir: Path) -> None:
        """Write model.safetensors.index.json for multi-shard output."""
        shard_files = set(self.weight_map.values())
        missing = [sf for sf in shard_files if not (dst_dir / sf).exists()]
        if missing:
            raise FileNotFoundError(
                f"Cannot write index: {len(missing)} shard(s) missing: {missing}"
            )
        total_size = sum((dst_dir / sf).stat().st_size for sf in shard_files)
        index = {"metadata": {"total_size": total_size}, "weight_map": self.weight_map}
        index_path = dst_dir / self.INDEX_FILENAME
        tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(index, f, indent=2)
        os.replace(str(tmp_path), str(index_path))

    # ── misc ─────────────────────────────────────────────────────────────────

    def copy_non_weight_files(self, dst_dir: Path) -> None:
        """Copy all auxiliary files from source to dst_dir.

        Always overwrites existing files so that ``configuration_*.py``,
        ``modeling_*.py`` and other code referenced by ``auto_map`` in
        ``config.json`` stay in sync with the source model.

        Skipped:
          - ``*.safetensors`` weight files (written separately)
          - ``config.json``  (written by ``_write_config``)
          - ``model.safetensors.index.json`` (written by ``write_index_json``)
          - directories (e.g. ``__pycache__``)
        """
        skip_suffixes = {".safetensors"}
        skip_names = {"config.json", self.INDEX_FILENAME}

        py_files: list[str] = []
        other_files: list[str] = []

        for src_file in sorted(self.model_dir.iterdir()):
            if src_file.is_dir():
                continue
            if src_file.suffix in skip_suffixes:
                continue
            if src_file.name in skip_names:
                continue
            shutil.copy2(src_file, dst_dir / src_file.name)  # always overwrite
            if src_file.suffix == ".py":
                py_files.append(src_file.name)
            else:
                other_files.append(src_file.name)

        if py_files:
            logger.info("Copied Python files (modeling/config code): %s", py_files)
        if other_files:
            logger.info("Copied auxiliary files: %s", other_files)


# ── header-only safetensors utilities ────────────────────────────────────────


def read_safetensors_header(path: Path) -> dict[str, tuple[str, list[int]]]:
    """Read only the JSON header of a safetensors file — no tensor data loaded.

    Safetensors layout:
      [8-byte little-endian header_size][header_size bytes of JSON][tensor data]

    Returns:
        {tensor_name: (dtype_string, shape_list)} for every tensor in the file.
    """
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    return {
        k: (v["dtype"], v["shape"]) for k, v in header.items() if k != "__metadata__"
    }


def nbytes_from_header(dtype: str, shape: list[int]) -> int:
    """Compute tensor byte size from safetensors metadata (no tensor load needed)."""
    elem = DTYPE_SIZES.get(dtype, 4)
    numel = 1
    for d in shape:
        numel *= d
    return elem * numel


def peek_model_config(model_dir: Path) -> dict:
    """Load config.json from a model directory; return {} if missing."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return dict(json.load(f))


def auto_detect_shard_size(model_dir: Path, shard_files: list[str]) -> int:
    """Infer target shard size from existing files (arithmetic mean).

    Falls back to 4 GB if no shard files are present on disk.
    """
    sizes = [
        (model_dir / sf).stat().st_size
        for sf in shard_files
        if (model_dir / sf).exists()
    ]
    if sizes:
        avg = int(sum(sizes) / len(sizes))
        logger.info(
            "Auto shard size: %.2f GB (mean of %d shards)", avg / 1e9, len(sizes)
        )
        return avg
    logger.info("No shard files found on disk — using 4 GB default")
    return DEFAULT_TARGET_SHARD_BYTES


# ── MoE expert key helpers ───────────────────────────────────────────────────

_EXPERT_RE = re.compile(r"^(.*\.mlp\.experts\.)(\d+)(\..*)$")


def is_expert_key(key: str) -> bool:
    """Check if a tensor key matches the ``mlp.experts.{i}.*`` pattern."""
    return bool(_EXPERT_RE.match(key))


def expert_idx(key: str) -> int:
    """Extract expert index from a key, or -1 if not an expert key."""
    m = _EXPERT_RE.match(key)
    return int(m.group(2)) if m else -1


def expert_key_offset(key: str, offset: int) -> str:
    """Rename expert index: ``experts.{i}.* → experts.{i+offset}.*``."""
    m = _EXPERT_RE.match(key)
    if m is None:
        return key
    return f"{m.group(1)}{int(m.group(2)) + offset}{m.group(3)}"


def get_hidden_size_from_index(src_index: ShardIndex) -> int:
    """Infer hidden_size from q_proj or embed_tokens shape (header-only).

    Tries ``self_attn.q_proj.weight`` in layer 0 first, then falls back
    to ``model.embed_tokens.weight``.  Returns 0 if neither is found.
    """
    for key in src_index.weight_map:
        if key.endswith("self_attn.q_proj.weight") and key.startswith(
            "model.layers.0."
        ):
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
    return 0
