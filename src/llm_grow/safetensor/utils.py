"""ShardIndex and tensor key utilities for safetensor-level expansion."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open

# ── tensor key helpers ──────────────────────────────────────────────────────

_LAYER_RE = re.compile(r"^(model\.layers\.)(\d+)(\..*)")


def parse_layer_idx(key: str) -> int | None:
    """Return layer index from 'model.layers.{i}.xxx', else None."""
    m = _LAYER_RE.match(key)
    return int(m.group(2)) if m else None


def rename_layer_idx(key: str, new_idx: int) -> str:
    """Replace the layer index in a tensor key."""
    return _LAYER_RE.sub(lambda m: f"{m.group(1)}{new_idx}{m.group(3)}", key)


def layer_suffix(key: str) -> str | None:
    """Return the part after 'model.layers.{i}.' or None for non-layer keys."""
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
        return len(self.shard_files) == 1 and self.shard_files[0] == self.SINGLE_FILENAME

    def total_size_bytes(self) -> int:
        return sum((self.model_dir / sf).stat().st_size for sf in self.shard_files)

    # ── access ───────────────────────────────────────────────────────────────

    def open_all_shards(self) -> dict[str, safe_open]:
        """Open all shards with mmap.  Caller responsible for resource lifecycle."""
        return {
            sf: safe_open(str(self.model_dir / sf), framework="pt", device="cpu") for sf in self.shard_files
        }

    def layer_suffixes(self) -> list[str]:
        """Return sorted list of per-layer tensor suffixes (from any layer)."""
        seen: set[str] = set()
        for key in self.weight_map:
            suf = layer_suffix(key)
            if suf is not None:
                seen.add(suf)
        return sorted(seen)

    def num_hidden_layers(self) -> int:
        """Infer num_hidden_layers from tensor keys."""
        indices = {parse_layer_idx(k) for k in self.weight_map if parse_layer_idx(k) is not None}
        return max(indices) + 1 if indices else 0

    # ── write helpers ────────────────────────────────────────────────────────

    def write_index_json(self, dst_dir: Path) -> None:
        """Write model.safetensors.index.json for multi-shard output."""
        index = {"metadata": {"total_size": 0}, "weight_map": self.weight_map}
        with open(dst_dir / self.INDEX_FILENAME, "w") as f:
            json.dump(index, f, indent=2)

    # ── misc ─────────────────────────────────────────────────────────────────

    def copy_non_weight_files(self, dst_dir: Path) -> None:
        """Copy tokenizer, generation_config, README, etc. (skip weights)."""
        skip_suffixes = {".safetensors"}
        skip_names = {"config.json", self.INDEX_FILENAME}
        for src_file in self.model_dir.iterdir():
            if src_file.is_dir():
                continue
            if src_file.suffix in skip_suffixes:
                continue
            if src_file.name in skip_names:
                continue
            dst_file = dst_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
