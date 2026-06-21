"""Structural verification of expanded safetensor models.

Library-level functions for verifying expansion correctness without
loading full models. Use programmatically or via ``llm-grow verify``.

All checks operate on safetensor headers and mmap — no GPU required,
memory usage is minimal even for 100B+ models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from safetensors import safe_open

from llm_grow.configs.constants import (
    DEFAULT_VERIFY_ATOL,
    DEFAULT_VERIFY_NUM_SAMPLES,
    DEFAULT_VERIFY_SEED,
    DEFAULT_VERIFY_SEQ_LEN,
    WEIGHT_PRESERVE_ATOL,
    ZERO_CHECK_ATOL,
)
from llm_grow.safetensor.utils import ShardIndex, peek_model_config
from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


class StructuralVerifier:
    """Run structural verification checks on an expanded model.

    Usage::

        verifier = StructuralVerifier(
            src_dir="/models/orig", dst_dir="/models/expanded",
        )
        results = verifier.run_all()
        assert all(results.values())
    """

    def __init__(self, src_dir: str | Path, dst_dir: str | Path) -> None:
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.src_idx = ShardIndex.load(self.src_dir)
        self.dst_idx = ShardIndex.load(self.dst_dir)

    def run_all(self) -> dict[str, bool]:
        """Run all structural checks and return results dict."""
        results: dict[str, bool] = {}
        results["config"] = self.check_config()
        results["tensor_counts"] = self.check_tensor_counts()
        results["weights_preserved"] = self.check_original_weights_preserved()
        results["identity_zeroed"] = self.check_identity_blocks_zeroed()
        return results

    def check_config(self) -> bool:
        """Compare config.json between source and destination.

        Returns True if the destination config is a valid expansion of the source.
        Identical configs are accepted (passthrough/no-op). If configs differ,
        at least one known expansion key must have changed, and layer counts must
        remain consistent with the weight map.
        """
        src_cfg = peek_model_config(self.src_dir)
        dst_cfg = peek_model_config(self.dst_dir)

        keys = sorted(set(src_cfg) | set(dst_cfg))
        diffs: list[str] = []
        changed_keys: set[str] = set()
        for k in keys:
            sv, dv = src_cfg.get(k), dst_cfg.get(k)
            if sv != dv:
                diffs.append(f"{k}: {sv} -> {dv}")
                changed_keys.add(k)

        if not diffs:
            logger.info("Config: no changes")
            return True

        expected_keys = {
            "num_hidden_layers",
            "num_layers",
            "num_experts",
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_topk",
            "intermediate_size",
            "hidden_size",
            "moe_intermediate_size",
            "zero_expert_num",
            "expert_expansion_factor",
            "use_group_routing",
        }
        if not (changed_keys & expected_keys):
            logger.warning(
                "Config changed but no expected expansion key modified: %s",
                "; ".join(diffs),
            )
            return False

        logger.info("Config diff: %s", "; ".join(diffs))

        # Verify layer count consistency with weight map
        src_layers = self.src_idx.num_hidden_layers()
        dst_layers = self.dst_idx.num_hidden_layers()
        if src_layers != dst_layers:
            layer_key = next(
                (k for k in ("num_hidden_layers", "num_layers") if k in dst_cfg),
                None,
            )
            if layer_key is None or dst_cfg.get(layer_key) != dst_layers:
                logger.error(
                    "Config layer count %s=%s inconsistent with weight map (%d layers)",
                    layer_key,
                    dst_cfg.get(layer_key),
                    dst_layers,
                )
                return False

        return True

    def check_tensor_counts(self) -> bool:
        """Verify output has the expected number of tensors."""
        src_layers = self.src_idx.num_hidden_layers()
        dst_layers = self.dst_idx.num_hidden_layers()
        per_layer = len(self.src_idx.layer_suffixes())
        expected = (
            len(self.src_idx.all_keys) - src_layers * per_layer + dst_layers * per_layer
        )
        actual = len(self.dst_idx.all_keys)
        ok = actual == expected
        logger.info(
            "Tensor counts: src=%d, dst=%d (expected %d) %s",
            len(self.src_idx.all_keys),
            actual,
            expected,
            "OK" if ok else "MISMATCH",
        )
        return ok

    def check_original_weights_preserved(self, sample: int = 4) -> bool:
        """Spot-check that original layer tensors are preserved in output.

        Uses layer offset mapping for O(added) candidate lookup instead of
        brute-force scanning all destination layers.  Shards are opened
        lazily and closed after the check to avoid holding many file handles.
        """
        wmap_src = self.src_idx.weight_map
        wmap_dst = self.dst_idx.weight_map

        src_layers = self.src_idx.num_hidden_layers()
        dst_layers = self.dst_idx.num_hidden_layers()
        suf = "mlp.gate_proj.weight"

        added = dst_layers - src_layers

        src_handles: dict[str, Any] = {}
        dst_handles: dict[str, Any] = {}

        def _get_src_handle(shard: str) -> Any:
            if shard not in src_handles:
                src_handles[shard] = safe_open(
                    str(self.src_idx.model_dir / shard), framework="pt", device="cpu"
                )
            return src_handles[shard]

        def _get_dst_handle(shard: str) -> Any:
            if shard not in dst_handles:
                dst_handles[shard] = safe_open(
                    str(self.dst_idx.model_dir / shard), framework="pt", device="cpu"
                )
            return dst_handles[shard]

        try:
            step = max(1, src_layers // sample)
            all_ok = True
            for orig_idx in range(0, src_layers, step):
                src_key = f"model.layers.{orig_idx}.{suf}"
                if src_key not in wmap_src:
                    continue
                src_t = (
                    _get_src_handle(wmap_src[src_key]).get_tensor(src_key).float()
                )

                best_idx, best_diff = -1, float("inf")
                if added >= 0:
                    search_indices = range(
                        orig_idx, min(orig_idx + added + 1, dst_layers)
                    )
                else:
                    search_indices = range(dst_layers)

                for dst_i in search_indices:
                    dst_key = f"model.layers.{dst_i}.{suf}"
                    if dst_key not in wmap_dst:
                        continue
                    dst_t = (
                        _get_dst_handle(wmap_dst[dst_key])
                        .get_tensor(dst_key)
                        .float()
                    )
                    if dst_t.shape != src_t.shape:
                        continue
                    diff = (src_t - dst_t).abs().max().item()
                    if diff < best_diff:
                        best_diff, best_idx = diff, dst_i
                    if best_diff < WEIGHT_PRESERVE_ATOL:
                        break

                if best_idx == -1:
                    logger.info(
                        "  layer %d: shapes differ (width expansion)", orig_idx
                    )
                else:
                    ok = best_diff < WEIGHT_PRESERVE_ATOL
                    logger.info(
                        "  orig layer %d -> dst layer %d  max|diff|=%.2e %s",
                        orig_idx,
                        best_idx,
                        best_diff,
                        "OK" if ok else "FAIL",
                    )
                    all_ok = all_ok and ok
            return all_ok
        finally:
            for handle in list(src_handles.values()) + list(dst_handles.values()):
                handle.__exit__(None, None, None)
            src_handles.clear()
            dst_handles.clear()

    def check_identity_blocks_zeroed(self) -> bool:
        """Verify that identity blocks have zeroed output projections."""
        from llm_grow.safetensor.detect import detect_model

        profile = detect_model(self.src_dir)
        zero_suffixes = set(
            profile.attn_zero_suffixes + profile.dense_mlp_zero_suffixes
        )
        if not zero_suffixes:
            zero_suffixes = {"self_attn.o_proj.weight", "mlp.down_proj.weight"}

        # Lazy shard handle cache: open only the shards we actually touch.
        dst_handles: dict[str, Any] = {}

        def _get_handle(shard: str) -> Any:
            if shard not in dst_handles:
                dst_handles[shard] = safe_open(
                    str(self.dst_idx.model_dir / shard), framework="pt", device="cpu"
                )
            return dst_handles[shard]

        try:
            total_zero = total_nonzero = 0
            for suf in zero_suffixes:
                for key, shard in self.dst_idx.weight_map.items():
                    if not key.endswith(suf):
                        continue
                    t = _get_handle(shard).get_tensor(key)
                    if t.abs().max().item() < ZERO_CHECK_ATOL:
                        total_zero += 1
                    else:
                        total_nonzero += 1

            if total_zero == 0:
                logger.info(
                    "Identity blocks: none found (non-FP method or no identity blocks)"
                )
                return True

            ok = total_nonzero == 0
            logger.info(
                "Identity blocks: %d zeroed, %d non-zero (original layers) %s",
                total_zero,
                total_nonzero,
                "OK" if ok else "FAIL",
            )
            return ok
        finally:
            for handle in list(dst_handles.values()):
                handle.__exit__(None, None, None)
            dst_handles.clear()


def check_fp(
    src_dir: Path,
    dst_dir: Path,
    seq_len: int = DEFAULT_VERIFY_SEQ_LEN,
    samples: int = DEFAULT_VERIFY_NUM_SAMPLES,
    atol: float = DEFAULT_VERIFY_ATOL,
    seed: int = DEFAULT_VERIFY_SEED,
    max_size_gb: float = 80.0,
) -> bool:
    """Full function-preserving check: load both models and compare logits.

    Requires both models to fit in memory. Use for models <= ~30B parameters.
    Delegates to ``verify_fp`` from ``llm_grow.eval.fp_verifier``.
    """
    from llm_grow.eval.fp_verifier import verify_fp

    return verify_fp(
        str(src_dir),
        str(dst_dir),
        num_samples=samples,
        seq_len=seq_len,
        atol=atol,
        seed=seed,
        max_size_gb=max_size_gb,
    )
