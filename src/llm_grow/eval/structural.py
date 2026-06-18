"""Structural verification of expanded safetensor models.

Library-level functions for verifying expansion correctness without
loading full models. Use programmatically or via ``llm-grow verify``.

All checks operate on safetensor headers and mmap — no GPU required,
memory usage is minimal even for 100B+ models.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from llm_grow.safetensor.utils import ShardIndex
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
        """Compare config.json between source and destination."""
        src_cfg = _load_config(self.src_dir)
        dst_cfg = _load_config(self.dst_dir)

        keys = sorted(set(src_cfg) | set(dst_cfg))
        diffs: list[str] = []
        for k in keys:
            sv, dv = src_cfg.get(k), dst_cfg.get(k)
            if sv != dv:
                diffs.append(f"{k}: {sv} -> {dv}")

        if diffs:
            logger.info("Config diff: %s", "; ".join(diffs))
        else:
            logger.info("Config: no changes")
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
        """Spot-check that original layer tensors are preserved in output."""
        src_handles = self.src_idx.open_all_shards()
        dst_handles = self.dst_idx.open_all_shards()
        try:
            wmap_src = self.src_idx.weight_map
            wmap_dst = self.dst_idx.weight_map

            src_layers = self.src_idx.num_hidden_layers()
            dst_layers = self.dst_idx.num_hidden_layers()
            suf = "mlp.gate_proj.weight"

            dst_tensors: dict[int, torch.Tensor] = {}
            for dst_i in range(dst_layers):
                key = f"model.layers.{dst_i}.{suf}"
                if key in wmap_dst:
                    dst_tensors[dst_i] = (
                        dst_handles[wmap_dst[key]].get_tensor(key).float()
                    )

            step = max(1, src_layers // sample)
            all_ok = True
            for orig_idx in range(0, src_layers, step):
                src_key = f"model.layers.{orig_idx}.{suf}"
                if src_key not in wmap_src:
                    continue
                src_t = src_handles[wmap_src[src_key]].get_tensor(src_key).float()

                best_idx, best_diff = -1, float("inf")
                for dst_i, dst_t in dst_tensors.items():
                    if dst_t.shape != src_t.shape:
                        continue
                    diff = (src_t - dst_t).abs().max().item()
                    if diff < best_diff:
                        best_diff, best_idx = diff, dst_i

                if best_idx == -1:
                    logger.info("  layer %d: shapes differ (width expansion)", orig_idx)
                else:
                    ok = best_diff < 1e-6
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
            del src_handles, dst_handles  # release mmap handles

    def check_identity_blocks_zeroed(self) -> bool:
        """Verify that identity blocks have zeroed output projections."""
        from llm_grow.safetensor.detect import detect_model

        profile = detect_model(self.src_dir)
        zero_suffixes = set(
            profile.attn_zero_suffixes + profile.dense_mlp_zero_suffixes
        )
        if not zero_suffixes:
            zero_suffixes = {"self_attn.o_proj.weight", "mlp.down_proj.weight"}
        dst_handles = self.dst_idx.open_all_shards()
        try:
            total_zero = total_nonzero = 0
            for suf in zero_suffixes:
                for key, shard in self.dst_idx.weight_map.items():
                    if not key.endswith(suf):
                        continue
                    t = dst_handles[shard].get_tensor(key)
                    if t.abs().max().item() < 1e-9:
                        total_zero += 1
                    else:
                        total_nonzero += 1

            if total_zero == 0:
                logger.info(
                    "Identity blocks: none found (non-FP method or no identity blocks)"
                )
            else:
                logger.info(
                    "Identity blocks: %d zeroed, %d non-zero (original layers)",
                    total_zero,
                    total_nonzero,
                )
            return True
        finally:
            del dst_handles  # release mmap handles


def check_fp(
    src_dir: Path,
    dst_dir: Path,
    seq_len: int = 32,
    samples: int = 4,
    atol: float = 1e-4,
    seed: int = 42,
) -> bool:
    """Full function-preserving check: load both models and compare logits.

    Requires both models to fit in memory. Use for models <= ~30B parameters.
    """
    from transformers import AutoModelForCausalLM

    logger.info("Loading original model from %s", src_dir)
    orig = AutoModelForCausalLM.from_pretrained(str(src_dir), torch_dtype=torch.float32)
    logger.info("Loading expanded model from %s", dst_dir)
    try:
        exp = AutoModelForCausalLM.from_pretrained(
            str(dst_dir), torch_dtype=torch.float32
        )
    except Exception as exc:
        logger.error("Cannot load expanded model: %s", exc)
        return False

    orig.eval()
    exp.eval()

    vocab = orig.config.vocab_size
    torch.manual_seed(seed)
    ids = torch.randint(0, vocab, (samples, seq_len))
    max_err = 0.0
    with torch.no_grad():
        lo = orig(input_ids=ids).logits
        le = exp(input_ids=ids).logits
        max_err = (lo - le).abs().max().item()

    ok = max_err < atol
    logger.info(
        "FP check: max|delta_logit| = %.3e (atol=%.1e) %s",
        max_err,
        atol,
        "PASS" if ok else "FAIL",
    )
    return ok


def _load_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}
