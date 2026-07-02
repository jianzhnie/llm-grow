"""Integration tests for llm_grow.safetensor.shard_writer.ShardWriter.

These tests exercise the full I/O pipeline (Pass 1 header scanning, Pass 2
shard writing, resume, parallel writing, and post-write validation) using
real temporary safetensor files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save

from llm_grow.safetensor.recipe import ExpansionPlan, TensorRecipe
from llm_grow.safetensor.shard_writer import ShardWriter
from llm_grow.safetensor.utils import ShardIndex


def _make_tiny_source_model(tmp_path: Path) -> Path:
    """Create a tiny two-layer model on disk and return its directory."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 32,
        "intermediate_size": 64,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(128, 32),
        "model.norm.weight": torch.randn(32),
        "lm_head.weight": torch.randn(128, 32),
    }
    for i in range(2):
        prefix = f"model.layers.{i}."
        tensors[f"{prefix}input_layernorm.weight"] = torch.randn(32)
        tensors[f"{prefix}self_attn.q_proj.weight"] = torch.randn(32, 32)
        tensors[f"{prefix}self_attn.o_proj.weight"] = torch.randn(32, 32)
        tensors[f"{prefix}mlp.gate_proj.weight"] = torch.randn(64, 32)
        tensors[f"{prefix}mlp.up_proj.weight"] = torch.randn(64, 32)
        tensors[f"{prefix}mlp.down_proj.weight"] = torch.randn(32, 64)

    safetensors_save(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _mixed_plan(src_index: ShardIndex) -> ExpansionPlan:
    """Build a plan that exercises passthrough, zero-out, padding, dup_rows,
    and create_shape recipes.  This ensures resume validation covers shape
    changes, not just identity transforms.
    """
    plan = ExpansionPlan(new_num_hidden_layers=3)
    wmap = src_index.weight_map

    # Layer 0: copied unchanged.
    for suf in src_index.layer_suffixes():
        src_key = f"model.layers.0.{suf}"
        if src_key in wmap:
            plan.passthrough(f"model.layers.0.{suf}", wmap[src_key])

    # Layer 1: identity block — zero out o_proj and down_proj.
    for suf in src_index.layer_suffixes():
        src_key = f"model.layers.1.{suf}"
        if src_key not in wmap:
            continue
        zero = suf in {"self_attn.o_proj.weight", "mlp.down_proj.weight"}
        plan.add(
            f"model.layers.1.{suf}",
            TensorRecipe(src_shard=wmap[src_key], src_key=src_key, zero_out=zero),
        )

    # Layer 2: width-expanded gate_proj (padding) and duplicated down_proj.
    for suf in src_index.layer_suffixes():
        src_key = f"model.layers.1.{suf}"
        if src_key not in wmap:
            continue
        new_key = f"model.layers.2.{suf}"
        if suf == "mlp.gate_proj.weight":
            plan.add(
                new_key,
                TensorRecipe(
                    src_shard=wmap[src_key],
                    src_key=src_key,
                    pad_rows=8,
                ),
            )
        elif suf == "mlp.down_proj.weight":
            plan.add(
                new_key,
                TensorRecipe(
                    src_shard=wmap[src_key],
                    src_key=src_key,
                    dup_rows=True,
                    dup_rows_noise_scale=1e-6,
                ),
            )
        else:
            plan.add(
                new_key,
                TensorRecipe(src_shard=wmap[src_key], src_key=src_key),
            )

    # Non-layer tensors pass through.
    for key, shard in wmap.items():
        if not key.startswith("model.layers."):
            plan.passthrough(key, shard)

    # A brand-new tensor (e.g. MoE router).
    plan.add(
        "model.layers.2.mlp.gate.weight",
        TensorRecipe(
            src_shard="model.safetensors",
            src_key="model.layers.0.input_layernorm.weight",
            create_shape=(8, 32),
            create_dtype="F32",
        ),
    )

    plan.config_patches = {"num_experts": 8}
    return plan


def _load_tensors(path: Path) -> dict[str, torch.Tensor]:
    """Load all tensors from a single safetensors file."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118


def _run_writer(
    src_dir: Path,
    dst_dir: Path,
    *,
    workers: int = 1,
    resume: bool = False,
    validate_output: bool = False,
    target_shard_bytes: int = 1_000_000,
) -> dict[str, torch.Tensor]:
    """Run ShardWriter and return the full output tensor map."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_index = ShardIndex.load(src_dir)
    plan = _mixed_plan(src_index)
    writer = ShardWriter(
        src_index=src_index,
        dst_dir=dst_dir,
        plan=plan,
        target_shard_bytes=target_shard_bytes,
        verbose=False,
        workers=workers,
        resume=resume,
    )
    if validate_output:
        writer.write_and_validate(src_dir)
    else:
        writer.write(src_dir)
    src_index.copy_non_weight_files(dst_dir)

    out_index = ShardIndex.load(dst_dir)
    tensors: dict[str, torch.Tensor] = {}
    for sf in out_index.shard_files:
        tensors.update(_load_tensors(dst_dir / sf))
    return tensors


class TestShardWriterSerial:
    def test_writes_all_tensors(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"
        tensors = _run_writer(src_dir, dst_dir, workers=1)

        src_index = ShardIndex.load(src_dir)
        plan = _mixed_plan(src_index)
        assert set(tensors) == set(plan.recipes)

    def test_identity_block_zeroed(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"
        tensors = _run_writer(src_dir, dst_dir, workers=1)

        assert torch.all(tensors["model.layers.1.self_attn.o_proj.weight"] == 0)
        assert torch.all(tensors["model.layers.1.mlp.down_proj.weight"] == 0)

    def test_padded_shape(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"
        tensors = _run_writer(src_dir, dst_dir, workers=1)

        assert tensors["model.layers.2.mlp.gate_proj.weight"].shape == (72, 32)

    def test_dup_rows_shape(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"
        tensors = _run_writer(src_dir, dst_dir, workers=1)

        assert tensors["model.layers.2.mlp.down_proj.weight"].shape == (64, 64)

    def test_create_shape_tensor(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"
        tensors = _run_writer(src_dir, dst_dir, workers=1)

        t = tensors["model.layers.2.mlp.gate.weight"]
        assert t.shape == (8, 32)
        assert t.dtype == torch.float32
        assert torch.all(t == 0)

    def test_multi_shard_output(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"
        _run_writer(src_dir, dst_dir, workers=1, target_shard_bytes=10_000)

        out_index = ShardIndex.load(dst_dir)
        assert len(out_index.shard_files) > 1
        assert (dst_dir / "model.safetensors.index.json").exists()

    def test_config_updated(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"
        _run_writer(src_dir, dst_dir, workers=1)

        cfg = json.loads((dst_dir / "config.json").read_text())
        assert cfg["num_hidden_layers"] == 3
        assert cfg["num_experts"] == 8

    def test_validate_output_passes(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"
        # Should not raise.
        _run_writer(src_dir, dst_dir, workers=1, validate_output=True)


class TestShardWriterParallel:
    def test_parallel_matches_serial(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        serial_dir = tmp_path / "serial"
        parallel_dir = tmp_path / "parallel"

        serial = _run_writer(src_dir, serial_dir, workers=1)
        parallel = _run_writer(src_dir, parallel_dir, workers=2)

        assert set(serial) == set(parallel)
        src_index = ShardIndex.load(src_dir)
        plan = _mixed_plan(src_index)
        for key in serial:
            recipe = plan.recipes[key]
            if recipe.dup_rows or recipe.add_noise_std > 0:
                # Noise is seeded independently; verify shape and that the
                # original half is preserved for dup_rows.
                assert serial[key].shape == parallel[key].shape
                if recipe.dup_rows:
                    half = serial[key].shape[0] // 2
                    assert torch.allclose(serial[key][:half], parallel[key][:half])
            else:
                assert torch.equal(serial[key], parallel[key]), key

    def test_parallel_multi_shard(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"
        _run_writer(src_dir, dst_dir, workers=2, target_shard_bytes=10_000)

        out_index = ShardIndex.load(dst_dir)
        assert len(out_index.shard_files) >= 1


class TestShardWriterResume:
    def test_serial_resume_skips_existing(self, tmp_path, caplog):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"

        # First write.
        first = _run_writer(src_dir, dst_dir, workers=1)

        # Corrupt one tensor in the destination to make sure resume would
        # detect the mismatch if validation were broken.
        plan = _mixed_plan(ShardIndex.load(src_dir))
        for key in first:
            if plan.recipes[key].create_shape:
                continue
            if plan.recipes[key].pad_rows or plan.recipes[key].dup_rows:
                bad_key = key
                break
        else:
            pytest.fail("no transform key found")

        out_index = ShardIndex.load(dst_dir)
        shard_path = dst_dir / out_index.weight_map[bad_key]
        corrupted = _load_tensors(shard_path)
        corrupted[bad_key] = torch.zeros(1, 1)
        safetensors_save(corrupted, str(shard_path))

        # Resume should rewrite the corrupted shard.
        resumed = _run_writer(src_dir, dst_dir, workers=1, resume=True)
        assert resumed[bad_key].shape == first[bad_key].shape

    def test_parallel_resume_skips_existing(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"

        first = _run_writer(src_dir, dst_dir, workers=2)
        resumed = _run_writer(src_dir, dst_dir, workers=2, resume=True)

        assert set(first) == set(resumed)
        for key in first:
            assert torch.equal(first[key], resumed[key]), key

    def test_resume_detects_wrong_create_shape(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"

        first = _run_writer(src_dir, dst_dir, workers=1)
        create_key = "model.layers.2.mlp.gate.weight"

        out_index = ShardIndex.load(dst_dir)
        shard_path = dst_dir / out_index.weight_map[create_key]
        corrupted = _load_tensors(shard_path)
        corrupted[create_key] = torch.zeros(4, 4)
        safetensors_save(corrupted, str(shard_path))

        resumed = _run_writer(src_dir, dst_dir, workers=1, resume=True)
        assert resumed[create_key].shape == first[create_key].shape

    def test_resume_detects_wrong_dtype(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"

        first = _run_writer(src_dir, dst_dir, workers=1)
        passthrough_key = "model.embed_tokens.weight"

        out_index = ShardIndex.load(dst_dir)
        shard_path = dst_dir / out_index.weight_map[passthrough_key]
        corrupted = _load_tensors(shard_path)
        corrupted[passthrough_key] = corrupted[passthrough_key].to(torch.float16)
        safetensors_save(corrupted, str(shard_path))

        resumed = _run_writer(src_dir, dst_dir, workers=1, resume=True)
        assert resumed[passthrough_key].dtype == first[passthrough_key].dtype

    def test_partial_resume_serial(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"

        # Write with a tiny shard size so multiple shards are produced.
        _run_writer(src_dir, dst_dir, workers=1, target_shard_bytes=10_000)
        out_index = ShardIndex.load(dst_dir)
        shards = out_index.shard_files
        assert len(shards) > 1

        # Delete the last shard.
        (dst_dir / shards[-1]).unlink()

        # Resume should rewrite only the missing shard (must use the same
        # target shard size so group assignment stays consistent).
        resumed = _run_writer(
            src_dir, dst_dir, workers=1, resume=True, target_shard_bytes=10_000
        )
        assert len(resumed) == len(out_index.all_keys)

    def test_partial_resume_parallel(self, tmp_path):
        src_dir = _make_tiny_source_model(tmp_path / "src")
        dst_dir = tmp_path / "dst"

        _run_writer(src_dir, dst_dir, workers=2, target_shard_bytes=10_000)
        out_index = ShardIndex.load(dst_dir)
        shards = out_index.shard_files
        assert len(shards) > 1

        (dst_dir / shards[-1]).unlink()
        resumed = _run_writer(
            src_dir, dst_dir, workers=2, resume=True, target_shard_bytes=10_000
        )
        assert len(resumed) == len(out_index.all_keys)
