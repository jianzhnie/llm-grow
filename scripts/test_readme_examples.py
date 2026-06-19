#!/usr/bin/env python
"""Test every code example in README.md against the real codebase.

Uses Qwen3-0.6B (local weights) for examples that need a model.
Skips examples that require unavailable models (Qwen3-8B).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile

SRC = "/Users/robin/hfhub/models/Qwen/Qwen3-0.6B"
SRC_MOE = "/Users/robin/hfhub/models/Qwen/Qwen3-30B-A3B"  # index only
SRC_LONGCAT = "/Users/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat"

results: dict[str, bool] = {}


def run(name: str, code: str, expect_ok: bool = True) -> bool:
    try:
        exec(code, {"__name__": "__test__"})
        ok = True
    except Exception as e:
        ok = False
        if expect_ok:
            print(f"  [FAIL] {name}: {e}")
    if ok == expect_ok:
        results[name] = True
        print(f"  [PASS] {name}")
    else:
        results[name] = False
        print(f"  [FAIL] {name}")
    return ok == expect_ok


def run_cmd(name: str, cmd: str, expect_ok: bool = True) -> bool:
    r = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd="/Users/robin/work_dir/llm-grow",
        timeout=60,
    )
    ok = (r.returncode == 0) == expect_ok
    results[name] = ok
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}] {name}")
    if not ok:
        print(f"    stderr: {r.stderr[:200]}")
    return ok


def main():
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Section: Imports ===")
    # ══════════════════════════════════════════════════════════════════════════

    # README line 160-161
    run(
        "import/llama_pro",
        """
from llm_grow.expanders.depth.identity_graft import (
    IdentityGraftConfig, IdentityGraftExpander,
)
""",
    )

    # README line 285-286
    run(
        "import/auto_expand",
        """
from llm_grow.safetensor.auto import auto_expand
from llm_grow.safetensor.detect import detect_model
""",
    )

    # README line 307
    run(
        "import/make_qwen3moe",
        """
from llm_grow.safetensor.moe_generic import make_qwen3moe_upcycling
""",
    )

    # README line 373-374
    run(
        "import/msg",
        """
from llm_grow.expanders.width.multi_axis_grow import (
    MultiAxisGrowConfig, MultiAxisGrowExpander,
)
""",
    )

    # README line 389-390
    run(
        "import/moe_upcycling",
        """
from llm_grow.expanders.sparse.dense_to_moe import (
    DenseToMoEConfig, DenseToMoEExpander,
)
from llm_grow.training.load_balance import combined_moe_loss
""",
    )

    # README line 404-406
    run(
        "import/expert_upcycling",
        """
from llm_grow.expanders.sparse.expert_clone import (
    ExpertCloneConfig, ExpertCloneExpander, ExpertSelectionStrategy
)
""",
    )

    # README line 444
    run(
        "import/freeze",
        """
from llm_grow.training.freeze import (
    freeze_original_layers, unfreeze_all, report_trainable,
)
""",
    )

    # README line 456
    run(
        "import/distillation",
        """
from llm_grow.training.distillation import (
    DistillConfig, DistillationLoss, run_teacher_inference,
)
""",
    )

    # README line 466
    run(
        "import/load_balance",
        """
from llm_grow.training.load_balance import combined_moe_loss
""",
    )

    # README line 482
    run(
        "import/fp_verifier",
        """
from llm_grow.eval.fp_verifier import verify_fp
""",
    )

    # README line 490
    run(
        "import/recovery_curve",
        """
from llm_grow.eval.recovery_curve import RecoveryCurveTracker
""",
    )

    # README line 122-123
    run(
        "import/expansion_plan",
        """
from llm_grow.safetensor.base import ExpansionPlan
""",
    )

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Section: detect_model ===")
    # ══════════════════════════════════════════════════════════════════════════

    # README line 184-188
    run(
        "detect_model/qwen3",
        f"""
from llm_grow.safetensor.detect import detect_model
profile = detect_model("{SRC}")
assert profile.family == "dense"
print(profile.summary())
""",
    )

    run(
        "detect_model/qwen3_moe",
        f"""
from llm_grow.safetensor.detect import detect_model
profile = detect_model("{SRC_MOE}")
assert profile.family == "standard_moe"
assert profile.is_moe
assert profile.experts_per_moe_layer == 128
""",
    )

    run(
        "detect_model/longcat",
        f"""
from llm_grow.safetensor.detect import detect_model
profile = detect_model("{SRC_LONGCAT}")
assert profile.family == "longcat"
""",
    )

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Section: Safetensor CLI ===")
    # ══════════════════════════════════════════════════════════════════════════

    # README line 137-141: auto depth
    with tempfile.TemporaryDirectory() as d:
        run_cmd(
            "cli/auto_depth",
            f"python scripts/safetensor_expand.py auto "
            f"--src {SRC} --dst {d}/out --method depth --num-new-layers 4",
        )

    # README line 144-146: auto dry-run
    run_cmd(
        "cli/auto_dryrun",
        f"python scripts/safetensor_expand.py auto "
        f"--src {SRC} --dst /tmp/x --method depth --dry-run",
    )

    # README line 149-152: auto expert (MoE, dry-run since no weights)
    run_cmd(
        "cli/auto_expert_dryrun",
        f"python scripts/safetensor_expand.py auto "
        f"--src {SRC_MOE} --dst /tmp/x --method expert --expand-factor 2 --dry-run",
    )

    # README line 256-258: auto width
    with tempfile.TemporaryDirectory() as d:
        run_cmd(
            "cli/auto_width",
            f"python scripts/safetensor_expand.py auto "
            f"--src {SRC} --dst {d}/out --method width --ffn-size-expansion 512",
        )

    # README line 260: Dense + expert should error
    run_cmd(
        "cli/dense_expert_error",
        f"python scripts/safetensor_expand.py auto "
        f"--src {SRC} --dst /tmp/x --method expert --expand-factor 2",
        expect_ok=False,
    )

    # README line 264-265: explicit llama_pro
    with tempfile.TemporaryDirectory() as d:
        run_cmd(
            "cli/llama_pro",
            f"python scripts/safetensor_expand.py llama_pro "
            f"--src {SRC} --dst {d}/out --num-new-layers 7",
        )

    # README line 267-268: explicit solar_dus
    with tempfile.TemporaryDirectory() as d:
        run_cmd(
            "cli/solar_dus",
            f"python scripts/safetensor_expand.py solar_dus "
            f"--src {SRC} --dst {d}/out --num-overlap 8",
        )

    # README line 270-272: explicit msg
    with tempfile.TemporaryDirectory() as d:
        run_cmd(
            "cli/msg",
            f"python scripts/safetensor_expand.py msg "
            f"--src {SRC} --dst {d}/out --num-new-layers 4 --ffn-size-expansion 1024",
        )

    # README line 274-275: moe_expert (dry-run)
    run_cmd(
        "cli/moe_expert_dryrun",
        f"python scripts/safetensor_expand.py moe_expert "
        f"--src {SRC_MOE} --dst /tmp/x --expand-factor 2 --dry-run",
    )

    # README line 278-279: dry-run
    run_cmd(
        "cli/dryrun_generic",
        f"python scripts/safetensor_expand.py auto "
        f"--src {SRC} --dst /tmp/x --method depth --dry-run",
    )

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Section: verify_safetensor CLI ===")
    # ══════════════════════════════════════════════════════════════════════════

    # README line 332-335: verify with --fp
    with tempfile.TemporaryDirectory() as d:
        # First expand, then verify
        subprocess.run(
            f"python scripts/safetensor_expand.py llama_pro "
            f"--src {SRC} --dst {d}/out --num-new-layers 4 --quiet",
            shell=True,
            capture_output=True,
            cwd="/Users/robin/work_dir/llm-grow",
            timeout=60,
        )
        run_cmd(
            "cli/verify_fp",
            f"python scripts/verify_safetensor.py --src {SRC} --dst {d}/out --fp",
        )

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Section: Python API (safetensor) ===")
    # ══════════════════════════════════════════════════════════════════════════

    # README line 296-304: auto_expand
    with tempfile.TemporaryDirectory() as d:
        run(
            "api/auto_expand",
            f"""
from llm_grow.safetensor.auto import auto_expand
auto_expand(
    src_dir="{SRC}",
    dst_dir="{d}/expanded",
    method="depth",
    num_new_layers=4,
    insert_strategy="uniform",
    dry_run=True,
)
""",
        )

    # README line 289-293: profile attributes
    run(
        "api/profile_attrs",
        f"""
from llm_grow.safetensor.detect import detect_model
profile = detect_model("{SRC}")
assert profile.family == "dense"
assert profile.is_moe is False
assert profile.experts_per_moe_layer == 0
assert profile.has_fp8 is False
""",
    )

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Section: In-memory API ===")
    # ══════════════════════════════════════════════════════════════════════════

    # README line 355-363: LLaMA-Pro config
    run(
        "inmem/llama_pro_config",
        """
from llm_grow.expanders.depth.identity_graft import (
    IdentityGraftConfig, IdentityGraftExpander,
)
config = IdentityGraftConfig(
    num_new_layers=9,
    insert_strategy="uniform",
    freeze_original=True,
)
assert config.num_new_layers == 9
""",
    )

    # README line 375-381: MSG config
    run(
        "inmem/msg_config",
        """
from llm_grow.expanders.width.multi_axis_grow import (
    MultiAxisGrowConfig, MultiAxisGrowExpander,
)
config = MultiAxisGrowConfig(
    num_new_layers=10,
    hidden_size_expansion=512,
    intermediate_size_expansion=3072,
    freeze_original=True,
)
assert config.num_new_layers == 10
""",
    )

    # README line 393-394: MoE Upcycling config
    run(
        "inmem/moe_upcycling_config",
        """
from llm_grow.expanders.sparse.dense_to_moe import DenseToMoEConfig
cfg = DenseToMoEConfig(num_experts=8, top_k=2)
assert cfg.num_experts == 8
""",
    )

    # README line 408-411: Expert Upcycling config
    run(
        "inmem/expert_upcycling_config",
        """
from llm_grow.expanders.sparse.expert_clone import (
    ExpertCloneConfig, ExpertSelectionStrategy
)
cfg = ExpertCloneConfig(
    expand_factor=2,
    selection_strategy=ExpertSelectionStrategy.UTILITY,
)
assert cfg.expand_factor == 2
""",
    )

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Section: Training tools ===")
    # ══════════════════════════════════════════════════════════════════════════

    # README line 443-450: freeze/unfreeze
    run(
        "training/freeze_api",
        """
import torch.nn as nn
from llm_grow.training.freeze import (
    freeze_original_layers, unfreeze_all, report_trainable,
)
model = nn.Linear(10, 10)
freeze_original_layers(model)
unfreeze_all(model)
info = report_trainable(model)
assert info["total"] > 0
""",
    )

    # README line 456-460: distillation
    run(
        "training/distillation_api",
        """
from llm_grow.training.distillation import DistillConfig, DistillationLoss
criterion = DistillationLoss(DistillConfig(temperature=2.0, alpha=0.5))
assert criterion.config.temperature == 2.0
""",
    )

    # README line 466-472: load_balance
    run(
        "training/load_balance_api",
        """
import torch
from llm_grow.training.load_balance import combined_moe_loss
lm_loss = torch.tensor(1.0)
router_logits = [torch.randn(16, 8)]
loss = combined_moe_loss(lm_loss, router_logits, num_experts=8, top_k=2)
assert loss.item() > 0
""",
    )

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Section: Evaluation ===")
    # ══════════════════════════════════════════════════════════════════════════

    # README line 490-495: RecoveryCurveTracker
    run(
        "eval/recovery_curve",
        """
import tempfile, os
from llm_grow.eval.recovery_curve import RecoveryCurveTracker
with tempfile.TemporaryDirectory() as d:
    tracker = RecoveryCurveTracker(os.path.join(d, "recovery.jsonl"))
    tracker.set_baseline({"mmlu": 0.72, "gsm8k": 0.65})
    tracker.log(step=1000, tokens_seen=2e9, scores={"mmlu": 0.70, "gsm8k": 0.60})
    tracker.summary()
""",
    )

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Section: CLI tool (llm-grow) ===")
    # ══════════════════════════════════════════════════════════════════════════

    # README line 114: llm-grow info
    run_cmd("cli/llm_grow_info", f"llm-grow info --src {SRC}")

    # README line 107-108: llm-grow expand
    with tempfile.TemporaryDirectory() as d:
        run_cmd(
            "cli/llm_grow_expand",
            f"llm-grow expand --src {SRC} --dst {d}/out "
            f"--method depth --num-new-layers 4",
        )

    # README line 111: llm-grow verify (expand first, then verify)
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(
            f"llm-grow expand --src {SRC} --dst {d}/out "
            f"--method depth --num-new-layers 4",
            shell=True,
            capture_output=True,
            cwd="/Users/robin/work_dir/llm-grow",
            timeout=60,
        )
        run_cmd("cli/llm_grow_verify", f"llm-grow verify --src {SRC} --dst {d}/out")

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== Section: Test scripts (README line 604-608) ===")
    # ══════════════════════════════════════════════════════════════════════════

    run_cmd("test/pytest", "python -m pytest tests/ -q")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  README Examples Summary")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    failed_names = [k for k, v in results.items() if not v]
    for name in failed_names:
        print(f"  [FAIL] {name}")
    print(f"\n  {passed} passed, {len(failed_names)} failed, {len(results)} total")
    sys.exit(0 if not failed_names else 1)


if __name__ == "__main__":
    main()
