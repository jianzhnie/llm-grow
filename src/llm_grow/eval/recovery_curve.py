"""Recovery curve tracker: log benchmark scores during continued pretraining."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RecoveryPoint:
    step: int
    tokens_seen: int
    scores: dict[str, float]
    timestamp: float = field(default_factory=time.time)


class RecoveryCurveTracker:
    """记录 CPT 过程中各 benchmark 指标随训练步数的变化。

    用法::

        tracker = RecoveryCurveTracker(save_path="recovery.jsonl")
        tracker.set_baseline({"mmlu": 0.72, "gsm8k": 0.65})

        for step, batch in enumerate(dataloader):
            ...
            if step % eval_interval == 0:
                scores = run_eval(model)
                tracker.log(step=step, tokens_seen=step * batch_tokens, scores=scores)
    """

    def __init__(self, save_path: str | Path = "recovery_curve.jsonl"):
        self.save_path = Path(save_path)
        self.baseline: dict[str, float] = {}
        self.points: list[RecoveryPoint] = []

    def set_baseline(self, scores: dict[str, float]) -> None:
        """记录原始模型的 baseline 分数（扩增前）。"""
        self.baseline = scores
        print(f"[Recovery] Baseline set: {scores}")

    def log(
        self,
        step: int,
        tokens_seen: int,
        scores: dict[str, float],
    ) -> None:
        pt = RecoveryPoint(step=step, tokens_seen=tokens_seen, scores=scores)
        self.points.append(pt)

        with self.save_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "step": pt.step,
                        "tokens_seen": pt.tokens_seen,
                        "scores": pt.scores,
                        "timestamp": pt.timestamp,
                    }
                )
                + "\n"
            )

        if self.baseline:
            recovery = {
                k: f"{v / self.baseline[k] * 100:.1f}%"
                for k, v in scores.items()
                if k in self.baseline and self.baseline[k] > 0
            }
            print(
                f"[Recovery] step={step} tokens={tokens_seen:,}  {scores}  "
                f"recovery={recovery}"
            )

    def summary(self) -> None:
        if not self.points:
            print("[Recovery] No data logged yet.")
            return
        last = self.points[-1]
        print(f"\n[Recovery Summary] {len(self.points)} checkpoints logged.")
        print(f"  Latest @ step={last.step}, tokens={last.tokens_seen:,}")
        if self.baseline:
            for k, v in last.scores.items():
                base = self.baseline.get(k, None)
                if base:
                    print(
                        f"  {k}: {v:.4f}  (baseline={base:.4f}, "
                        f"recovery={v / base * 100:.1f}%)"
                    )
