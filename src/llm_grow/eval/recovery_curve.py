"""Recovery curve tracker: log benchmark scores during continued pretraining."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


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
        logger.info("Baseline set: %s", scores)

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
            logger.info(
                "step=%d tokens=%s  %s  recovery=%s",
                step,
                f"{tokens_seen:,}",
                scores,
                recovery,
            )

    def summary(self) -> None:
        if not self.points:
            logger.info("No data logged yet.")
            return
        last = self.points[-1]
        logger.info(
            "%d checkpoints logged. Latest @ step=%d, tokens=%s",
            len(self.points),
            last.step,
            f"{last.tokens_seen:,}",
        )
        if self.baseline:
            for k, v in last.scores.items():
                base = self.baseline.get(k, None)
                if base:
                    logger.info(
                        "  %s: %.4f  (baseline=%.4f, recovery=%.1f%%)",
                        k,
                        v,
                        base,
                        v / base * 100,
                    )
