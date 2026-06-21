"""Tests for llm_grow.eval.recovery_curve."""

from __future__ import annotations

from llm_grow.eval.recovery_curve import RecoveryCurveTracker


class TestRecoveryCurveTracker:
    def test_set_baseline(self):
        tracker = RecoveryCurveTracker(save_path="/tmp/test_rc.jsonl")
        tracker.set_baseline({"mmlu": 0.72, "gsm8k": 0.65})
        assert tracker.baseline == {"mmlu": 0.72, "gsm8k": 0.65}

    def test_log_appends_point(self, tmp_path):
        save_path = tmp_path / "curve.jsonl"
        tracker = RecoveryCurveTracker(save_path=save_path)
        tracker.set_baseline({"mmlu": 0.72})
        tracker.log(step=100, tokens_seen=1000, scores={"mmlu": 0.60})
        tracker.log(step=200, tokens_seen=2000, scores={"mmlu": 0.68})

        assert len(tracker.points) == 2
        assert tracker.points[0].step == 100
        assert tracker.points[1].tokens_seen == 2000
        assert save_path.exists()
        lines = save_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_log_without_baseline(self, tmp_path):
        save_path = tmp_path / "curve.jsonl"
        tracker = RecoveryCurveTracker(save_path=save_path)
        tracker.log(step=50, tokens_seen=500, scores={"acc": 0.5})
        assert len(tracker.points) == 1

    def test_summary_empty(self):
        tracker = RecoveryCurveTracker(save_path="/tmp/test_summary.jsonl")
        tracker.summary()

    def test_summary_with_data(self, tmp_path):
        save_path = tmp_path / "curve.jsonl"
        tracker = RecoveryCurveTracker(save_path=save_path)
        tracker.set_baseline({"mmlu": 0.72})
        tracker.log(step=100, tokens_seen=1000, scores={"mmlu": 0.65})
        tracker.summary()
        assert len(tracker.points) == 1
