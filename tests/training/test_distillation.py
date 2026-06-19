"""Tests for llm_grow.training.distillation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_grow.training.distillation import (
    DistillationLoss,
    DistillConfig,
    run_teacher_inference,
)
from tests.conftest import FakeModel


class TestDistillationLoss:
    def test_forward_shape(self):
        criterion = DistillationLoss(DistillConfig(alpha=0.5, temperature=2.0))
        student_logits = torch.randn(2, 8, 256)
        teacher_logits = torch.randn(2, 8, 256)
        labels = torch.randint(0, 256, (2, 8))
        loss = criterion(student_logits, teacher_logits, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_alpha_one_is_pure_ce(self):
        criterion = DistillationLoss(DistillConfig(alpha=1.0, temperature=2.0))
        student_logits = torch.randn(2, 4, 10)
        teacher_logits = torch.randn(2, 4, 10)
        labels = torch.randint(0, 10, (2, 4))
        loss = criterion(student_logits, teacher_logits, labels)
        expected_ce = F.cross_entropy(student_logits.view(-1, 10), labels.view(-1))
        assert abs(loss.item() - expected_ce.item()) < 1e-5

    def test_vocab_mismatch_raises(self):
        criterion = DistillationLoss(DistillConfig())
        with pytest.raises(ValueError):
            criterion(
                torch.randn(2, 4, 10),
                torch.randn(2, 4, 8),
                torch.zeros(2, 4, dtype=torch.long),
            )

    def test_all_ignored_labels(self):
        criterion = DistillationLoss(DistillConfig(alpha=0.5, temperature=2.0))
        student_logits = torch.randn(2, 4, 10)
        teacher_logits = torch.randn(2, 4, 10)
        labels = torch.full((2, 4), -100, dtype=torch.long)
        loss = criterion(student_logits, teacher_logits, labels)
        # When all labels are -100, should return ce_loss (which itself ignores all)
        assert loss.shape == ()


class _TeacherModel(nn.Module):
    """FakeModel variant whose forward accepts keyword arguments."""

    def __init__(self, num_layers=2, d=32):
        super().__init__()
        self._inner = FakeModel(num_layers=num_layers, d=d)

    def forward(self, input_ids, attention_mask=None):
        return self._inner(input_ids)


class TestRunTeacherInference:
    def test_returns_correct_shape(self):
        teacher = _TeacherModel(num_layers=2, d=32)
        input_ids = torch.randint(0, 256, (6, 8))
        logits = run_teacher_inference(teacher, input_ids, batch_size=4)
        assert logits.shape == (6, 8, 256)

    def test_teacher_set_to_eval(self):
        teacher = _TeacherModel(num_layers=2, d=32)
        teacher.train()
        run_teacher_inference(teacher, torch.randint(0, 256, (2, 4)), batch_size=2)
        assert not teacher.training

    def test_with_attention_mask(self):
        teacher = _TeacherModel(num_layers=2, d=32)
        input_ids = torch.randint(0, 256, (4, 8))
        logits = run_teacher_inference(
            teacher, input_ids, attention_mask=torch.ones(4, 8), batch_size=2
        )
        assert logits.shape == (4, 8, 256)
