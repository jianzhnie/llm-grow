"""Tests for training: freeze, growth_scheduler, distillation, loss."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from llm_grow.training.distillation import (
    DistillationLoss,
    DistillConfig,
    run_teacher_inference,
)
from llm_grow.training.freeze import (
    freeze_layers_by_index,
    freeze_original_layers,
    mark_new_params,
    report_trainable,
    snapshot_param_ids,
    unfreeze_all,
)
from llm_grow.training.growth_scheduler import GrowthScheduleConfig, GrowthScheduler
from llm_grow.training.load_balance import combined_moe_loss, load_balance_loss, z_loss
from tests.conftest import FakeModel

# ---------------------------------------------------------------------------
# freeze.py tests
# ---------------------------------------------------------------------------


class TestFreezeOriginalLayers:
    def _make_model(self):
        return FakeModel(num_layers=4, d=32)

    def test_freezes_unmarked_params(self):
        model = self._make_model()
        # Mark one layer as new growth
        for p in model.layers[0].parameters():
            p._is_new_growth = True
        frozen_count = freeze_original_layers(model)
        # All params without _is_new_growth should be frozen
        for p in model.layers[1].parameters():
            assert not p.requires_grad
        # New growth params should remain trainable
        for p in model.layers[0].parameters():
            assert p.requires_grad
        assert frozen_count > 0

    def test_returns_correct_numel(self):
        model = self._make_model()
        total = sum(p.numel() for p in model.parameters())
        frozen_count = freeze_original_layers(model)
        assert frozen_count == total


class TestMarkNewParams:
    def test_marks_new_parameters(self):
        model = FakeModel(num_layers=4, d=32)
        original_ids = snapshot_param_ids(model)
        # Simulate expansion by adding a new layer
        model.layers.append(nn.Linear(32, 32))
        count = mark_new_params(model, original_ids)
        new_layer = model.layers[-1]
        assert getattr(new_layer.weight, "_is_new_growth", False) is True
        assert getattr(new_layer.bias, "_is_new_growth", False) is True
        expected = new_layer.weight.numel() + new_layer.bias.numel()
        assert count == expected


class TestSnapshotParamIds:
    def test_returns_set_of_ids(self):
        model = FakeModel(num_layers=2, d=32)
        ids = snapshot_param_ids(model)
        assert isinstance(ids, set)
        assert len(ids) == len(list(model.parameters()))


class TestUnfreezeAll:
    def test_unfreezes_all_params(self):
        model = FakeModel(num_layers=4, d=32)
        # Freeze everything first
        for p in model.parameters():
            p.requires_grad_(False)
        count = unfreeze_all(model)
        for p in model.parameters():
            assert p.requires_grad
        assert count == sum(p.numel() for p in model.parameters())


class TestReportTrainable:
    def test_all_trainable(self):
        model = FakeModel(num_layers=2, d=32)
        report = report_trainable(model)
        total = sum(p.numel() for p in model.parameters())
        assert report["trainable"] == total
        assert report["frozen"] == 0
        assert report["total"] == total

    def test_mixed_trainable(self):
        model = FakeModel(num_layers=2, d=32)
        # Freeze embed layer
        for p in model.embed.parameters():
            p.requires_grad_(False)
        report = report_trainable(model)
        assert report["frozen"] == sum(p.numel() for p in model.embed.parameters())
        assert report["trainable"] + report["frozen"] == report["total"]


class TestFreezeLayersByIndex:
    def test_freezes_specific_layers(self):
        model = FakeModel(num_layers=4, d=32)
        freeze_layers_by_index(model, [0, 2], layer_attr="layers")
        for p in model.layers[0].parameters():
            assert not p.requires_grad
        for p in model.layers[1].parameters():
            assert p.requires_grad
        for p in model.layers[2].parameters():
            assert not p.requires_grad
        for p in model.layers[3].parameters():
            assert p.requires_grad

    def test_raises_on_bad_attr(self):
        model = FakeModel(num_layers=4, d=32)
        with pytest.raises(AttributeError):
            freeze_layers_by_index(model, [0], layer_attr="nonexistent.attr")


# ---------------------------------------------------------------------------
# growth_scheduler.py tests
# ---------------------------------------------------------------------------


class TestGrowthSchedulerGetUnlockRatio:
    def test_linear_during_warmup(self):
        cfg = GrowthScheduleConfig(total_steps=100, warmup_ratio=0.3, strategy="linear")
        scheduler = GrowthScheduler(cfg)
        assert scheduler.get_unlock_ratio(0) == 0.0
        assert scheduler.get_unlock_ratio(10) == 0.0
        assert scheduler.get_unlock_ratio(29) == 0.0

    def test_linear_after_warmup(self):
        cfg = GrowthScheduleConfig(total_steps=100, warmup_ratio=0.0, strategy="linear")
        scheduler = GrowthScheduler(cfg)
        assert scheduler.get_unlock_ratio(0) == 0.0
        assert abs(scheduler.get_unlock_ratio(50) - 0.5) < 1e-6
        assert abs(scheduler.get_unlock_ratio(100) - 1.0) < 1e-6

    def test_cosine_strategy(self):
        cfg = GrowthScheduleConfig(total_steps=100, warmup_ratio=0.0, strategy="cosine")
        scheduler = GrowthScheduler(cfg)
        assert scheduler.get_unlock_ratio(0) == 0.0
        mid = scheduler.get_unlock_ratio(50)
        expected = 0.5 * (1 - math.cos(math.pi * 0.5))
        assert abs(mid - expected) < 1e-6
        assert abs(scheduler.get_unlock_ratio(100) - 1.0) < 1e-6

    def test_step_strategy(self):
        cfg = GrowthScheduleConfig(total_steps=100, warmup_ratio=0.0, strategy="step")
        scheduler = GrowthScheduler(cfg)
        assert scheduler.get_unlock_ratio(0) == 0.0
        assert scheduler.get_unlock_ratio(24) == 0.0
        assert scheduler.get_unlock_ratio(25) == 0.25
        assert scheduler.get_unlock_ratio(50) == 0.50
        assert scheduler.get_unlock_ratio(75) == 0.75
        assert scheduler.get_unlock_ratio(100) == 1.0

    def test_unknown_strategy_raises(self):
        cfg = GrowthScheduleConfig(total_steps=100, strategy="unknown")
        scheduler = GrowthScheduler(cfg)
        with pytest.raises(ValueError):
            scheduler.get_unlock_ratio(50)


class TestGrowthSchedulerApplyMasks:
    def test_apply_masks_zero_ratio(self):
        model = FakeModel(num_layers=2, d=32)
        for p in model.layers[0].parameters():
            p._is_new_growth = True
        cfg = GrowthScheduleConfig(total_steps=100)
        scheduler = GrowthScheduler(cfg)
        scheduler.apply_masks(model, 0.0)
        for p in model.layers[0].parameters():
            assert not p.requires_grad

    def test_apply_masks_positive_ratio(self):
        model = FakeModel(num_layers=2, d=32)
        for p in model.layers[0].parameters():
            p._is_new_growth = True
            p.requires_grad_(False)
        cfg = GrowthScheduleConfig(total_steps=100)
        scheduler = GrowthScheduler(cfg)
        scheduler.apply_masks(model, 0.5)
        for p in model.layers[0].parameters():
            assert p.requires_grad


class TestGrowthSchedulerRegisterNewParams:
    def test_register_with_original_ids(self):
        model = FakeModel(num_layers=2, d=32)
        original_ids = GrowthScheduler.snapshot_param_ids(model)
        model.layers.append(nn.Linear(32, 32))
        cfg = GrowthScheduleConfig(total_steps=100)
        scheduler = GrowthScheduler(cfg)
        count = scheduler.register_new_params(model, original_ids)
        new_layer = model.layers[-1]
        expected = new_layer.weight.numel() + new_layer.bias.numel()
        assert count == expected

    def test_register_without_original_ids(self):
        model = FakeModel(num_layers=2, d=32)
        for p in model.layers[0].parameters():
            p._is_new_growth = True
        cfg = GrowthScheduleConfig(total_steps=100)
        scheduler = GrowthScheduler(cfg)
        count = scheduler.register_new_params(model, None)
        expected = sum(p.numel() for p in model.layers[0].parameters())
        assert count == expected


# ---------------------------------------------------------------------------
# distillation.py tests
# ---------------------------------------------------------------------------


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
        import torch.nn.functional as F

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


# ---------------------------------------------------------------------------
# load_balance.py tests
# ---------------------------------------------------------------------------


class TestLoadBalanceLoss:
    def test_uniform_routing(self):
        num_experts = 4
        # Uniform logits => uniform routing => loss should be small
        logits = torch.zeros(16, num_experts)
        loss = load_balance_loss(logits, num_experts, top_k=2, coeff=1e-2)
        assert loss.item() >= 0

    def test_skewed_routing_higher_loss(self):
        num_experts = 4
        uniform_logits = torch.zeros(16, num_experts)
        skewed_logits = torch.zeros(16, num_experts)
        skewed_logits[:, 0] = 100.0  # Everything routes to expert 0
        loss_uniform = load_balance_loss(
            uniform_logits, num_experts, top_k=1, coeff=1e-2
        )
        loss_skewed = load_balance_loss(skewed_logits, num_experts, top_k=1, coeff=1e-2)
        assert loss_skewed.item() > loss_uniform.item()

    def test_scalar_output(self):
        logits = torch.randn(8, 4)
        loss = load_balance_loss(logits, num_experts=4)
        assert loss.shape == ()


class TestZLoss:
    def test_zero_logits(self):
        logits = torch.zeros(8, 4)
        loss = z_loss(logits, coeff=1e-3)
        # logsumexp(zeros) = log(4), z_loss = coeff * log(4)^2
        expected = 1e-3 * (math.log(4) ** 2)
        assert abs(loss.item() - expected) < 1e-5

    def test_positive_loss(self):
        logits = torch.randn(16, 8)
        loss = z_loss(logits, coeff=1e-3)
        assert loss.item() > 0

    def test_scalar_output(self):
        logits = torch.randn(8, 4)
        loss = z_loss(logits)
        assert loss.shape == ()


class TestCombinedMoeLoss:
    def test_combined_includes_all(self):
        lm_loss = torch.tensor(2.0)
        router_logits_list = [torch.randn(8, 4), torch.randn(8, 4)]
        total = combined_moe_loss(
            lm_loss,
            router_logits_list,
            num_experts=4,
            top_k=2,
            balance_coeff=1e-2,
            z_coeff=1e-3,
        )
        assert total.item() > lm_loss.item()

    def test_empty_router_list(self):
        lm_loss = torch.tensor(3.0)
        total = combined_moe_loss(lm_loss, [], num_experts=4)
        assert total.item() == lm_loss.item()

    def test_scalar_output(self):
        lm_loss = torch.tensor(1.0)
        total = combined_moe_loss(lm_loss, [torch.randn(4, 2)], num_experts=2)
        assert total.shape == ()
