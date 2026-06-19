"""Tests for llm_grow.training.growth_scheduler."""

from __future__ import annotations

import math

import pytest
import torch.nn as nn

from llm_grow.training.growth_scheduler import GrowthScheduleConfig, GrowthScheduler
from tests.conftest import FakeModel


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
