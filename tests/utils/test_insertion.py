"""Tests for llm_grow.utils.insertion."""

from __future__ import annotations

import pytest

from llm_grow.utils.insertion import insert_positions


class TestInsertPositions:
    def test_uniform(self):
        positions = insert_positions(8, 4, "uniform")
        assert len(positions) == 4
        assert all(0 <= p < 8 for p in positions)
        assert positions == sorted(positions)

    def test_front(self):
        positions = insert_positions(8, 3, "front")
        assert positions == [0, 1, 2]

    def test_rear(self):
        positions = insert_positions(8, 3, "rear")
        assert positions == [5, 6, 7]

    def test_zero_new(self):
        assert insert_positions(8, 0, "uniform") == []

    def test_num_new_exceeds_orig_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            insert_positions(4, 5, "uniform")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown insert strategy"):
            insert_positions(8, 2, "middle")

    def test_single_insert_uniform(self):
        positions = insert_positions(8, 1, "uniform")
        assert len(positions) == 1

    def test_front_single(self):
        assert insert_positions(8, 1, "front") == [0]

    def test_rear_single(self):
        assert insert_positions(8, 1, "rear") == [7]
