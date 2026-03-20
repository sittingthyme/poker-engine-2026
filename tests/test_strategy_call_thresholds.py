"""Smoke tests for strategy calling / sizing helpers (post pot-odds refactor)."""

import unittest
from unittest import mock

from submission import strategy


class TestStrategyCallSizing(unittest.TestCase):
    def test_semi_bluff_equity_band(self):
        self.assertEqual(strategy.SEMI_BLUFF_EQUITY_MIN, 0.30)
        self.assertEqual(strategy.SEMI_BLUFF_EQUITY_MAX, 0.45)

    def test_raise_frac_polarized_river_overbet(self):
        with mock.patch.object(strategy.random, "uniform", return_value=1.5):
            f = strategy._raise_frac_value(
                flush_danger=0,
                pair_danger=0,
                street=3,
                is_polarized=True,
            )
        self.assertEqual(f, 1.5)

    def test_raise_frac_non_polar_uses_standard_range(self):
        with mock.patch.object(strategy.random, "uniform", return_value=0.72):
            f = strategy._raise_frac_value(
                flush_danger=0,
                pair_danger=0,
                street=3,
                is_polarized=False,
            )
        self.assertEqual(f, 0.72)


if __name__ == "__main__":
    unittest.main()
