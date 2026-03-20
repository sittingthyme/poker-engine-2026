"""Abstract state key matches runtime StrategyTable format."""

import unittest

from submission.abstract_state import (
    abstract_state_key,
    continue_cost_band,
    equity_band,
    pot_band,
)


class TestAbstractState(unittest.TestCase):
    def test_pot_band_edges(self):
        self.assertEqual(pot_band(0), 0)
        self.assertEqual(pot_band(9), 0)
        self.assertEqual(pot_band(10), 1)
        self.assertEqual(pot_band(49), 1)
        self.assertEqual(pot_band(50), 2)

    def test_equity_band_edges(self):
        self.assertEqual(equity_band(0.0), 0)
        self.assertEqual(equity_band(0.199), 0)
        self.assertEqual(equity_band(0.20), 1)
        self.assertEqual(equity_band(0.78), 4)
        self.assertEqual(equity_band(1.0), 4)

    def test_key_format(self):
        k = abstract_state_key(
            street=2,
            in_position=True,
            pot=24,
            continue_cost=6,
            equity=0.55,
        )
        self.assertEqual(k, "s2_p1_pb1_cb2_eb2")


if __name__ == "__main__":
    unittest.main()
