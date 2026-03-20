"""Tests for SB fold vs unopened pot (submission.strategy)."""

import unittest

from submission.opponent_model import OpponentModel
from submission.strategy import PREFLOP_SB_UNOPENED_FOLD_BELOW_EQUITY, decide_action


def _sb_unopened_obs() -> dict:
    return {
        "valid_actions": [True, True, False, True],
        "street": 0,
        "my_bet": 1,
        "opp_bet": 2,
        "min_raise": 4,
        "max_raise": 100,
        "blind_position": 0,
        "pot_size": 3,
    }


class TestSbUnopenedFold(unittest.TestCase):
    def test_folds_below_threshold(self):
        obs = _sb_unopened_obs()
        a = decide_action(
            PREFLOP_SB_UNOPENED_FOLD_BELOW_EQUITY - 0.05,
            obs,
            OpponentModel(),
            info={"hand_number": 500},
        )
        self.assertEqual(a[0], 0)

    def test_does_not_fold_at_or_above_threshold(self):
        """Above SB trash threshold, complete with a hand that clears the medium band (cheap blind)."""
        obs = _sb_unopened_obs()
        a = decide_action(
            0.58,
            obs,
            OpponentModel(),
            info={"hand_number": 500},
        )
        self.assertNotEqual(a[0], 0)

    def test_bb_not_affected(self):
        obs = _sb_unopened_obs()
        obs["blind_position"] = 1
        obs["my_bet"] = 2
        obs["opp_bet"] = 2
        a = decide_action(
            PREFLOP_SB_UNOPENED_FOLD_BELOW_EQUITY - 0.05,
            obs,
            OpponentModel(),
            info={"hand_number": 500},
        )
        self.assertNotEqual(a[0], 0)


if __name__ == "__main__":
    unittest.main()
