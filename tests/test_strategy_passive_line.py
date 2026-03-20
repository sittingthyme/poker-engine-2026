"""Passive-line value betting (opp checks postflop → relax thresholds)."""

import unittest
from unittest.mock import MagicMock

from submission.opponent_model import OpponentModel
from submission.strategy import (
    PASSIVE_MADE_HAND_MAX_RANK,
    PASSIVE_NUT_WAIVE_MIN_CHECKS,
    decide_action,
)


def _base_obs(
    *,
    street: int = 2,
    continue_cost: int = 0,
    my_bet: int = 10,
    opp_bet: int = 10,
    pot: int = 20,
    community: list[int] | None = None,
    hole: list[int] | None = None,
) -> dict:
    comm = community if community is not None else [0, 1, 2, -1, -1]
    h = hole if hole is not None else [10, 11]
    return {
        "valid_actions": [False, True, True, False],
        "street": street,
        "my_bet": my_bet,
        "opp_bet": opp_bet,
        "min_raise": 4,
        "max_raise": 100,
        "community_cards": comm,
        "my_cards": h + [-1, -1, -1],
        "opp_last_action": "CHECK",
        "acting_agent": 0,
        "blind_position": 0,
        "pot_size": pot,
    }


class TestPassiveLine(unittest.TestCase):
    def test_constants_two_pair_plus_gate(self):
        self.assertEqual(PASSIVE_MADE_HAND_MAX_RANK, 7)
        self.assertEqual(PASSIVE_NUT_WAIVE_MIN_CHECKS, 2)

    def test_passive_waiver_allows_raise_straight_flush_board(self):
        """With enough opp checks, straight (class 5) should raise despite flush nut-block."""
        om = MagicMock(spec=OpponentModel)
        om.hands_seen = 100
        om.fold_rate.return_value = 0.35
        om.aggression.return_value = 1.0
        om.is_tight.return_value = False
        om.is_loose.return_value = False
        om.streets = [MagicMock() for _ in range(4)]
        for s in om.streets:
            s.actions = 20
            s.raises = 2
            s.calls = 10
            s.recent_actions = 0
            s.raise_rate = 0.1
        om.overall = om.streets[0]

        obs = _base_obs()
        info = {
            "hand_number": 400,
            "opp_postflop_check_count": 3,
            "opp_postflop_raise_density": 0.0,
            "opp_postflop_reraise_density": 0.0,
            "opp_high_commit_pressure_density": 0.0,
            "my_raises_this_street": 0,
            "my_raises_this_hand": 0,
            "bankroll_0": 100.0,
            "bankroll_1": 100.0,
        }
        import submission.strategy as strat

        with (
            unittest.mock.patch.object(strat, "_flush_danger", return_value=1),
            unittest.mock.patch.object(strat, "_paired_board_danger", return_value=0),
            unittest.mock.patch.object(strat, "_is_straight_dominated", return_value=True),
            unittest.mock.patch.object(strat, "_is_non_nut_flush", return_value=False),
            unittest.mock.patch.object(strat, "_opp_flush_signal", return_value=0),
            unittest.mock.patch.object(strat, "_get_strategy_table", return_value=None),
        ):
            a = decide_action(
                0.92,
                obs,
                om,
                info=info,
                hand_rank_class=5,
            )
        self.assertEqual(a[0], 1, msg="expected value raise, got %r" % (a,))


if __name__ == "__main__":
    unittest.main()
