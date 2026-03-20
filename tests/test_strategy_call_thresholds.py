"""Unit tests for _build_call_requirements in submission.strategy."""

import unittest

from submission.strategy import (
    CALL_MARGIN_ABOVE_POT_ODDS,
    RIVER_SMALL_BET_EXTRA_MARGIN,
    RIVER_SMALL_BET_FRAC,
    TEXTURE_REF_FRAC,
    _build_call_requirements,
)


def _base_kwargs():
    return dict(
        street=3,
        commit_band=0,
        early_phase=False,
        loss_pressure=0,
        opp_postflop_pressure=0,
        opp_reraise_pressure=0,
        opp_high_commit_pressure=0,
        multi_street_raise_pressure=False,
        second_raise_this_street=False,
        opp_flush_sig=0,
        opp_discarded_pair=False,
        adapted=False,
        opp_agg=1.0,
        my_bet=2,
        my_raises_this_hand=0,
        hand_rank_class=None,
        non_nut_flush=False,
        straight_dominated=False,
    )


class TestBuildCallRequirements(unittest.TestCase):
    def test_build_call_requirements_returns_tuple(self):
        pot_odds = 7 / (34 + 14)  # continue_cost=7, pot=34
        min_eq, floor = _build_call_requirements(
            pot_odds=pot_odds,
            continue_cost=7,
            pot=34,
            flush_danger=0,
            pair_danger=0,
            include_extended_opp_pressure=False,
            **_base_kwargs(),
        )
        self.assertIsInstance(min_eq, float)
        self.assertIsInstance(floor, float)
        self.assertGreaterEqual(min_eq, 0.0)
        self.assertLessEqual(min_eq, 1.0)
        self.assertGreaterEqual(floor, 0.0)
        self.assertLessEqual(floor, 1.0)

    def test_river_small_bet_caps_min_equity(self):
        """Cheap river bets should not require absurd equity above pot_odds + margin + relief."""
        pot_odds = 7 / (34 + 14)
        min_eq, _ = _build_call_requirements(
            pot_odds=pot_odds,
            continue_cost=7,
            pot=34,
            flush_danger=1,
            pair_danger=2,
            include_extended_opp_pressure=False,
            **_base_kwargs(),
        )
        cap = pot_odds + CALL_MARGIN_ABOVE_POT_ODDS + RIVER_SMALL_BET_EXTRA_MARGIN
        bf = 7 / 34
        self.assertLessEqual(bf, RIVER_SMALL_BET_FRAC)
        self.assertLessEqual(min_eq, cap + 1e-9)

    def test_texture_scaling_reduces_pair_bump_at_small_bet_fraction(self):
        """Smaller bet vs pot scales down texture bumps (pair danger)."""
        pot_odds_large = 18 / (100 + 36)
        pot_odds_small = 6 / (100 + 12)
        self.assertLess(6 / 100, TEXTURE_REF_FRAC)
        self.assertLess(18 / 100, TEXTURE_REF_FRAC)
        _, floor_large = _build_call_requirements(
            pot_odds=pot_odds_large,
            continue_cost=18,
            pot=100,
            flush_danger=0,
            pair_danger=2,
            include_extended_opp_pressure=False,
            **_base_kwargs(),
        )
        _, floor_small = _build_call_requirements(
            pot_odds=pot_odds_small,
            continue_cost=6,
            pot=100,
            flush_danger=0,
            pair_danger=2,
            include_extended_opp_pressure=False,
            **_base_kwargs(),
        )
        self.assertLess(floor_small, floor_large)

    def test_marginal_includes_extra_pressure_higher_than_medium(self):
        """Extended opp pressure adds bumps only for marginal (include_extended True)."""
        pot_odds = 0.2
        kwargs = dict(
            pot_odds=pot_odds,
            street=2,
            continue_cost=10,
            pot=50,
            commit_band=0,
            early_phase=False,
            loss_pressure=0,
            opp_postflop_pressure=1,
            opp_reraise_pressure=2,
            opp_high_commit_pressure=0,
            multi_street_raise_pressure=False,
            second_raise_this_street=False,
            flush_danger=0,
            pair_danger=0,
            opp_flush_sig=0,
            opp_discarded_pair=False,
            adapted=False,
            opp_agg=1.0,
            my_bet=5,
            my_raises_this_hand=0,
            hand_rank_class=None,
            non_nut_flush=False,
            straight_dominated=False,
        )
        min_m, fl_m = _build_call_requirements(**kwargs, include_extended_opp_pressure=False)
        min_x, fl_x = _build_call_requirements(**kwargs, include_extended_opp_pressure=True)
        self.assertGreaterEqual(min_x, min_m)
        self.assertGreaterEqual(fl_x, fl_m)


if __name__ == "__main__":
    unittest.main()
