"""
Approximation of raise-heavy tournament opponents (e.g. log profile in logs/3.cssv):
high postflop raise share, frequent aggression. Not the real AA998 binary.
"""
from __future__ import annotations

import random

from agents.prob_agent import ProbabilityAgent, action_types


class AA998StyleAgent(ProbabilityAgent):
    """Monte Carlo equity like ProbabilityAgent, but raises far more postflop."""

    def __init__(self, stream: bool = False, rng: random.Random | None = None):
        super().__init__(stream=stream)
        self._rng = rng or random.Random()

    def __name__(self):
        return "AA998StyleAgent"

    def act(self, observation, reward, terminated, truncated, info):
        if observation["valid_actions"][action_types.DISCARD.value]:
            return super().act(observation, reward, terminated, truncated, info)

        # Preflop: same baseline as ProbabilityAgent (opens / defends).
        if observation["street"] == 0:
            return super().act(observation, reward, terminated, truncated, info)

        my_cards = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        opp_disc = list(observation.get("opp_discarded_cards", [-1, -1, -1]))
        if len(my_cards) != 2:
            my_cards = my_cards[:2]

        equity = self._compute_equity(
            my_cards, community, opp_disc, num_simulations=280
        )
        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["my_bet"] + observation["opp_bet"]
        pot_odds = (
            continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0.0
        )
        va = observation["valid_actions"]

        if equity > 0.70 and va[action_types.RAISE.value]:
            amt = int(pot_size * 0.68)
            amt = max(amt, observation["min_raise"])
            amt = min(amt, observation["max_raise"])
            return (action_types.RAISE.value, amt, 0, 0)

        if continue_cost > 0 and equity >= pot_odds + 0.02 and va[action_types.RAISE.value]:
            if self._rng.random() < 0.58:
                amt = max(
                    observation["min_raise"],
                    min(int(pot_size * 0.55), observation["max_raise"]),
                )
                return (action_types.RAISE.value, amt, 0, 0)
        if continue_cost > 0 and equity >= pot_odds and va[action_types.CALL.value]:
            return (action_types.CALL.value, 0, 0, 0)

        if continue_cost <= 0 and va[action_types.RAISE.value] and equity > 0.30:
            if self._rng.random() < 0.40:
                amt = max(
                    observation["min_raise"],
                    min(int(pot_size * 0.50), observation["max_raise"]),
                )
                return (action_types.RAISE.value, amt, 0, 0)
        if va[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, 0, 0)
        if continue_cost > 0 and va[action_types.CALL.value]:
            return (action_types.CALL.value, 0, 0, 0)
        return (action_types.FOLD.value, 0, 0, 0)
