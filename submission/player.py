"""
Hybrid AI Poker Bot – CMU 2026 Tournament

Combines Monte Carlo equity estimation with opponent modelling and
adaptive betting strategy.  All heavy work is delegated to modules
inside the submission/ package.
"""

from agents.agent import Agent
from gym_env import PokerEnv

from submission.equity import compute_equity, best_discard
from submission.opponent_model import OpponentModel
from submission.strategy import decide_action
from submission.strategy_table import StrategyTable


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType

        # Persistent across the entire 1000-hand match
        self.opp_model = OpponentModel()
        self.strategy_table = StrategyTable.load()

        # Per-hand bookkeeping
        self._current_hand: int | None = None
        self._my_discarded: list[int] = []
        self._blind_position: int = 0  # 0=SB, 1=BB; detected at hand start

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def __name__(self):
        return "PlayerAgent"

    def act(self, observation, reward, terminated, truncated, info):
        """Choose an action given the current observation."""
        hand_num = info.get("hand_number", 0)

        # Detect new hand → reset per-hand state & tell opponent model
        if hand_num != self._current_hand:
            self._current_hand = hand_num
            self._my_discarded = []
            self.opp_model.new_hand()
            # Detect blind position from the first observation of the hand.
            # Street 0 start: SB posts 1, BB posts 2.
            my_bet = observation.get("my_bet", 0)
            opp_bet = observation.get("opp_bet", 0)
            if my_bet == 1 and opp_bet == 2:
                self._blind_position = 0  # SB
            elif my_bet == 2 and opp_bet == 1:
                self._blind_position = 1  # BB
            else:
                self._blind_position = 0  # fallback

        # Inject computed fields that may be missing in API mode
        observation["pot_size"] = observation.get(
            "pot_size", observation["my_bet"] + observation["opp_bet"]
        )
        observation["blind_position"] = observation.get(
            "blind_position", self._blind_position
        )

        valid = observation["valid_actions"]

        # ---- Discard phase (mandatory on flop) ----
        if valid[self.action_types.DISCARD.value]:
            return self._do_discard(observation)

        # ---- Betting phase ----
        return self._do_bet(observation)

    def observe(self, observation, reward, terminated, truncated, info):
        """
        Called when it is NOT our turn – the opponent just acted.
        We use this to update the opponent model.
        """
        # Track opponent action from the extra field injected by the engine
        opp_action_name = observation.get("opp_last_action", "None")
        if opp_action_name not in ("None", None):
            street = observation.get("street", 0)
            # We don't know the exact raise amount from observe; approximate
            raise_amt = max(0, observation.get("opp_bet", 0) - observation.get("my_bet", 0))
            pot_size = observation.get("pot_size", observation.get("my_bet", 0) + observation.get("opp_bet", 0))
            self.opp_model.record_action(street, opp_action_name, raise_amount=raise_amt, pot_size=pot_size)

        if terminated:
            self.opp_model.end_hand()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_discard(self, obs) -> tuple[int, int, int, int]:
        """Pick the best 2 cards to keep from 5 hole cards with time-aware scaling."""
        my_cards = [c for c in obs["my_cards"] if c != -1]
        community = [c for c in obs["community_cards"] if c != -1]
        opp_disc = [c for c in obs.get("opp_discarded_cards", [-1, -1, -1]) if c != -1]

        # Time-aware scaling for discard phase
        time_left = obs.get("time_left", 500.0)
        time_used = obs.get("time_used", 0.0)
        total = time_used + time_left
        time_limit = total if total > 0 else 1000.0  # Phase 2 default
        time_ratio = time_left / time_limit if time_limit > 0 else 1.0

        # Base discard simulations (evaluating 10 pairs, so each pair gets sims_per_pair)
        base_sims = 350  # Phase 2: increased from 250 for better discard decisions

        # Apply time scaling
        if time_ratio < 0.30:
            time_multiplier = 0.5
        elif time_ratio < 0.50:
            time_multiplier = 0.7
        elif time_ratio < 0.70:
            time_multiplier = 0.85
        else:
            time_multiplier = 1.0

        sims_per_pair = max(100, int(base_sims * time_multiplier))

        keep_i, keep_j, eq = best_discard(
            my_cards, community, opp_discarded=opp_disc if opp_disc else None,
            sims_per_pair=sims_per_pair,
        )

        # Remember what we're discarding for later equity calls
        self._my_discarded = [my_cards[k] for k in range(5) if k != keep_i and k != keep_j]

        self.logger.debug(
            f"Discard: keep [{keep_i},{keep_j}] equity={eq:.2f} (sims={sims_per_pair})"
        )
        return (self.action_types.DISCARD.value, 0, keep_i, keep_j)

    def _do_bet(self, obs) -> tuple[int, int, int, int]:
        """Equity-driven betting decision with adaptive simulation scaling."""
        my_cards = [c for c in obs["my_cards"] if c != -1]
        community = [c for c in obs["community_cards"] if c != -1]
        opp_disc = [c for c in obs.get("opp_discarded_cards", [-1, -1, -1]) if c != -1]

        street = obs["street"]
        pot = obs.get("pot_size", obs["my_bet"] + obs["opp_bet"])
        time_left = obs.get("time_left", 500.0)
        time_used = obs.get("time_used", 0.0)
        total = time_used + time_left
        time_limit = total if total > 0 else 1000.0  # Phase 2 default
        time_ratio = time_left / time_limit if time_limit > 0 else 1.0

        # Base simulation counts (Phase 2: increased for 1000s time bank)
        if street <= 1:
            base_sims = 450  # Pre-flop/flop
        elif street == 2:
            base_sims = 550  # Turn
        else:
            base_sims = 850  # River: most critical

        # Scale by pot size importance (larger pots = more important decisions)
        # Pot size typically ranges from 2-200+ chips
        pot_multiplier = 1.0 + min(0.5, (pot - 2) / 200.0)  # 1.0 to 1.5x based on pot size

        # Time-aware scaling: reduce simulations when time is running low
        if time_ratio < 0.30:
            # Critical: less than 30% time remaining
            time_multiplier = 0.5  # Cut simulations in half
        elif time_ratio < 0.50:
            # Warning: less than 50% time remaining
            time_multiplier = 0.7  # Reduce by 30%
        elif time_ratio < 0.70:
            # Caution: less than 70% time remaining
            time_multiplier = 0.85  # Reduce by 15%
        else:
            # Safe: plenty of time remaining
            time_multiplier = 1.0

        # Calculate final simulation count
        n_sims = int(base_sims * pot_multiplier * time_multiplier)
        # Ensure minimum of 100 sims for reasonable accuracy
        n_sims = max(100, min(n_sims, 1200))  # Phase 2: cap at 1200

        equity = compute_equity(
            my_cards[:2],
            community,
            opp_discarded=opp_disc if opp_disc else None,
            my_discarded=self._my_discarded if self._my_discarded else None,
            num_simulations=n_sims,
        )

        action = decide_action(equity, obs, self.opp_model, strategy_table=self.strategy_table)

        # Log time usage for monitoring
        if time_ratio < 0.50:
            self.logger.debug(
                f"Street {street}: equity={equity:.2f} → action={self.action_types(action[0]).name} amt={action[1]} "
                f"(sims={n_sims}, time_left={time_left:.1f}s, ratio={time_ratio:.2f})"
            )
        else:
            self.logger.debug(
                f"Street {street}: equity={equity:.2f} → action={self.action_types(action[0]).name} amt={action[1]}"
            )
        return action
