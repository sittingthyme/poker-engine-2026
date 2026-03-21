"""
Hybrid AI Poker Bot – CMU 2026 Tournament

Combines Monte Carlo equity estimation with opponent modelling and
adaptive betting strategy.  All heavy work is delegated to modules
inside the submission/ package.
"""

from agents.agent import Agent
from gym_env import PokerEnv

from submission.equity import (
    compute_equity,
    compute_equity_best2_of5,
    compute_equity_best2_of5_vs_raise_shape,
    compute_equity_best2_of5_vs_shove_top15,
    compute_equity_vs_flush_draw,
    compute_equity_vs_board_pair,
    compute_equity_vs_straight_draw,
    best_discard,
    get_hand_rank_class,
    get_hand_rank_class_partial,
)
from submission.opponent_model import OpponentModel
from submission.opponent_range import OpponentRangeModel, analyze_opponent_discards
from submission.strategy import (
    MIN_HANDS_FOR_ADAPT,
    BANDIT_REVIEW_INTERVAL,
    StrategyBandit,
    decide_action,
)


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType

        # Persistent across the entire 1000-hand match
        self.opp_model = OpponentModel()
        self.opp_range = OpponentRangeModel()
        self._preflop_opp_max_bet: int = 0
        self._bandit = StrategyBandit()
        self._bandit_interval_start_hand: int = -1  # -1 = not yet seeded

        # Per-hand bookkeeping
        self._current_hand: int | None = None
        self._my_discarded: list[int] = []
        self._blind_position: int = 0  # 0=SB, 1=BB; detected at hand start
        # Self-tracked state for fold-to-win (works even when runner doesn't pass bankroll/hand_number)
        self._my_cumulative_reward: float = 0.0
        self._hands_played: int = 0
        self._last_cards_hash: int = 0  # detect new hand by card change
        self._hand_start_bankroll: float = 0.0
        self._recent_hand_deltas: list[float] = []
        self._my_raises_by_street: dict[int, int] = {}
        self._my_raises_this_hand: int = 0
        self._opp_postflop_raise_events: list[int] = []
        self._opp_postflop_reraise_events: list[int] = []
        self._opp_high_commit_pressure_events: list[int] = []
        self._opp_turn_raise_events: list[int] = []
        # After we RAISE, next observe records opponent response for fold_to_big_bet / call_small_bet.
        self._pending_response_to_our_bet: tuple[int, float] | None = None
        self._last_sims_log_hand: int | None = None  # one [SIMS] line per hand in production

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def __name__(self):
        return "PlayerAgent"

    def act(self, observation, reward, terminated, truncated, info):
        """Choose an action given the current observation."""
        info = info or {}
        self._my_cumulative_reward += float(reward)

        # Detect new hand: prefer info["hand_number"], fall back to card-change detection
        cards_hash = hash(tuple(observation.get("my_cards", ())))
        if "hand_number" in info:
            hand_num = info["hand_number"]
        elif cards_hash != self._last_cards_hash and observation.get("street", 0) == 0:
            self._hands_played += 1
            hand_num = self._hands_played
        else:
            hand_num = self._current_hand if self._current_hand is not None else 0
        self._last_cards_hash = cards_hash

        # Detect new hand → reset per-hand state & tell opponent model
        if hand_num != self._current_hand:
            # Finalize previous hand delta for lightweight match-state adaptation.
            if self._current_hand is not None:
                hand_delta = self._my_cumulative_reward - self._hand_start_bankroll
                self._recent_hand_deltas.append(hand_delta)
                if len(self._recent_hand_deltas) > 40:
                    self._recent_hand_deltas = self._recent_hand_deltas[-40:]

            self._current_hand = hand_num
            self._my_discarded = []
            self._my_raises_by_street = {}
            self._my_raises_this_hand = 0
            self._hand_start_bankroll = self._my_cumulative_reward
            self.opp_model.new_hand()
            self.opp_range.new_hand()
            self._preflop_opp_max_bet = 0
            self._pending_response_to_our_bet = None

            # Meta-controller: UCB1 strategy bandit.
            if self._bandit_interval_start_hand < 0:
                # First hand of the match — seed the bandit.
                self._bandit.select_strategy()
                self._bandit.begin_interval(self._my_cumulative_reward)
                self._bandit_interval_start_hand = hand_num
                self.logger.info(
                    "Bandit initialised – %s", self._bandit.summary()
                )
            else:
                hands_in_interval = hand_num - self._bandit_interval_start_hand
                if hands_in_interval >= BANDIT_REVIEW_INTERVAL:
                    delta = self._bandit.end_interval(
                        self._my_cumulative_reward,
                        hands_in_interval,
                    )
                    self._bandit.select_strategy()
                    self._bandit.begin_interval(self._my_cumulative_reward)
                    self._bandit_interval_start_hand = hand_num
                    self.logger.info(
                        "Bandit review (hand %d, delta=%+.0f, ~%+.2f/hand) – %s",
                        hand_num,
                        delta,
                        delta / max(1, hands_in_interval),
                        self._bandit.summary(),
                    )
            # Detect blind position from the first observation of the hand.
            # Street 0 start: SB posts 1, BB posts 2.
            my_bet = observation.get("my_bet", 0)
            opp_bet = observation.get("opp_bet", 0)
            if my_bet == 1 and opp_bet == 2:
                self._blind_position = 0  # SB
            elif my_bet == 2 and opp_bet == 1:
                self._blind_position = 1  # BB
            else:
                # Use engine's blind_position when bets don't reveal it (e.g. both limped)
                self._blind_position = observation.get("blind_position", 0)

        # Inject computed fields that may be missing in API mode
        observation["pot_size"] = observation.get(
            "pot_size", observation["my_bet"] + observation["opp_bet"]
        )
        observation["blind_position"] = observation.get(
            "blind_position", self._blind_position
        )

        if observation.get("street", 0) == 0:
            self._preflop_opp_max_bet = max(
                self._preflop_opp_max_bet,
                int(observation.get("opp_bet", 0)),
            )

        valid = observation["valid_actions"]

        # ---- Discard phase (mandatory on flop) ----
        if valid[self.action_types.DISCARD.value]:
            self._pending_response_to_our_bet = None
            return self._do_discard(observation)

        # ---- Betting phase ----
        action = self._do_bet(observation, info, hand_num)
        if action[0] == self.action_types.RAISE.value:
            street = int(observation.get("street", 0))
            pot = float(
                observation.get(
                    "pot_size",
                    observation.get("my_bet", 0) + observation.get("opp_bet", 0),
                )
            )
            amt = int(action[1])
            frac = amt / max(1.0, pot)
            self._pending_response_to_our_bet = (street, min(2.0, max(0.0, frac)))
        else:
            self._pending_response_to_our_bet = None
        return action

    def observe(self, observation, reward, terminated, truncated, info):
        """
        Called when it is NOT our turn – the opponent just acted.
        We use this to update the opponent model.
        """
        self._my_cumulative_reward += float(reward)
        # Track opponent action from the extra field injected by the engine
        opp_action_name = observation.get("opp_last_action", "None")
        if self._pending_response_to_our_bet is not None and opp_action_name not in (
            "None",
            None,
        ):
            st, frac = self._pending_response_to_our_bet
            self.opp_model.record_response_to_our_bet(st, str(opp_action_name), frac)
            self._pending_response_to_our_bet = None
        if opp_action_name not in ("None", None):
            street = observation.get("street", 0)
            if street == 0:
                self._preflop_opp_max_bet = max(
                    self._preflop_opp_max_bet,
                    int(observation.get("opp_bet", 0)),
                )
            # We don't know the exact raise amount from observe; approximate
            raise_amt = max(0, observation.get("opp_bet", 0) - observation.get("my_bet", 0))
            pot_size = observation.get("pot_size", observation.get("my_bet", 0) + observation.get("opp_bet", 0))
            self.opp_model.record_action(street, opp_action_name, raise_amount=raise_amt, pot_size=pot_size)
            if street >= 1 and opp_action_name in ("RAISE", "CALL", "CHECK", "FOLD"):
                self._opp_postflop_raise_events.append(1 if opp_action_name == "RAISE" else 0)
                if len(self._opp_postflop_raise_events) > 40:
                    self._opp_postflop_raise_events = self._opp_postflop_raise_events[-40:]
            if street >= 1 and opp_action_name in ("RAISE", "CALL", "CHECK", "FOLD"):
                is_reraise = opp_action_name == "RAISE" and observation.get("my_bet", 0) > 0
                self._opp_postflop_reraise_events.append(1 if is_reraise else 0)
                if len(self._opp_postflop_reraise_events) > 40:
                    self._opp_postflop_reraise_events = self._opp_postflop_reraise_events[-40:]
                high_commit_pressure = (
                    opp_action_name == "RAISE"
                    and max(observation.get("my_bet", 0), observation.get("opp_bet", 0)) >= 50
                )
                self._opp_high_commit_pressure_events.append(1 if high_commit_pressure else 0)
                if len(self._opp_high_commit_pressure_events) > 40:
                    self._opp_high_commit_pressure_events = self._opp_high_commit_pressure_events[-40:]
            if street == 2 and opp_action_name in ("RAISE", "CALL", "CHECK", "FOLD"):
                self._opp_turn_raise_events.append(1 if opp_action_name == "RAISE" else 0)
                if len(self._opp_turn_raise_events) > 30:
                    self._opp_turn_raise_events = self._opp_turn_raise_events[-30:]

        if terminated:
            self.opp_model.end_hand()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preflop_action_strength(self) -> float:
        """[0, 1] — higher when opponent put in larger pre-flop bets (opens / 3-bets / jams)."""
        m = self._preflop_opp_max_bet
        if m <= 2:
            return 0.48
        if m <= 4:
            return 0.55
        if m <= 8:
            return 0.62
        if m <= 14:
            return 0.72
        if m <= 24:
            return 0.82
        return 0.90

    def _do_discard(self, obs) -> tuple[int, int, int, int]:
        """Pick the best 2 cards to keep from 5 hole cards with time-aware scaling."""
        my_cards = [c for c in obs["my_cards"] if c != -1]
        community = [c for c in obs["community_cards"] if c != -1]
        opp_disc = [c for c in obs.get("opp_discarded_cards", [-1, -1, -1]) if c != -1]
        if len(opp_disc) == 3:
            self.opp_range.update_from_discards(
                opp_disc,
                preflop_action_strength=self._preflop_action_strength(),
            )

        # Time-aware scaling for discard phase
        time_left = obs.get("time_left", 500.0)
        time_used = obs.get("time_used", 0.0)
        total = time_used + time_left
        time_limit = total if total > 0 else 1000.0  # Phase 2 default
        time_ratio = time_left / time_limit if time_limit > 0 else 1.0

        # Base discard simulations — keep modest to avoid time forfeits (was 800).
        BASE_SIMS_DISCARD = 80
        base_sims = BASE_SIMS_DISCARD

        # Apply time scaling
        if time_ratio < 0.30:
            time_multiplier = 0.5
        elif time_ratio < 0.50:
            time_multiplier = 0.7
        elif time_ratio < 0.70:
            time_multiplier = 0.85
        else:
            time_multiplier = 1.0

        sims_per_pair = max(80, int(base_sims * time_multiplier))

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

    def _do_bet(self, obs, info: dict | None = None, hand_num: int | None = None) -> tuple[int, int, int, int]:
        """Equity-driven betting decision with adaptive simulation scaling."""
        info = dict(info) if info else {}
        # When match/tournament doesn't pass bankroll, use our own tracking so fold-to-win works
        need_bankroll = (
            "bankroll_0" not in info or "bankroll_1" not in info
            or info.get("bankroll_0") is None or info.get("bankroll_1") is None
        )
        if need_bankroll:
            acting = obs.get("acting_agent", 0)
            if acting == 0:
                info["bankroll_0"] = self._my_cumulative_reward
                info["bankroll_1"] = -self._my_cumulative_reward
            else:
                info["bankroll_0"] = -self._my_cumulative_reward
                info["bankroll_1"] = self._my_cumulative_reward
        if info.get("hand_number") is None and hand_num is not None:
            info["hand_number"] = hand_num
        info["opp_preflop_raise_rate"] = float(self.opp_model.preflop_raise_rate)
        info["opp_preflop_fold_rate"] = float(self.opp_model.preflop_fold_rate)
        info["opp_river_raise_rate"] = float(self.opp_model.river_raise_rate)
        if self._recent_hand_deltas:
            info["recent_delta_10"] = float(sum(self._recent_hand_deltas[-10:]))
            info["recent_delta_30"] = float(sum(self._recent_hand_deltas[-30:]))
        else:
            info["recent_delta_10"] = 0.0
            info["recent_delta_30"] = 0.0
        if self._opp_postflop_raise_events:
            window = self._opp_postflop_raise_events[-24:]
            info["opp_postflop_raise_density"] = float(sum(window) / len(window))
        else:
            info["opp_postflop_raise_density"] = 0.0
        if self._opp_turn_raise_events:
            tw = self._opp_turn_raise_events[-30:]
            info["opp_turn_raise_density"] = float(sum(tw) / len(tw))
        else:
            info["opp_turn_raise_density"] = 0.0
        if self._opp_postflop_reraise_events:
            rw = self._opp_postflop_reraise_events[-24:]
            info["opp_postflop_reraise_density"] = float(sum(rw) / len(rw))
        else:
            info["opp_postflop_reraise_density"] = 0.0
        if self._opp_high_commit_pressure_events:
            hw = self._opp_high_commit_pressure_events[-24:]
            info["opp_high_commit_pressure_density"] = float(sum(hw) / len(hw))
        else:
            info["opp_high_commit_pressure_density"] = 0.0
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

        # Critical time guard: never burn the clock on heavy MC (forfeit protection).
        if time_ratio < 0.10:
            n_sims = 50
        else:
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
                time_multiplier = 0.5
            elif time_ratio < 0.50:
                time_multiplier = 0.7
            elif time_ratio < 0.70:
                time_multiplier = 0.85
            else:
                time_multiplier = 1.0

            n_sims = int(base_sims * pot_multiplier * time_multiplier)
            n_sims = max(100, min(n_sims, 1200))

        _log_hn = int(info.get("hand_number") or hand_num or 0)
        if _log_hn > 0 and self._last_sims_log_hand != _log_hn:
            self._last_sims_log_hand = _log_hn
            print(f"[SIMS] street={street} n_sims={n_sims} ratio={time_ratio:.2f}")

        # Preflop: use best 2 of 5 equity; post-flop: use our 2 kept cards
        if street == 0 and len(my_cards) == 5:
            # Cached 450-sim preflop equity (no live MC on every decision).
            equity = compute_equity_best2_of5(my_cards, num_simulations=450)
            my_bet = obs.get("my_bet", 0)
            opp_bet = obs.get("opp_bet", 0)
            _pf_cc = max(0, opp_bet - my_bet)
            # Shove-range MC for small+ jams (4+ chips — covers ~6-chip shove bots).
            if opp_bet > my_bet and _pf_cc >= 4:
                info["preflop_equity_vs_shove"] = compute_equity_best2_of5_vs_shove_top15(
                    my_cards, num_simulations=min(n_sims, 300)
                )
            # Facing a real raise: blend with equity vs a plausible raising range
            # (uniform random villain massively over-calls weak 5-card bundles).
            if opp_bet > my_bet and opp_bet > 2:
                eq_raise = compute_equity_best2_of5_vs_raise_shape(
                    my_cards, num_simulations=min(n_sims, 450)
                )
                w = 0.55
                adapted = self.opp_model.hands_seen >= MIN_HANDS_FOR_ADAPT
                if adapted:
                    if self.opp_model.is_tight():
                        w += 0.10
                    elif self.opp_model.is_loose():
                        w -= 0.08
                    sb0 = self.opp_model.streets[0]
                    if sb0.raises >= 8 and sb0.actions > 0:
                        rr = sb0.raises / sb0.actions
                        if rr > 0.35:
                            w += 0.05
                w = max(0.40, min(0.72, w))
                equity = (1.0 - w) * equity + w * eq_raise
        else:
            range_bias = 0.0
            if self.opp_model.hands_seen >= MIN_HANDS_FOR_ADAPT:
                sb = self.opp_model.streets[min(street, 3)]
                if sb.actions >= 6:
                    range_bias = min(0.82, 0.12 + 0.75 * sb.raise_rate)
            equity = compute_equity(
                my_cards[:2],
                community,
                opp_discarded=opp_disc if opp_disc else None,
                my_discarded=self._my_discarded if self._my_discarded else None,
                num_simulations=n_sims,
                opponent_range_bias=range_bias,
            )

        # Post-flop: discard signals → range-based equity blend
        if street >= 1 and len(opp_disc) == 3 and len(community) >= 3:
            sig = analyze_opponent_discards(opp_disc, community)
            info["opp_flush_signal"] = sig.opp_flush_signal
            info["opp_discarded_pair"] = sig.opp_discarded_pair
            info["opp_likely_has_pair"] = sig.opp_likely_has_pair
            info["opp_likely_has_full_house"] = sig.opp_likely_has_full_house
            info["opp_straight_signal"] = sig.opp_straight_signal
            info["opp_kept_high_cards"] = sig.opp_kept_high_cards
            info["opp_kept_high_flush"] = sig.opp_kept_high_flush

            range_sims = min(n_sims // 2, 400)
            _disc = self._my_discarded if self._my_discarded else None

            # Pick the strongest signal and blend equity against that range.
            # Only apply one blend to avoid compounding adjustments.
            if sig.opp_flush_signal >= 2 and sig.flush_suit >= 0:
                eq_vs_range = compute_equity_vs_flush_draw(
                    my_cards[:2], community,
                    flush_suit=sig.flush_suit,
                    opp_discarded=opp_disc, my_discarded=_disc,
                    num_simulations=range_sims,
                )
                # Stronger blend vs flush-made range when discard read is clear (was 0.4 / 0.6).
                blend_w = 0.75 if sig.opp_kept_high_flush else 0.55
                equity = (1.0 - blend_w) * equity + blend_w * eq_vs_range
            elif sig.opp_straight_signal >= 2 and sig.straight_helping_ranks:
                eq_vs_range = compute_equity_vs_straight_draw(
                    my_cards[:2], community,
                    straight_ranks=sig.straight_helping_ranks,
                    opp_discarded=opp_disc, my_discarded=_disc,
                    num_simulations=range_sims,
                )
                equity = 0.6 * equity + 0.4 * eq_vs_range
            elif sig.opp_likely_has_pair:
                eq_vs_range = compute_equity_vs_board_pair(
                    my_cards[:2], community,
                    opp_discarded=opp_disc, my_discarded=_disc,
                    num_simulations=range_sims,
                )
                equity = 0.7 * equity + 0.3 * eq_vs_range

        # Post-flop: evaluate hand strength so we never fold trips+
        hand_rank_class = None
        if street >= 1 and len(my_cards) >= 2 and len(community) >= 3:
            if len(community) == 5:
                hand_rank_class = get_hand_rank_class(my_cards, community)
            else:
                hand_rank_class = get_hand_rank_class_partial(my_cards, community)

        info["my_raises_this_street"] = int(self._my_raises_by_street.get(street, 0))
        info["my_raises_this_hand"] = int(self._my_raises_this_hand)
        action = decide_action(
            equity, obs, self.opp_model,
            info=info or {},
            hand_rank_class=hand_rank_class,
            strategy_profile=self._bandit.get_profile(),
        )
        if action[0] == self.action_types.RAISE.value:
            self._my_raises_by_street[street] = self._my_raises_by_street.get(street, 0) + 1
            self._my_raises_this_hand += 1

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
