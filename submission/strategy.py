"""
Betting strategy for the hybrid poker bot.

Position-aware, equity-driven decisions with adaptive bluffing
tuned to the opponent model collected over the match.
"""

from __future__ import annotations

import logging
import math
import random
from submission.opponent_model import OpponentModel

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Meta-controller: UCB1 multi-armed bandit over strategy profiles
# ---------------------------------------------------------------------------

STRATEGY_PROFILES: dict[int, dict[str, float]] = {
    # 0 – GTO Baseline (current defaults)
    0: {},
    # 1 – Aggressive / Bluffy (wider VPIP, frequent bluffs, bigger sizing)
    1: {
        "BASE_STRONG_EQUITY": -0.10,     # value-bet with weaker hands
        "BASE_MEDIUM_EQUITY": -0.12,     # call / continue wider
        "BASE_BLUFF_FREQ": +0.18,        # bluff much more often
        "MAX_BLUFF_FREQ": +0.20,
        "SEMI_BLUFF_EQUITY_MIN": -0.05,  # bluff-raise with thinner draws
        "VALUE_BET_EQUITY": -0.10,       # stab wider when checked to
        "CBET_EQUITY_THRESHOLD": -0.10,  # c-bet light
        "PREFLOP_OPEN_RAISE_MULTIPLIER": +1.5,  # bigger opens
    },
    # 2 – Tight / Conservative (premium only, never bluff)
    2: {
        "BASE_STRONG_EQUITY": +0.05,
        "BASE_MEDIUM_EQUITY": +0.08,
        "BASE_BLUFF_FREQ": -0.14,        # almost never bluff
        "MAX_BLUFF_FREQ": -0.25,
        "SEMI_BLUFF_EQUITY_MIN": +0.10,  # only semi-bluff with very strong draws
        "SEMI_BLUFF_EQUITY_MAX": -0.05,  # narrow the window
        "VALUE_BET_EQUITY": +0.06,
        "CBET_EQUITY_THRESHOLD": +0.10,
    },
    # 3 – Discard exploit (pressure when villain kept weak / range is weak post-discard)
    3: {
        "BASE_BLUFF_FREQ": +0.10,
        "VALUE_BET_EQUITY": -0.06,
        "CBET_EQUITY_THRESHOLD": -0.08,
        "BASE_STRONG_EQUITY": -0.05,
    },
    # 4 – Barrel exploit (vs hyper-aggressive postflop: less air, more value / thin rivers)
    4: {
        "BASE_BLUFF_FREQ": -0.12,
        "VALUE_BET_EQUITY": -0.08,
        "CBET_EQUITY_THRESHOLD": -0.06,
        "BASE_STRONG_EQUITY": -0.04,
    },
}

BANDIT_REVIEW_INTERVAL = 75  # re-evaluate every N hands (was 20; short intervals are noise)


_BANDIT_NAMES = {
    0: "Baseline",
    1: "Aggressive",
    2: "Tight",
    3: "DiscardExploit",
    4: "BarrelExploit",
}


def _merge_profile_deltas(base: dict[str, float], extra: dict[str, float]) -> dict[str, float]:
    """Layer ``extra`` on top of ``base`` (additive for numeric tuning keys)."""
    out = dict(base)
    for k, v in extra.items():
        out[k] = out.get(k, 0.0) + float(v)
    return out


def opponent_reactive_adjustments(opp_model: OpponentModel, *, adapted: bool) -> dict[str, float]:
    """
    Targeted parameter adjustments from observed opponent tendencies.
    Merged on top of the active bandit arm in ``decide_action``.
    """
    if not adapted:
        return {}
    adj: dict[str, float] = {}

    # Pot-odds stations: fold preflop a lot but rarely postflop — bluffs lose EV.
    if opp_model.is_calling_station_postflop():
        adj["BASE_BLUFF_FREQ"] = adj.get("BASE_BLUFF_FREQ", 0.0) - 0.14
        adj["MAX_BLUFF_FREQ"] = adj.get("MAX_BLUFF_FREQ", 0.0) - 0.14
        adj["SEMI_BLUFF_EQUITY_MIN"] = adj.get("SEMI_BLUFF_EQUITY_MIN", 0.0) + 0.07
        adj["SEMI_BLUFF_EQUITY_MAX"] = adj.get("SEMI_BLUFF_EQUITY_MAX", 0.0) - 0.04
        # VALUE_BET_EQUITY: handled in decide_action via is_calling_station_postflop (thin value)
        adj["CBET_EQUITY_THRESHOLD"] = adj.get("CBET_EQUITY_THRESHOLD", 0.0) + 0.07
        adj["raise_fraction_bonus"] = adj.get("raise_fraction_bonus", 0.0) - 0.10
        adj["PREFLOP_OPEN_RAISE_MULTIPLIER"] = adj.get("PREFLOP_OPEN_RAISE_MULTIPLIER", 0.0) - 0.75
        _log.debug("reactive: calling_station_postflop → bluff− value bar+, smaller opens")

    # Barrel / hyper-aggressive postflop: tighten flop calls; value rivers; raise strong flop hands.
    elif opp_model.is_hyper_aggressive_postflop():
        adj["flop_call_floor_bump"] = adj.get("flop_call_floor_bump", 0.0) + 0.08
        adj["river_vbet_eq_discount"] = adj.get("river_vbet_eq_discount", 0.0) - 0.10
        adj["force_raise_strong_flop"] = adj.get("force_raise_strong_flop", 0.0) + 1.0
        _log.debug("reactive: hyper_aggressive_postflop → flop call+, river value, force flop raises")

    ftbb = opp_model.fold_to_big_bet_rate
    if ftbb is not None:
        if ftbb > 0.60:
            adj["raise_fraction_bonus"] = adj.get("raise_fraction_bonus", 0.0) + 0.15
            adj["BASE_BLUFF_FREQ"] = adj.get("BASE_BLUFF_FREQ", 0.0) + 0.08
            _log.debug("reactive: fold_to_big_bet=%.2f → raise_fraction + bluff+", ftbb)
        elif ftbb < 0.30:
            adj["raise_fraction_bonus"] = adj.get("raise_fraction_bonus", 0.0) - 0.10
            adj["BASE_BLUFF_FREQ"] = adj.get("BASE_BLUFF_FREQ", 0.0) - 0.08
            adj["MAX_BLUFF_FREQ"] = adj.get("MAX_BLUFF_FREQ", 0.0) - 0.08
            _log.debug("reactive: fold_to_big_bet=%.2f → raise− bluff−", ftbb)

    csb = opp_model.call_small_bet_rate
    if csb is not None:
        if csb > 0.70:
            adj["BASE_BLUFF_FREQ"] = adj.get("BASE_BLUFF_FREQ", 0.0) - 0.12
            adj["VALUE_BET_EQUITY"] = adj.get("VALUE_BET_EQUITY", 0.0) - 0.05
            _log.debug("reactive: call_small_bet=%.2f → bluff− value thinner", csb)

    tend = opp_model.discard_tendency()
    if tend == "keeps_weak":
        adj["BASE_BLUFF_FREQ"] = adj.get("BASE_BLUFF_FREQ", 0.0) - 0.08
        _log.debug("reactive: discard_tendency=keeps_weak → bluff−")
    elif tend == "keeps_strong":
        adj["BASE_STRONG_EQUITY"] = adj.get("BASE_STRONG_EQUITY", 0.0) - 0.04
        _log.debug("reactive: discard_tendency=keeps_strong → value wider")

    if opp_model.is_tight():
        adj["BASE_BLUFF_FREQ"] = adj.get("BASE_BLUFF_FREQ", 0.0) + 0.08
        adj["CBET_EQUITY_THRESHOLD"] = adj.get("CBET_EQUITY_THRESHOLD", 0.0) - 0.08
        _log.debug("reactive: is_tight → bluff+, c-bet lighter")
    elif opp_model.is_loose():
        adj["MAX_BLUFF_FREQ"] = adj.get("MAX_BLUFF_FREQ", 0.0) - 0.12
        _log.debug("reactive: is_loose → max bluff cap−")

    return adj
# Exploration bonus (UCB1). Rewards are **per hand** after normalization — keep same scale.
_UCB_EXPLORE_C = 3.0


class StrategyBandit:
    """UCB1 multi-armed bandit that selects the best strategy profile online."""

    def __init__(self) -> None:
        n = len(STRATEGY_PROFILES)
        self.counts: list[int] = [0] * n
        self.values: list[float] = [0.0] * n
        self._current_arm: int = 0
        self._arm_chips_start: float = 0.0

    @property
    def current_arm(self) -> int:
        return self._current_arm

    def select_strategy(self) -> int:
        """Pick the next arm using UCB1 (called once per review interval)."""
        n_arms = len(STRATEGY_PROFILES)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                self._current_arm = arm
                return arm
        best_arm = 0
        best_ucb = -float("inf")
        t = max(1, sum(self.counts))
        for arm in range(n_arms):
            exploitation = self.values[arm]
            exploration = _UCB_EXPLORE_C * math.sqrt(
                (2.0 * math.log(t)) / max(1, self.counts[arm])
            )
            ucb = exploitation + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm
        self._current_arm = best_arm
        return best_arm

    def begin_interval(self, cumulative_chips: float) -> None:
        """Mark the start of a review interval (called from player)."""
        self._arm_chips_start = cumulative_chips

    def end_interval(self, cumulative_chips: float, hands_in_interval: int) -> float:
        """Close the interval; reward is chip delta **per hand** (variance-normalized)."""
        chip_delta = cumulative_chips - self._arm_chips_start
        h = max(1, int(hands_in_interval))
        reward_per_hand = chip_delta / h
        arm = self._current_arm
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1.0 / n) * reward_per_hand
        return chip_delta

    def get_profile(self) -> dict[str, float]:
        return STRATEGY_PROFILES.get(self._current_arm, {})

    def summary(self) -> str:
        parts = []
        for arm in range(len(STRATEGY_PROFILES)):
            name = _BANDIT_NAMES.get(arm, str(arm))
            parts.append(f"{name}: n={self.counts[arm]} avg={self.values[arm]:+.3f}/hand")
        cur = _BANDIT_NAMES.get(self._current_arm, str(self._current_arm))
        return f"[Bandit active={cur}] " + " | ".join(parts)


def _raise_frac_value(
    *,
    flush_danger: int = 0,
    pair_danger: int = 0,
    street: int = 0,
    is_polarized: bool = False,
) -> float:
    """
    Unified pot-fraction sizing for all raises (value, bluff, semi-bluff).
    On the river, polarized spots (nut value or semi-bluff) may use large overbets.
    """
    # street 3 = river: massive overbets for polarized ranges (nuts or semi-bluff bluffs)
    if street == 3 and is_polarized:
        return random.uniform(1.25, 1.75)
    lo, hi = 0.60, 0.85
    wet = flush_danger + pair_danger
    if wet >= 3:
        lo, hi = 0.68, 0.85
    elif wet >= 2:
        lo, hi = 0.64, 0.85
    elif wet >= 1:
        lo, hi = 0.62, 0.85
    return random.uniform(lo, hi)


# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

# Base equity bands (adjusted dynamically based on opponent)
# 27-card deck: straights/flushes/full house are common; two pair is vulnerable – play tight
BASE_STRONG_EQUITY = 0.85       # value-bet/raise only with very strong hands (was 0.82)
BASE_MEDIUM_EQUITY = 0.65       # call only with solid equity (was 0.60)
# Semi-bluff raises: high-variance draws (not pure air)
SEMI_BLUFF_EQUITY_MIN = 0.30
SEMI_BLUFF_EQUITY_MAX = 0.45

# Pot-size tightening: large pots on turn/river signal multi-street aggression
LARGE_POT_THRESHOLD = 40     # pot >= 40 chips
LARGE_POT_EQUITY_BUMP = 0.04
HUGE_POT_THRESHOLD = 80      # pot >= 80 chips
HUGE_POT_EQUITY_BUMP = 0.04  # cumulative with above → +0.08 for huge pots

# Board flush danger: 3-suit 27-card deck makes flushes very common.
# When 3+ community cards share a suit, tighten calling if we don't hold the flush.
FLUSH_DANGER_HIGH_EQUITY_BUMP = 0.06    # 3+ of one suit on board, we hold 0 of that suit
FLUSH_DANGER_MODERATE_EQUITY_BUMP = 0.02  # 3+ of one suit on board, we hold 1 of that suit

# Paired-board danger: when board has a pair/trips, full houses and quads are common.
# Tighten if we don't connect with the paired rank ourselves.
PAIRED_BOARD_HIGH_EQUITY_BUMP = 0.05   # board paired/trips and we don't match that rank at all
PAIRED_BOARD_MODERATE_EQUITY_BUMP = 0.02  # board paired/trips but we match with 1 card

# Post-flop commitment control: avoid escalating marginal hands in bloated pots.
SOFT_COMMIT_BET = 40
HIGH_COMMIT_BET = 60
NEAR_ALLIN_BET = 80
SOFT_COMMIT_CALL_BUMP = 0.02
HIGH_COMMIT_CALL_BUMP = 0.05
NEAR_ALLIN_CALL_BUMP = 0.08
SOFT_COMMIT_POT_ODDS_BUMP = 0.01
HIGH_COMMIT_POT_ODDS_BUMP = 0.03
NEAR_ALLIN_POT_ODDS_BUMP = 0.05

# Raise-back: re-raise threshold when facing aggression in committed pots.
POSTFLOP_RAISEBACK_PREMIUM_EQUITY = 0.72

# Danger compounding: when both flush and paired-board danger are present.
COMPOUND_DANGER_TURN_BUMP = 0.03
COMPOUND_DANGER_RIVER_BUMP = 0.05

# River overcommit guard when facing a raise in large commitments.
RIVER_OVERCOMMIT_CALL_BUMP = 0.08

# Early-match phase (short; no artificial call-floor penalties — baseline from hand 1).
EARLY_PHASE_HAND_LIMIT = 50
EARLY_PREFLOP_CALL_BUMP = 0.0
EARLY_POSTFLOP_CALL_BUMP = 0.0
EARLY_RERAISE_PREMIUM_EQUITY = 0.92

# Opponent post-flop raise-density pressure (short rolling window from player state).
OPP_POSTFLOP_PRESSURE_MILD = 0.22
OPP_POSTFLOP_PRESSURE_HIGH = 0.35
OPP_POSTFLOP_PRESSURE_PREFLOP_BUMP = 0.02
OPP_POSTFLOP_PRESSURE_MILD_CALL_BUMP = 0.028
OPP_POSTFLOP_PRESSURE_HIGH_CALL_BUMP = 0.055
# Turn/river: extra tax vs barreling opponents (reduces call-call-fold chip bleed)
TURN_RIVER_VS_BARREL_BUMP = 0.04

# Expanded pressure signals from player-level rolling counters.
OPP_POSTFLOP_RERAISE_PRESSURE_MILD = 0.25
OPP_POSTFLOP_RERAISE_PRESSURE_HIGH = 0.45
OPP_HIGH_COMMIT_PRESSURE_MILD = 0.20
OPP_HIGH_COMMIT_PRESSURE_HIGH = 0.35

# Extra penalty when board danger and opponent aggression combine.
AGG_DANGER_THRESHOLD = 1.8
AGG_DANGER_EXTRA_BUMP = 0.03

# Multi-street aggression pressure and raise-war controls.
MULTISTREET_RAISE_PATTERN_BUMP = 0.03
SECOND_RAISE_STREET_CALL_BUMP = 0.05
SECOND_RAISE_STREET_PREMIUM_EQUITY = 0.90

# Medium-pot spiral guardrail (where repeated pressure folds accumulate losses).
MEDIUM_POT_MIN = 24
MEDIUM_POT_MAX = 75
MEDIUM_POT_SPIRAL_BUMP = 0.04
MEDIUM_POT_SPIRAL_EXTRA_BUMP = 0.02

# Match14/15: preflop raise-war stop and high-commit anti-volatility.
PREFLOP_RAISE_WAR_PREMIUM_EQUITY = 0.84
COMMIT50_BET = 50
COMMIT50_CALL_BUMP = 0.03
COMMIT50_POTODDS_BUMP = 0.02
COMMIT80_CALL_BUMP_EXTRA = 0.04
COMMIT80_POTODDS_BUMP_EXTRA = 0.03
HIGH_COMMIT_DANGER_EXTRA_BUMP = 0.03
INVESTED_THEN_PRESSURED_CALL_BUMP = 0.03

# Dynamic threshold adjustments (tight = stronger ranges → higher bar; loose → lower bar)
TIGHT_OPPONENT_STRONG_ADJ = 0.08
TIGHT_OPPONENT_MEDIUM_ADJ = 0.05
LOOSE_OPPONENT_STRONG_ADJ = -0.05
LOOSE_OPPONENT_MEDIUM_ADJ = -0.05

# Preflop open-raise sizing (multiples of BB=2) — short deck: charge draws harder
PREFLOP_OPEN_RAISE_MULTIPLIER = 5.0
PREFLOP_3BET_MULTIPLIER = 3.75

# Facing a 3-bet: tighter ranges to avoid bloated pots with marginal hands
PREFLOP_3BET_CALL_MIN_EQUITY = 0.58  # min equity to call a real preflop raise / 3-bet line
PREFLOP_4BET_MIN_EQUITY = 0.80       # need 80%+ equity to 4-bet (very premium only)

# Large preflop jams: use pot odds vs ``preflop_equity_vs_shove`` (top-15% range MC), not linear penalties.
PREFLOP_SHOVE_POT_ODDS_MIN_BET = 28  # opp_bet >= this: compare shove-range equity to pot odds
PREFLOP_SHOVE_POT_ODDS_MARGIN = 0.06  # require equity clearly above pot odds vs shove (was 0.02)
# Extra discount when opponent puts in a large preflop open (e.g. SB → 8).
PREFLOP_LARGE_OPEN_EXTRA = 0.03
PREFLOP_LARGE_OPEN_BET = 8
# Extra floors vs large opens (on top of PREFLOP_3BET_CALL_MIN_EQUITY bumps).
PREFLOP_LARGE_OPEN_CALL_FLOOR_BUMP = 0.025
PREFLOP_LARGE_OPEN_DEFEND_FLOOR_BUMP = 0.03

# Extra penalty when BB facing a large SB open-raise (we didn't raise, they did).
PREFLOP_BB_OPEN_RAISE_THRESHOLD = 10  # opp_bet >= this
PREFLOP_BB_OPEN_RAISE_PENALTY = 0.03

# Re-raise threshold: when facing a bet, re-raise for value if equity >= this
RERAISE_EQUITY_THRESHOLD = 0.78

# Facing a small probe (thin bet vs pot): still value-raise with trips+ even when
# commit_band / straight_dominated / RERAISE_EQUITY_THRESHOLD would force a flat call.
VALUE_RERAISE_VS_PROBE_MAX_BET_FRAC = 0.22
VALUE_RERAISE_VS_PROBE_MIN_EQUITY = 0.52  # MC equity can sit low with chops / dominated straight
VALUE_RERAISE_VS_PROBE_MAX_RANK_CLASS = 6  # trips or better (same as TRIPS_OR_BETTER_RANK_CLASS)

# Value betting: when checked to, bet with medium-strong hands instead of checking
VALUE_BET_EQUITY = 0.72

# Continuation bet (c-bet): bet the flop when we raised preflop, even without a strong hand
CBET_EQUITY_THRESHOLD = 0.45   # c-bet with 45%+ equity on the flop
CBET_FRAC = 0.55               # c-bet size: 55% of pot
CBET_GIVE_UP_EQUITY = 0.30     # don't c-bet with very weak hands (below 30%)

# Preflop escalation cap: stop re-raising if already committed this many chips
PREFLOP_RERAISE_CAP = 10
PREFLOP_PREMIUM_EQUITY = 0.75  # only keep re-raising with top equity

# Hand strength floor: never fold trips or better post-flop (treys rank_class <= 6)
TRIPS_OR_BETTER_RANK_CLASS = 6
# Soft exception: on extremely dangerous river runouts with very heavy pressure,
# allow trips to fold when equity estimate is very low.
RIVER_TRIPS_SOFT_FOLD_MAX_EQUITY = 0.52
RIVER_TRIPS_SOFT_FOLD_MIN_BET_FRAC = 0.75
RIVER_TRIPS_SOFT_FOLD_AGG_MIN = 1.8

# Maximum total bump that can accumulate on call_floor above its base value.
# Must be large enough that nut-flush vs non-nut + strong discard signals can fold
# marginal "50% MC" hands (was 0.15 and erased most of the stacked penalties).
MAX_CALL_FLOOR_BUMP = 0.34

# Pot-odds sanity: when continue_cost is tiny relative to pot, always call
# if we have ANY reasonable hand. Prevents invest-then-fold for trivial amounts.
TINY_BET_POT_FRAC = 0.15       # bet < 15% of pot = trivially small
TINY_BET_MAX_EQUITY_REQ = 0.35  # need only 35% equity to call a tiny bet

# Min-raise stabs (e.g. 2 chips): pot-fraction rule can miss these on small pots.
# Postflop only — don't fold decent equity to a trivial price.
MICRO_BET_MAX_CONTINUE = 6
MICRO_BET_MAX_EQUITY_REQ = 0.40  # "decent" hand vs a 2-chip stab

# Pot-odds override: on flop/turn, call when equity beats pot odds even if
# below the 55% marginal floor (fixes folding pair+draw, flush draws with good odds).
POT_ODDS_OVERRIDE_MARGIN = 0.05   # equity must be at least pot_odds + this
POT_ODDS_OVERRIDE_MIN_EQUITY = 0.35  # don't call with very weak hands

# Nut-awareness: in this 27-card 3-suit deck, flushes/full houses are common.
# Suppress raising with non-nut hands on dangerous boards.
# rank_class: 1=SF, 2=quads, 3=FH, 4=flush, 5=straight, 6=trips, 7=two pair, 8=pair, 9=high
NUT_RAISE_MIN_ON_FLUSH_BOARD = 4     # need flush (4) or better to raise on 3+ suited board
NUT_RAISE_MIN_ON_PAIRED_BOARD = 4    # need flush (4) or better to raise on paired board
NUT_VBET_MIN_ON_FLUSH_BOARD = 5      # need straight (5) or better to value-bet on flush board
NUT_VBET_MIN_ON_PAIRED_BOARD = 5     # need straight (5) or better to value-bet on paired board

# Hand-class pot-size gates (pair/two-pair caps removed — use pot odds / MDF).
TRIPS_MAX_COMMIT = 80        # trips (rank_class 6): cautious above 80 on scary boards
HAND_CLASS_COMMIT_BUMP = 0.08  # extra equity required when exceeding commit gate

# Non-nut flush penalty: when we have a flush but our highest flush card
# is low (rank < 7, i.e., below 9), we have a weak flush that loses to
# better flushes.  Suppress raising and tighten calling.
NON_NUT_FLUSH_CALL_BUMP = 0.06
# Opponent discarded a pair → likely kept trips/two pair
OPP_DISCARDED_PAIR_CALL_BUMP = 0.03
# rank_index 0–7 = 2–9 high; only rank 8 (A) counts as nut in this 9-rank/suit deck.
NON_NUT_FLUSH_RANK_THRESHOLD = 7
# Extra call tightness: our non-nut flush vs opp flush signal / high kept flush
NON_NUT_FLUSH_VS_OPP_NUT_BUMP = 0.20
# Stack on top of base bumps when texture + discard line up (capped by MAX_CALL_FLOOR_BUMP)
NON_NUT_FLUSH_COMPOUND_STACK_BUMP = 0.12
OPP_KEPT_HIGH_FLUSH_SIG_BUMP = 0.06

# Straight-dominated: when we hold the low end of a straight on a
# board where higher straights are possible, suppress raising.
STRAIGHT_DOMINATED_CALL_BUMP = 0.04

# Bluff parameters (mixed with value lines; sizing uses _raise_frac_value)
BASE_BLUFF_FREQ = 0.15
MAX_BLUFF_FREQ = 0.35
MIN_HANDS_FOR_ADAPT = 50    # hands before trusting opponent model (faster convergence)

# Position bonus: being in position (acting last) is an advantage
IP_EQUITY_BONUS = 0.03      # small bonus when in position post-flop

# Opponent checked: they showed weakness – lower the bar to value bet (we’re more likely to have the best hand)
OPP_CHECK_STRONG_EQUITY_DISCOUNT = 0.05   # require 5% less equity to value bet when opp checked

# Calling thresholds when facing aggression
AGG_CALL_DISCOUNT = 0.05    # widen call range vs hyper-aggressive opponents

# Bet size → hand strength inference (large bet = likely stronger hand)
BET_FRAC_LARGE = 0.80      # bet ≥ 80% of pot = big bet
BET_FRAC_MEDIUM = 0.50     # bet ≥ 50% of pot = medium bet
EQUITY_DISCOUNT_LARGE = 0.05   # discount when facing big bet
EQUITY_DISCOUNT_MEDIUM = 0.02  # discount when facing medium bet
# Current bet vs history: unusually large = likely strong
BET_VS_HISTORY_THRESHOLD = 1.5   # current bet >= 1.5x their typical -> extra discount
EQUITY_DISCOUNT_VS_HISTORY = 0.03  # extra discount when bet is unusually large

# Table strategy mode: "off" | "simple" | "conf"
# - off: heuristic only, table never used
# - simple: fixed 30% blend when table has entry
# - conf: confidence-weighted blend + EV safety check
TABLE_MODE = "off"
TABLE_BASE_ALPHA = 0.5
TABLE_VISIT_THRESHOLD = 200
TABLE_EV_EPSILON = 1.0


# ---------------------------------------------------------------------------
# Explicit chip-EV helpers (fold / call / raise)
# ---------------------------------------------------------------------------

def ev_fold() -> float:
    """
    Chip-EV of folding, used as the baseline.

    We measure EV relative to the choice of folding *now*, so the fold EV
    is defined as 0 and other actions are interpreted as gains/losses
    against this baseline.
    """
    return 0.0


def ev_call(equity: float, pot: int, continue_cost: int) -> float:
    """
    Approximate chip-EV for calling a bet.

    Parameters
    ----------
    equity : float in [0, 1]
        Probability that our hand wins at showdown given current information.
    pot : int
        Current pot size *before* we call, including all chips already
        invested by both players on this street so far.
    continue_cost : int
        Chips we must invest to call (opp_bet - my_bet).  This is the
        only additional risk we take relative to folding.

    Model
    -----
    - If we call and win, we capture the entire final pot.
    - If we call and lose, we lose the additional `continue_cost`.
    - We ignore future betting for this local decision and treat the
      post-call state as going directly to showdown.

    EV_call = equity * (pot + continue_cost) - (1 - equity) * continue_cost
    """
    if continue_cost <= 0:
        # Nothing to call; EV relative to fold is effectively 0 in this
        # simplified local model.
        return 0.0

    final_pot = pot + continue_cost
    win_term = equity * final_pot
    lose_term = (1.0 - equity) * continue_cost
    return win_term - lose_term


def ev_raise(
    equity: float,
    pot: int,
    my_bet: int,
    raise_to: int,
    opp_fold_prob: float,
    opp_call_prob: float,
    opp_raise_prob: float = 0.0,
) -> float:
    """
    Approximate chip-EV for raising to a given size.

    Parameters
    ----------
    equity : float in [0, 1]
        Probability our hand wins at showdown if called.
    pot : int
        Current pot size before we raise (my_bet + opp_bet).
    my_bet : int
        Our current committed chips this street before raising.
    raise_to : int
        Target bet size we raise to (absolute bet, consistent with engine).
    opp_fold_prob : float
        Probability opponent folds facing this raise.
    opp_call_prob : float
        Probability opponent calls our raise.
    opp_raise_prob : float, optional
        Probability opponent re-raises.  For now we approximate this
        branch by treating it like a call with an effectively larger pot,
        since `decide_action` does not model future re-raises explicitly.

    Model
    -----
    Let delta = raise_to - my_bet be our additional investment now.

    - If opponent folds:
        We win the existing pot plus our additional chips that went in
        with the raise (which we immediately get back as part of the pot).
        EV_fold_branch = pot + delta

    - If opponent calls:
        Final pot ≈ pot + 2 * delta (they match our additional delta).
        We win this pot with probability `equity` and lose our extra
        investment `delta` with probability (1 - equity).
        EV_call_branch = equity * (pot + 2 * delta) - (1 - equity) * delta

    - If opponent re-raises:
        We approximate this as another "call" branch with the same EV
        expression, since we currently do not model re-raise trees.

    Total:
        EV_raise = p_fold * EV_fold_branch
                 + (p_call + p_raise) * EV_call_branch
    """
    if raise_to <= my_bet:
        # Not a meaningful raise; treat as call-like in this abstraction.
        return ev_call(equity, pot, continue_cost=0)

    delta = raise_to - my_bet
    p_fold = max(0.0, min(1.0, opp_fold_prob))
    p_call = max(0.0, min(1.0 - p_fold, opp_call_prob))
    p_raise = max(0.0, min(1.0 - p_fold - p_call, opp_raise_prob))

    # Normalize slightly if numeric issues push sum over 1.0
    total_p = p_fold + p_call + p_raise
    if total_p > 1.0 and total_p > 0.0:
        p_fold /= total_p
        p_call /= total_p
        p_raise /= total_p

    ev_fold_branch = pot + delta
    final_pot_if_called = pot + 2 * delta
    ev_call_branch = equity * final_pot_if_called - (1.0 - equity) * delta

    return p_fold * ev_fold_branch + (p_call + p_raise) * ev_call_branch


def _approx_ev_for_action(
    action: tuple[int, int, int, int],
    equity: float,
    pot: int,
    my_bet: int,
    opp_bet: int,
    opp_fold_rate: float,
    min_raise: int,
    max_raise: int,
) -> float:
    """Approximate EV of an engine action tuple for safety checks."""
    atype, amount, _, _ = action
    continue_cost = opp_bet - my_bet
    if atype == 0:  # FOLD
        return ev_fold()
    if atype in (2, 3):  # CHECK or CALL
        return ev_call(equity, pot, continue_cost if atype == 3 else 0)
    if atype == 1:  # RAISE
        raise_to = my_bet + amount
        p_fold = opp_fold_rate
        p_call = max(0.0, 1.0 - p_fold)
        return ev_raise(equity, pot, my_bet, raise_to, p_fold, p_call)
    return 0.0


def blind_position_from_obs(observation: dict) -> int:
    """
    Infer blind position from observation fields.
    Returns 0 for SB, 1 for BB.
    On street 0, SB has my_bet==1 and BB has my_bet==2 at the start.
    We check: if acting_agent is us and street==0 and my_bet < opp_bet → we're SB (0).
    Fallback: if my_bet <= opp_bet on street 0 → SB (0), else BB (1).
    """
    my_bet = observation.get("my_bet", 0)
    opp_bet = observation.get("opp_bet", 0)
    # At the very start of a hand, SB posts 1, BB posts 2.
    # Even after calls/raises the SB started lower, but the initial
    # observation is the most reliable signal.  We use a simple heuristic:
    # if our initial bet is smaller, we're SB.
    # This heuristic works perfectly on street 0 before any action.
    # On later streets bets are equal after each street, so we fall back to 0.
    if my_bet == 1 and opp_bet == 2:
        return 0  # SB
    if my_bet == 2 and opp_bet == 1:
        return 1  # BB
    return 0  # default fallback


def _flush_danger(observation: dict) -> int:
    """
    Detect flush danger from board texture in the 3-suit 27-card deck.

    Card encoding: suit = card_int // 9  (0=d, 1=h, 2=s).

    Returns:
        0 — safe (fewer than 3 of any suit on board, or we hold 2 of that suit)
        1 — moderate (3+ of one suit on board, we hold exactly 1 of that suit)
        2 — high (3+ of one suit on board, we hold 0 of that suit)
    """
    community = [c for c in observation.get("community_cards", []) if c != -1]
    if len(community) < 3:
        return 0

    my_cards = [c for c in observation.get("my_cards", []) if c != -1][:2]

    board_suits: dict[int, int] = {}
    for c in community:
        s = c // 9
        board_suits[s] = board_suits.get(s, 0) + 1

    danger_suit = None
    for s, cnt in board_suits.items():
        if cnt >= 3:
            danger_suit = s
            break

    if danger_suit is None:
        return 0

    my_suit_count = sum(1 for c in my_cards if c // 9 == danger_suit)
    if my_suit_count >= 2:
        return 0  # we likely have the flush
    if my_suit_count == 1:
        return 1
    return 2


def _paired_board_danger(observation: dict) -> int:
    """
    Detect paired-board danger (full house / quads threat).

    Card encoding: rank = card_int % 9.

    Returns:
        0 — safe (board has no pair, or we hold 2+ cards matching the paired rank)
        1 — moderate (board paired and we match the paired rank with exactly 1 card)
        2 — high (board paired/trips and we don't match the paired rank at all)
    """
    community = [c for c in observation.get("community_cards", []) if c != -1]
    if len(community) < 3:
        return 0

    my_cards = [c for c in observation.get("my_cards", []) if c != -1][:2]

    board_ranks: dict[int, int] = {}
    for c in community:
        r = c % 9
        board_ranks[r] = board_ranks.get(r, 0) + 1

    paired_rank = None
    for r, cnt in board_ranks.items():
        if cnt >= 2:
            paired_rank = r
            break

    if paired_rank is None:
        return 0

    my_match = sum(1 for c in my_cards if c % 9 == paired_rank)
    if my_match >= 2:
        return 0  # we have trips/quads with the board pair
    if my_match == 1:
        return 1  # we have a full house draw / trips
    return 2


def _opp_flush_signal(observation: dict) -> int:
    """
    Infer whether opponent likely kept suited cards from their revealed discards.

    In this variant, discards are revealed.  If the board has 3+ of one suit
    and the opponent discarded 0 cards of that suit, they likely kept 2 suited
    cards → probable flush.

    Returns:
        0 — no signal (board not flushy, or insufficient info)
        1 — possible (opponent discarded 1 of the danger suit)
        2 — likely (opponent discarded 0 of the danger suit → kept all)
    """
    community = [c for c in observation.get("community_cards", []) if c != -1]
    if len(community) < 3:
        return 0

    board_suits: dict[int, int] = {}
    for c in community:
        s = c // 9
        board_suits[s] = board_suits.get(s, 0) + 1

    danger_suit = None
    for s, cnt in board_suits.items():
        if cnt >= 3:
            danger_suit = s
            break

    if danger_suit is None:
        return 0

    opp_discards = [c for c in observation.get("opp_discarded_cards", [-1, -1, -1]) if c != -1]
    if len(opp_discards) < 3:
        return 0

    danger_in_discards = sum(1 for c in opp_discards if c // 9 == danger_suit)
    if danger_in_discards == 0:
        return 2  # kept all danger-suit cards → likely flush
    if danger_in_discards == 1:
        return 1  # discarded 1, might still have 1+ of danger suit
    return 0


def _is_overpair_on_weak_board(observation: dict, hand_rank_class: int | None) -> bool:
    """
    Return True if we have an overpair (pair higher than all board cards)
    on a weak board (no flush draw, no paired board).

    Overpairs on weak boards are strong and should not be folded to normal bets.
    """
    if hand_rank_class is None or hand_rank_class != 8:
        return False

    community = [c for c in observation.get("community_cards", []) if c != -1]
    my_cards = [c for c in observation.get("my_cards", []) if c != -1][:2]
    if len(community) < 3 or len(my_cards) < 2:
        return False

    pair_rank = my_cards[0] % 9
    if my_cards[1] % 9 != pair_rank:
        return False  # not a pair

    board_ranks = {c % 9 for c in community}
    if pair_rank <= max(board_ranks):
        return False  # not an overpair

    if _flush_danger(observation) != 0 or _paired_board_danger(observation) != 0:
        return False  # board not weak

    return True


def _is_non_nut_flush(observation: dict, hand_rank_class: int | None) -> bool:
    """
    Return True if we have a flush (rank_class==4) but our highest card
    in the flush suit is low — meaning better flushes are likely to exist.

    In this 3-suit deck with 9 cards per suit, only an Ace-high flush (rank
    index 8) is treated as nut; ``max(rank) <= NON_NUT_FLUSH_RANK_THRESHOLD``
    marks 2–9 high flushes as non-nut / vulnerable.
    """
    if hand_rank_class is None or hand_rank_class != 4:
        return False

    community = [c for c in observation.get("community_cards", []) if c != -1]
    my_cards = [c for c in observation.get("my_cards", []) if c != -1][:2]
    if len(community) < 3:
        return False

    board_suits: dict[int, int] = {}
    for c in community:
        s = c // 9
        board_suits[s] = board_suits.get(s, 0) + 1

    flush_suit = None
    for s, cnt in board_suits.items():
        if cnt >= 3:
            flush_suit = s
            break

    if flush_suit is None:
        return False

    my_flush_ranks = [c % 9 for c in my_cards if c // 9 == flush_suit]
    if not my_flush_ranks:
        return False

    return max(my_flush_ranks) <= NON_NUT_FLUSH_RANK_THRESHOLD


def _is_straight_dominated(observation: dict, hand_rank_class: int | None) -> bool:
    """
    Return True if we have a straight (rank_class==5) but we hold
    the low end, meaning higher straights are possible on this board.

    Checks: if the board contains ranks above our highest hole card
    that could extend the straight window, we're on the low end.
    """
    if hand_rank_class is None or hand_rank_class != 5:
        return False

    community = [c for c in observation.get("community_cards", []) if c != -1]
    my_cards = [c for c in observation.get("my_cards", []) if c != -1][:2]
    if len(community) < 3:
        return False

    my_ranks = [c % 9 for c in my_cards]
    board_ranks = [c % 9 for c in community]
    all_ranks = set(my_ranks + board_ranks)
    my_max = max(my_ranks)

    # Find which straight window(s) we complete
    straight_windows = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
    ]
    for window in straight_windows:
        if set(window).issubset(all_ranks):
            # We make this straight. Check if we hold the low end:
            # if our highest card is NOT the top of this window AND
            # a higher straight window is possible on the board.
            top = max(window)
            if my_max < top:
                # We hold the bottom — a higher straight might exist
                # Check if board could support a higher window
                for higher in straight_windows:
                    if max(higher) > top:
                        board_in_higher = sum(1 for r in board_ranks if r in higher)
                        if board_in_higher >= 3:
                            return True
    return False


def _eligible_value_reraise_vs_small_probe(
    *,
    street: int,
    continue_cost: int,
    pot: int,
    hand_rank_class: int | None,
    adj_equity: float,
) -> bool:
    """True if we should value-raise vs a small bet with a strong made hand."""
    if street < 1 or continue_cost <= 0 or pot <= 0:
        return False
    if hand_rank_class is None or hand_rank_class > VALUE_RERAISE_VS_PROBE_MAX_RANK_CLASS:
        return False
    bet_frac = continue_cost / pot
    if bet_frac > VALUE_RERAISE_VS_PROBE_MAX_BET_FRAC:
        return False
    if adj_equity < VALUE_RERAISE_VS_PROBE_MIN_EQUITY:
        return False
    return True


def _eligible_semi_bluff_check_raise(
    *,
    street: int,
    continue_cost: int,
    adj_equity: float,
    pot_odds: float,
    flush_danger: int,
    pair_danger: int,
    hand_rank_class: int | None,
    valid: list[bool],
    commit_band: int,
) -> bool:
    """Monster-draw spots: re-raise on flop/turn with ~marginal equity on wet boards."""
    if not valid[1] or street not in (1, 2) or continue_cost <= 0:
        return False
    if commit_band >= 3:
        return False
    if not (0.40 <= adj_equity <= 0.56):
        return False
    if adj_equity + 0.02 < pot_odds:
        return False
    if flush_danger + pair_danger < 1:
        return False
    if hand_rank_class is not None and hand_rank_class <= 5:
        return False
    return True


def decide_action(
    equity: float,
    observation: dict,
    opp_model: OpponentModel,
    info: dict | None = None,
    hand_rank_class: int | None = None,
    strategy_profile: dict[str, float] | None = None,
) -> tuple[int, int, int, int]:
    """
    Choose (action_type, raise_amount, keep1, keep2) for a betting decision.

    action_type values: FOLD=0, RAISE=1, CHECK=2, CALL=3

    hand_rank_class: optional treys rank class (1=SF..9=high card).
        When provided on the river, used to enforce a strength floor.
    strategy_profile: optional delta overrides from the meta-controller bandit.
    """
    adapted = opp_model.hands_seen >= MIN_HANDS_FOR_ADAPT
    _react = opponent_reactive_adjustments(opp_model, adapted=adapted) if adapted else {}
    _sp = _merge_profile_deltas(strategy_profile or {}, _react)
    # Profile-adjusted constants (local to this call)
    _bluff_freq_base = max(0.0, BASE_BLUFF_FREQ + _sp.get("BASE_BLUFF_FREQ", 0.0))
    _bluff_freq_cap = max(0.0, MAX_BLUFF_FREQ + _sp.get("MAX_BLUFF_FREQ", 0.0))
    _semi_bluff_min = max(0.0, SEMI_BLUFF_EQUITY_MIN + _sp.get("SEMI_BLUFF_EQUITY_MIN", 0.0))
    _semi_bluff_max = min(1.0, SEMI_BLUFF_EQUITY_MAX + _sp.get("SEMI_BLUFF_EQUITY_MAX", 0.0))
    _vbet_eq = max(0.0, VALUE_BET_EQUITY + _sp.get("VALUE_BET_EQUITY", 0.0))
    # Passive postflop callers: bet thin for value — they'll look us up with weak hands.
    if adapted and opp_model.is_calling_station_postflop():
        _vbet_eq = max(0.45, _vbet_eq - 0.15)
    _cbet_eq = max(0.0, CBET_EQUITY_THRESHOLD + _sp.get("CBET_EQUITY_THRESHOLD", 0.0))
    _open_raise_mult = max(2.0, PREFLOP_OPEN_RAISE_MULTIPLIER + _sp.get("PREFLOP_OPEN_RAISE_MULTIPLIER", 0.0))
    _rf_scale = max(0.75, min(1.35, 1.0 + float(_sp.get("raise_fraction_bonus", 0.0))))
    valid = observation["valid_actions"]
    street = observation["street"]
    my_bet = observation["my_bet"]
    opp_bet = observation["opp_bet"]
    min_raise = observation["min_raise"]
    max_raise = observation["max_raise"]

    # Fold-to-win: if we can fold every remaining hand and still win, do it
    _info = info or {}
    acting_agent = observation.get("acting_agent", 0)
    # Support both bankroll_0/1 (local match) and team_0_bankroll/team_1_bankroll (tournament)
    key_my = "bankroll_0" if acting_agent == 0 else "bankroll_1"
    key_opp = "bankroll_1" if acting_agent == 0 else "bankroll_0"
    alt_my = "team_0_bankroll" if acting_agent == 0 else "team_1_bankroll"
    alt_opp = "team_1_bankroll" if acting_agent == 0 else "team_0_bankroll"
    my_bankroll = float(_info.get(key_my, _info.get(alt_my, 0)))
    opp_bankroll = float(_info.get(key_opp, _info.get(alt_opp, 0)))
    hand_number = int(_info.get("hand_number", 0))
    hands_left = 1000 - hand_number
    early_phase = hand_number < EARLY_PHASE_HAND_LIMIT
    my_raises_this_street = int(_info.get("my_raises_this_street", 0))
    my_raises_this_hand = int(_info.get("my_raises_this_hand", 0))
    opp_postflop_raise_density = float(_info.get("opp_postflop_raise_density", 0.0))
    opp_postflop_reraise_density = float(_info.get("opp_postflop_reraise_density", 0.0))
    opp_high_commit_pressure_density = float(_info.get("opp_high_commit_pressure_density", 0.0))

    opp_postflop_pressure = 0
    if opp_postflop_raise_density >= OPP_POSTFLOP_PRESSURE_HIGH:
        opp_postflop_pressure = 2
    elif opp_postflop_raise_density >= OPP_POSTFLOP_PRESSURE_MILD:
        opp_postflop_pressure = 1
    opp_reraise_pressure = 0
    if opp_postflop_reraise_density >= OPP_POSTFLOP_RERAISE_PRESSURE_HIGH:
        opp_reraise_pressure = 2
    elif opp_postflop_reraise_density >= OPP_POSTFLOP_RERAISE_PRESSURE_MILD:
        opp_reraise_pressure = 1
    opp_high_commit_pressure = 0
    if opp_high_commit_pressure_density >= OPP_HIGH_COMMIT_PRESSURE_HIGH:
        opp_high_commit_pressure = 2
    elif opp_high_commit_pressure_density >= OPP_HIGH_COMMIT_PRESSURE_MILD:
        opp_high_commit_pressure = 1

    # Need a comfortable lead so folding every remaining hand still wins.
    # vs frequent preflop jams, variance is higher — require a larger cushion (2.5×).

    min_lead_to_fold = 1.5 * hands_left
    if valid[0] and hands_left > 0 and my_bankroll > min_lead_to_fold:
        return (0, 0, 0, 0)  # FOLD

    # Cost to continue (computed early for preflop cap check)
    continue_cost = opp_bet - my_bet

    # Preflop: large jams — fold if MC equity vs ~top-15% shove range is below pot odds (no linear chip penalty).
    if street == 0 and continue_cost > 0 and opp_bet >= PREFLOP_SHOVE_POT_ODDS_MIN_BET:
        pot_pf = observation.get("pot_size", my_bet + opp_bet)
        denom = pot_pf + 2 * continue_cost
        if denom > 0:
            po_need = continue_cost / denom
            eq_shove = float(_info.get("preflop_equity_vs_shove", -1.0))
            if eq_shove >= 0 and eq_shove + 1e-9 < po_need + PREFLOP_SHOVE_POT_ODDS_MARGIN and valid[0]:
                return (0, 0, 0, 0)
    # Smaller raises: still discount huge opens (range strength), not per-chip scaling.
    if street == 0 and continue_cost > 0 and opp_bet >= PREFLOP_LARGE_OPEN_BET:
        equity -= PREFLOP_LARGE_OPEN_EXTRA
        blind_pos = observation.get("blind_position", blind_position_from_obs(observation))
        if (
            blind_pos == 1
            and my_raises_this_hand == 0
            and opp_bet >= PREFLOP_BB_OPEN_RAISE_THRESHOLD
        ):
            equity -= PREFLOP_BB_OPEN_RAISE_PENALTY

    # Preflop escalation cap: stop re-raising when already committed heavily
    # unless we have premium equity.  Just call to see a flop.
    if street == 0 and continue_cost > 0 and my_bet >= PREFLOP_RERAISE_CAP and equity < PREFLOP_PREMIUM_EQUITY:
        if valid[3]:
            return (3, 0, 0, 0)  # CALL – see a flop cheaply
        if valid[0]:
            return (0, 0, 0, 0)  # FOLD if can't call

    # Facing a significant preflop raise: tighten ranges to control pot size.
    # Triggers vs any raise >= 3x BB (opp_bet >= 6), whether it's an initial
    # open-raise we're facing from BB or a 3-bet after our open.
    if street == 0 and continue_cost > 0 and opp_bet >= 6:
        preflop_call_floor = PREFLOP_3BET_CALL_MIN_EQUITY
        preflop_4bet_floor = PREFLOP_4BET_MIN_EQUITY
        if early_phase:
            preflop_call_floor += EARLY_PREFLOP_CALL_BUMP
            preflop_4bet_floor += 0.03
        if opp_postflop_pressure >= 1:
            preflop_call_floor += OPP_POSTFLOP_PRESSURE_PREFLOP_BUMP
        if opp_postflop_pressure >= 2:
            preflop_4bet_floor += OPP_POSTFLOP_PRESSURE_PREFLOP_BUMP
        if opp_bet >= PREFLOP_LARGE_OPEN_BET:
            preflop_call_floor += PREFLOP_LARGE_OPEN_CALL_FLOOR_BUMP
            preflop_4bet_floor += 0.02
        if equity < preflop_call_floor:
            if valid[0]:
                return (0, 0, 0, 0)  # FOLD — not strong enough vs a real raise
        elif equity < preflop_4bet_floor:
            if valid[3]:
                return (3, 0, 0, 0)  # CALL — see a flop, don't escalate

    # Facing any real preflop raise (not blind completion): fold low-equity hands early.
    if street == 0 and continue_cost > 0 and opp_bet > 2:
        preflop_defend_floor = 0.55
        if early_phase:
            preflop_defend_floor += 0.02
        if opp_postflop_pressure >= 1:
            preflop_defend_floor += 0.02
        if opp_bet >= PREFLOP_LARGE_OPEN_BET:
            preflop_defend_floor += PREFLOP_LARGE_OPEN_DEFEND_FLOOR_BUMP
        if equity < preflop_defend_floor and valid[0]:
            return (0, 0, 0, 0)

    # Match14/15 guardrail: if a preflop raise-war is underway, stop inflation
    # unless equity is clearly premium.
    preflop_raise_war = (
        street == 0
        and continue_cost > 0
        and opp_bet >= 6
        and my_raises_this_street >= 1
    )
    if preflop_raise_war and equity < PREFLOP_RAISE_WAR_PREMIUM_EQUITY:
        if equity >= PREFLOP_3BET_CALL_MIN_EQUITY and valid[3]:
            return (3, 0, 0, 0)
        if valid[0]:
            return (0, 0, 0, 0)

    # Compute pot and blind position from available fields
    # (pot_size and blind_position may be stripped by Pydantic in API mode)
    pot = observation.get("pot_size", my_bet + opp_bet)
    blind_pos = observation.get("blind_position", blind_position_from_obs(observation))
    opp_last = (observation.get("opp_last_action") or "").upper()

    # Post-flop commitment bands (relative to 100-chip per-street cap)
    if street >= 1 and my_bet >= NEAR_ALLIN_BET:
        commit_band = 3
    elif street >= 1 and my_bet >= HIGH_COMMIT_BET:
        commit_band = 2
    elif street >= 1 and my_bet >= SOFT_COMMIT_BET:
        commit_band = 1
    else:
        commit_band = 0

    # Facing a post-flop raise in an already committed line:
    # lock out re-raise wars unless we have premium equity.
    facing_postflop_raise = street >= 1 and continue_cost > 0 and opp_last == "RAISE"
    raise_war_underway = facing_postflop_raise and my_raises_this_street >= 1
    second_raise_this_street = raise_war_underway
    lockout_reraise = (facing_postflop_raise and commit_band >= 1) or raise_war_underway

    # Position: who acts LAST has position advantage.
    #   Pre-flop (street 0):  SB acts first → BB (blind_pos=1) is in position.
    #   Post-flop (street≥1): BB acts first → SB (blind_pos=0) is in position.
    in_position = (street == 0 and blind_pos == 1) or (street >= 1 and blind_pos == 0)

    # Adjust equity for position
    adj_equity = equity + (IP_EQUITY_BONUS if in_position else 0.0)

    # Pot odds to call: equity needed = continue_cost / (pot + continue_cost)
    # (pot = chips in the middle before we match; matches engine / prob_agent convention)
    pot_odds = (
        continue_cost / (pot + continue_cost)
        if continue_cost > 0 and (pot + continue_cost) > 0
        else 0.0
    )

    # Opponent adaptation
    opp_fold_rate = opp_model.fold_rate(street) if adapted else 0.30
    opp_agg = opp_model.aggression(street) if adapted else 1.0
    
    # Use recent trends if available (weighted toward recent hands)
    bucket = opp_model.streets[street] if street is not None else opp_model.overall
    if adapted and bucket.recent_actions > 5:
        opp_fold_rate = opp_model.recent_fold_rate(street)
        opp_agg = opp_model.recent_aggression(street)
    # Postflop: blend in aggregate fold rate — per-street noise should not inflate bluff freq
    # when the villain stations (many preflop folds skew overall stats elsewhere).
    if adapted and street >= 1 and opp_model.postflop_actions_count() >= 12:
        opp_fold_rate = min(opp_fold_rate, opp_model.postflop_fold_rate())
    multi_street_raise_pressure = False
    if adapted:
        pressure_streets = 0
        for s in (1, 2, 3):
            sb = opp_model.streets[s]
            if sb.actions >= 8 and sb.raise_rate >= 0.45:
                pressure_streets += 1
        multi_street_raise_pressure = pressure_streets >= 2
    if opp_postflop_pressure >= 2 and street >= 1:
        multi_street_raise_pressure = True

    # Dynamic threshold adjustment based on opponent type
    strong_equity = BASE_STRONG_EQUITY + _sp.get("BASE_STRONG_EQUITY", 0.0)
    medium_equity = BASE_MEDIUM_EQUITY + _sp.get("BASE_MEDIUM_EQUITY", 0.0)

    if adapted:
        if opp_model.is_calling_station_postflop():
            # Calls pot odds postflop; value bet tighter, do not treat like a folder.
            strong_equity += LOOSE_OPPONENT_STRONG_ADJ
            medium_equity += LOOSE_OPPONENT_MEDIUM_ADJ
        elif opp_model.is_tight():
            # Tight opponent: lower thresholds (more aggressive)
            # They fold more, so we can bet/raise with weaker hands
            strong_equity += TIGHT_OPPONENT_STRONG_ADJ
            medium_equity += TIGHT_OPPONENT_MEDIUM_ADJ
        elif opp_model.is_loose():
            # Loose opponent: higher thresholds (tighter)
            # They call more, so we need stronger hands to value bet
            strong_equity += LOOSE_OPPONENT_STRONG_ADJ
            medium_equity += LOOSE_OPPONENT_MEDIUM_ADJ
    
    # Street-specific: stay tight on all streets (short deck – draws hit often)
    if street == 0:  # Pre-flop
        strong_equity += 0.02
        medium_equity += 0.02
    # Pre-flop when facing only the BB (or tiny call): don’t require 70% to call – we’d fold too much as SB
    if street == 0 and continue_cost > 0 and continue_cost <= 2 and pot <= 6:
        medium_equity = min(medium_equity, 0.52)  # call BB with 52%+ equity instead of 70%
    # River: when we can bet (no bet to call), value-bet two pair+ – we’re only behind trips/boats/AA
    if street == 3 and continue_cost == 0:
        strong_equity = min(strong_equity, 0.68)  # value bet with 68%+ on river (two pair usually 70%+)

    # Opponent checked: they showed weakness – we’re more likely to have the best hand, so value bet with slightly less equity
    if continue_cost == 0 and opp_last == "CHECK":
        strong_equity = max(0.50, strong_equity - OPP_CHECK_STRONG_EQUITY_DISCOUNT)

    # Against hyper-aggressive opponents, widen our calling range
    call_threshold_adj = 0.0
    if adapted and opp_agg > 2.0:
        call_threshold_adj = -AGG_CALL_DISCOUNT
    if street >= 1 and continue_cost > 0:
        if opp_postflop_pressure == 1:
            call_threshold_adj += OPP_POSTFLOP_PRESSURE_MILD_CALL_BUMP
        elif opp_postflop_pressure >= 2:
            call_threshold_adj += OPP_POSTFLOP_PRESSURE_HIGH_CALL_BUMP
        if opp_reraise_pressure == 1:
            call_threshold_adj += OPP_POSTFLOP_PRESSURE_MILD_CALL_BUMP
        elif opp_reraise_pressure >= 2:
            call_threshold_adj += OPP_POSTFLOP_PRESSURE_HIGH_CALL_BUMP
        if opp_high_commit_pressure == 1:
            call_threshold_adj += OPP_POSTFLOP_PRESSURE_MILD_CALL_BUMP
        elif opp_high_commit_pressure >= 2:
            call_threshold_adj += OPP_POSTFLOP_PRESSURE_HIGH_CALL_BUMP

    # Bet size → hand strength: large bets suggest stronger opponent hand.
    # Skip on preflop — every raise dwarfs the blind pot, making the fraction meaningless.
    if continue_cost > 0 and pot > 0 and street >= 1:
        bet_fraction = continue_cost / pot
        discount = 0.0
        if bet_fraction >= BET_FRAC_LARGE:
            discount = EQUITY_DISCOUNT_LARGE
        elif bet_fraction >= BET_FRAC_MEDIUM:
            discount = EQUITY_DISCOUNT_MEDIUM
        if discount > 0 and adapted:
            tendency = opp_model.bet_sizing_tendency(street)
            if tendency == "large":
                discount *= 0.5  # they often bet big, may overbet bluffs
            elif tendency == "small":
                discount *= 1.25  # big bet is unusual = likely strong
        # Current bet vs history: unusually large compared to their typical sizing
        bucket = opp_model.streets[min(street, 3)]
        if adapted and bucket.raises >= 5:
            avg_frac = bucket.avg_raise_fraction
            if avg_frac > 0.05 and bet_fraction >= avg_frac * BET_VS_HISTORY_THRESHOLD:
                discount += EQUITY_DISCOUNT_VS_HISTORY
        adj_equity -= discount

    # ---- Board texture: flush and paired-board danger in 3-suit deck ----
    flush_danger = _flush_danger(observation) if street >= 1 else 0
    pair_danger = _paired_board_danger(observation) if street >= 1 else 0
    opp_flush_sig = int(_info.get("opp_flush_signal", -1))
    if opp_flush_sig < 0 and street >= 1:
        opp_flush_sig = _opp_flush_signal(observation)
    elif opp_flush_sig < 0:
        opp_flush_sig = 0
    opp_discarded_pair = bool(_info.get("opp_discarded_pair", False))
    opp_likely_has_pair = bool(_info.get("opp_likely_has_pair", False))
    opp_straight_sig = int(_info.get("opp_straight_signal", 0))
    non_nut_flush = _is_non_nut_flush(observation, hand_rank_class) if street >= 1 else False
    straight_dominated = _is_straight_dominated(observation, hand_rank_class) if street >= 1 else False

    def _raise_frac_with_polarization(is_semi_bluff: bool) -> float:
        polar = (adj_equity > 0.95) or (street == 3 and is_semi_bluff)
        base = _raise_frac_value(
            flush_danger=flush_danger,
            pair_danger=pair_danger,
            street=street,
            is_polarized=polar,
        )
        return base * _rf_scale

    # ---- Optional baseline policy via StrategyTable (blended with heuristic) ----
    blended_action: tuple[int, int, int, int] | None = None
    use_table = False
    # Restrict table to high-impact regions: post-flop and medium/large pots only
    pot_band = 0 if pot < 10 else (1 if pot < 50 else 2)
    if street < 1 or pot_band < 1:
        use_table = False
    if use_table:
        state_key = _abstract_state_key(
            street=street,
            in_position=in_position,
            pot=pot,
            continue_cost=continue_cost,
            equity=adj_equity,
        )
        if TABLE_MODE == "simple":
            probs = strategy_table.get(state_key)
            conf = 1.0
        else:
            probs, conf = strategy_table.get_with_confidence(
                state_key, visit_threshold=TABLE_VISIT_THRESHOLD
            )
        if probs and conf > 0.0:
            p_table = TABLE_BASE_ALPHA * conf
            if random.random() < p_table:
                cand = _pick_action_from_table(
                    probs=probs,
                    valid_actions=valid,
                    pot=pot,
                    my_bet=my_bet,
                    min_raise=min_raise,
                    max_raise=max_raise,
                    equity=adj_equity,
                    opp_fold_rate=opp_fold_rate,
                )
                if cand is not None:
                    blended_action = cand

    def _ev_params():
        return {
            "equity": adj_equity,
            "pot": pot,
            "my_bet": my_bet,
            "opp_bet": opp_bet,
            "opp_fold_rate": opp_fold_rate,
            "min_raise": min_raise,
            "max_raise": max_raise,
        }

    def _choose_final(heuristic_action: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if blended_action is None:
            final_action = heuristic_action
        else:
            final_action = blended_action
            if TABLE_MODE == "conf":
                ev_h = _approx_ev_for_action(heuristic_action, **_ev_params())
                ev_t = _approx_ev_for_action(blended_action, **_ev_params())
                if ev_t < ev_h - TABLE_EV_EPSILON:
                    final_action = heuristic_action

        # Safety invariant: when continuing costs nothing, never fold by mistake.
        # (Intentional "fold-to-win" exits occur earlier, before _choose_final.)
        if final_action[0] == 0 and continue_cost <= 0:
            if valid[2]:
                return (2, 0, 0, 0)  # CHECK
            if valid[3]:
                return (3, 0, 0, 0)  # CALL fallback if engine marks it legal
            if valid[1]:
                # Last-resort legal action if check/call are unexpectedly unavailable.
                amount = max(0, min(min_raise, max_raise))
                if amount > 0:
                    return (1, amount, 0, 0)  # RAISE
        return final_action

    # ---- Hand-strength floor: never fold trips or better post-flop ----
    if (
        street >= 1
        and hand_rank_class is not None
        and hand_rank_class <= TRIPS_OR_BETTER_RANK_CLASS
        and continue_cost > 0
    ):
        # Safety valve: discard + texture say we're beaten — fold even with trips+ (see match11).
        dangerous_signal = (
            (
                opp_flush_sig >= 2
                and flush_danger >= 1
                and not non_nut_flush
                and hand_rank_class == 6
            )
            or (non_nut_flush and opp_flush_sig >= 1)
            or (
                opp_straight_sig >= 2
                and hand_rank_class is not None
                and 4 < hand_rank_class <= TRIPS_OR_BETTER_RANK_CLASS
            )
        )
        if (
            adapted
            and dangerous_signal
            and adj_equity <= 0.50
            and continue_cost > 0
            and valid[0]
        ):
            return (0, 0, 0, 0)

        # Soft exception (river only): allow fold in extreme danger + pressure + low equity.
        if street == 3:
            river_bet_frac = (continue_cost / pot) if pot > 0 else 0.0
            extreme_board_danger = (flush_danger == 2) or (flush_danger >= 1 and pair_danger >= 1)
            extreme_pressure = (
                river_bet_frac >= RIVER_TRIPS_SOFT_FOLD_MIN_BET_FRAC
                and (
                    opp_agg >= RIVER_TRIPS_SOFT_FOLD_AGG_MIN
                    or opp_postflop_pressure >= 2
                    or opp_reraise_pressure >= 2
                    or opp_high_commit_pressure >= 1
                )
            )
            if (
                extreme_board_danger
                and extreme_pressure
                and adj_equity <= RIVER_TRIPS_SOFT_FOLD_MAX_EQUITY
                and valid[0]
            ):
                return (0, 0, 0, 0)

        # Barrel exploit: on flop, raise trips+ for value vs hyper-aggressive lines.
        if (
            float(_sp.get("force_raise_strong_flop", 0)) > 0.5
            and street == 1
            and valid[1]
        ):
            frac = _raise_frac_with_polarization(False)
            raw = int(pot * frac)
            amount = max(min_raise, min(raw, max_raise))
            return (1, amount, 0, 0)

        # Default behavior: with trips+, at least call; raise if equity is high.
        trips_raise_floor = RERAISE_EQUITY_THRESHOLD
        if commit_band >= 2:
            trips_raise_floor = max(trips_raise_floor, POSTFLOP_RAISEBACK_PREMIUM_EQUITY)
        if lockout_reraise:
            trips_raise_floor = max(trips_raise_floor, POSTFLOP_RAISEBACK_PREMIUM_EQUITY)
        if early_phase:
            trips_raise_floor = max(trips_raise_floor, EARLY_RERAISE_PREMIUM_EQUITY)
        if adj_equity >= trips_raise_floor and valid[1]:
            frac = _raise_frac_with_polarization(False)
            raw = int(pot * frac)
            amount = max(min_raise, min(raw, max_raise))
            return (1, amount, 0, 0)
        if (
            valid[1]
            and _eligible_value_reraise_vs_small_probe(
                street=street,
                continue_cost=continue_cost,
                pot=pot,
                hand_rank_class=hand_rank_class,
                adj_equity=adj_equity,
            )
        ):
            frac = _raise_frac_with_polarization(False)
            raw = int(pot * frac)
            amount = max(min_raise, min(raw, max_raise))
            return (1, amount, 0, 0)
        if valid[3]:
            return (3, 0, 0, 0)

    # ---- Overpair on weak board: protect (never fold to normal bets) ----
    if (
        street >= 1
        and hand_rank_class is not None
        and _is_overpair_on_weak_board(observation, hand_rank_class)
        and continue_cost > 0
    ):
        # At least call; raise if equity supports it.
        overpair_raise_floor = RERAISE_EQUITY_THRESHOLD
        if adj_equity >= overpair_raise_floor and valid[1]:
            frac = _raise_frac_with_polarization(False)
            raw = int(pot * frac)
            amount = max(min_raise, min(raw, max_raise))
            return (1, amount, 0, 0)
        if valid[3]:
            return (3, 0, 0, 0)

    # ---- Semi-bluff check-raise: combo draws on wet boards (~45% equity) ----
    if (
        not (adapted and opp_model.is_calling_station_postflop())
        and _eligible_semi_bluff_check_raise(
            street=street,
            continue_cost=continue_cost,
            adj_equity=adj_equity,
            pot_odds=pot_odds,
            flush_danger=flush_danger,
            pair_danger=pair_danger,
            hand_rank_class=hand_rank_class,
            valid=valid,
            commit_band=commit_band,
        )
    ):
        frac = _raise_frac_with_polarization(True)
        raw = int(pot * frac)
        amount = max(min_raise, min(raw, max_raise))
        return _choose_final((1, amount, 0, 0))

    # ---- Heuristic decision tree (existing logic) ----

    # 1. STRONG HAND: value bet / raise (high bar in short deck)
    if adj_equity >= strong_equity:
        can_strong_raise = valid[1]
        if preflop_raise_war and street == 0 and adj_equity < PREFLOP_RAISE_WAR_PREMIUM_EQUITY:
            can_strong_raise = False
        if lockout_reraise and adj_equity < POSTFLOP_RAISEBACK_PREMIUM_EQUITY:
            can_strong_raise = False
        if second_raise_this_street and adj_equity < SECOND_RAISE_STREET_PREMIUM_EQUITY:
            can_strong_raise = False
        if early_phase and street >= 1 and continue_cost > 0 and adj_equity < EARLY_RERAISE_PREMIUM_EQUITY:
            can_strong_raise = False
        # Nut-awareness: on dangerous boards, only raise with nut-class hands.
        if can_strong_raise and street >= 1 and hand_rank_class is not None:
            if flush_danger >= 1 and hand_rank_class > NUT_RAISE_MIN_ON_FLUSH_BOARD:
                can_strong_raise = False
            if pair_danger >= 2 and hand_rank_class > NUT_RAISE_MIN_ON_PAIRED_BOARD:
                can_strong_raise = False
            if opp_flush_sig >= 2 and hand_rank_class > NUT_RAISE_MIN_ON_FLUSH_BOARD:
                can_strong_raise = False
            if non_nut_flush:
                can_strong_raise = False
            if straight_dominated:
                can_strong_raise = False
        # Value-raise small probes with trips+ even if commit / dominated straight blocked raise.
        if (
            continue_cost > 0
            and _eligible_value_reraise_vs_small_probe(
                street=street,
                continue_cost=continue_cost,
                pot=pot,
                hand_rank_class=hand_rank_class,
                adj_equity=adj_equity,
            )
        ):
            can_strong_raise = True
        if can_strong_raise:  # RAISE
            if street == 0:
                # Preflop: use BB-based sizing instead of pot-fraction
                if continue_cost <= 0 or my_bet <= 2:
                    # Open-raise or first raise: 3.5x BB
                    raw = int(2 * _open_raise_mult)
                else:
                    # 3-bet: 3x the incoming raise
                    raw = int(opp_bet * PREFLOP_3BET_MULTIPLIER)
            else:
                frac = _raise_frac_with_polarization(False)
                raw = int(pot * frac)
            amount = max(min_raise, min(raw, max_raise))
            action = (1, amount, 0, 0)
        elif valid[3]:  # CALL
            action = (3, 0, 0, 0)
        elif valid[2]:  # CHECK
            action = (2, 0, 0, 0)
        else:
            action = (0, 0, 0, 0)
        return _choose_final(action)

    # 2. MEDIUM HAND: value bet / re-raise / call depending on situation
    effective_medium = medium_equity + call_threshold_adj
    # Base: pure pot odds (no street floors or extra margin)
    min_equity_to_call = pot_odds if continue_cost > 0 else 0.0
    call_floor = pot_odds if continue_cost > 0 else 0.0
    # Commitment tightening: as invested chips rise, avoid thin continues.
    if commit_band == 1:
        call_floor += SOFT_COMMIT_CALL_BUMP
        min_equity_to_call += SOFT_COMMIT_POT_ODDS_BUMP
    elif commit_band == 2:
        call_floor += HIGH_COMMIT_CALL_BUMP
        min_equity_to_call += HIGH_COMMIT_POT_ODDS_BUMP
    elif commit_band >= 3:
        call_floor += NEAR_ALLIN_CALL_BUMP
        min_equity_to_call += NEAR_ALLIN_POT_ODDS_BUMP
    # Additional anti-volatility tightening used in match14/15:
    # clamp medium/high commitments even before maxed commit bands.
    if street >= 1 and continue_cost > 0 and my_bet >= COMMIT50_BET:
        call_floor += COMMIT50_CALL_BUMP
        min_equity_to_call += COMMIT50_POTODDS_BUMP
    if street >= 1 and continue_cost > 0 and my_bet >= NEAR_ALLIN_BET:
        call_floor += COMMIT80_CALL_BUMP_EXTRA
        min_equity_to_call += COMMIT80_POTODDS_BUMP_EXTRA
    # Early-match anti-variance: tighten post-flop continues in first 300 hands.
    if early_phase and street >= 1 and continue_cost > 0:
        call_floor += EARLY_POSTFLOP_CALL_BUMP
        min_equity_to_call += EARLY_POSTFLOP_CALL_BUMP
    if continue_cost > 0 and street >= 1:
        if opp_postflop_pressure == 1:
            call_floor += OPP_POSTFLOP_PRESSURE_MILD_CALL_BUMP
            min_equity_to_call += OPP_POSTFLOP_PRESSURE_MILD_CALL_BUMP
        elif opp_postflop_pressure >= 2:
            call_floor += OPP_POSTFLOP_PRESSURE_HIGH_CALL_BUMP
            min_equity_to_call += OPP_POSTFLOP_PRESSURE_HIGH_CALL_BUMP
    # Turn/river facing heavy postflop raise density: fold earlier, avoid big call-call-fold losses
    if continue_cost > 0 and street >= 2 and opp_postflop_pressure >= 2:
        call_floor += TURN_RIVER_VS_BARREL_BUMP
        min_equity_to_call += TURN_RIVER_VS_BARREL_BUMP
    if continue_cost > 0 and street >= 1 and multi_street_raise_pressure:
        call_floor += MULTISTREET_RAISE_PATTERN_BUMP
        min_equity_to_call += MULTISTREET_RAISE_PATTERN_BUMP
    if second_raise_this_street:
        call_floor += SECOND_RAISE_STREET_CALL_BUMP
        min_equity_to_call += SECOND_RAISE_STREET_CALL_BUMP
    if (
        continue_cost > 0
        and street >= 1
        and MEDIUM_POT_MIN <= pot <= MEDIUM_POT_MAX
        and opp_postflop_pressure >= 1
    ):
        call_floor += MEDIUM_POT_SPIRAL_BUMP
        min_equity_to_call += MEDIUM_POT_SPIRAL_BUMP
        if opp_postflop_pressure >= 2:
            call_floor += MEDIUM_POT_SPIRAL_EXTRA_BUMP
            min_equity_to_call += MEDIUM_POT_SPIRAL_EXTRA_BUMP
    # Large pot tightening: multi-street aggression makes marginal hands worse
    if street >= 2 and pot >= LARGE_POT_THRESHOLD:
        call_floor += LARGE_POT_EQUITY_BUMP
    if street >= 2 and pot >= HUGE_POT_THRESHOLD:
        call_floor += HUGE_POT_EQUITY_BUMP
    # Flush-heavy board: demand more equity when we don't hold the flush
    if flush_danger == 2:
        call_floor += FLUSH_DANGER_HIGH_EQUITY_BUMP
    elif flush_danger == 1:
        call_floor += FLUSH_DANGER_MODERATE_EQUITY_BUMP
    # Paired board: demand more equity when we don't connect with the pair
    if pair_danger == 2:
        call_floor += PAIRED_BOARD_HIGH_EQUITY_BUMP
    elif pair_danger == 1:
        call_floor += PAIRED_BOARD_MODERATE_EQUITY_BUMP
    # Compound board danger in committed pots (flush + paired board together).
    if flush_danger >= 1 and pair_danger >= 1 and street >= 2 and commit_band >= 1:
        if street == 3:
            call_floor += COMPOUND_DANGER_RIVER_BUMP
        else:
            call_floor += COMPOUND_DANGER_TURN_BUMP
    if street >= 1 and continue_cost > 0 and my_bet >= COMMIT50_BET and (flush_danger >= 1 or pair_danger >= 1):
        call_floor += HIGH_COMMIT_DANGER_EXTRA_BUMP
        min_equity_to_call += HIGH_COMMIT_DANGER_EXTRA_BUMP
    if flush_danger >= 1 and pair_danger >= 1 and street >= 1 and adapted and opp_agg >= AGG_DANGER_THRESHOLD:
        call_floor += AGG_DANGER_EXTRA_BUMP
    # Opponent discard signal: if opponent likely has a flush, tighten calling
    if opp_flush_sig >= 2 and street >= 1 and continue_cost > 0:
        call_floor += FLUSH_DANGER_HIGH_EQUITY_BUMP
        min_equity_to_call += FLUSH_DANGER_HIGH_EQUITY_BUMP
    elif opp_flush_sig >= 1 and street >= 1 and continue_cost > 0:
        call_floor += FLUSH_DANGER_MODERATE_EQUITY_BUMP
        min_equity_to_call += FLUSH_DANGER_MODERATE_EQUITY_BUMP
    if opp_discarded_pair and street >= 1 and continue_cost > 0:
        call_floor += OPP_DISCARDED_PAIR_CALL_BUMP
        min_equity_to_call += OPP_DISCARDED_PAIR_CALL_BUMP
    # Invest-then-pressure guard: after multiple raises in-hand, avoid thin continues.
    if street >= 1 and continue_cost > 0 and my_raises_this_hand >= 2:
        call_floor += INVESTED_THEN_PRESSURED_CALL_BUMP
        min_equity_to_call += INVESTED_THEN_PRESSURED_CALL_BUMP
    # River overcommit guard: when already deep and facing aggression, demand more.
    if street == 3 and continue_cost > 0 and commit_band >= 2:
        min_equity_to_call += RIVER_OVERCOMMIT_CALL_BUMP
    # Hand-class pot-size gates: trips on scary boards only (pair/two-pair use pot odds).
    if continue_cost > 0 and street >= 1 and hand_rank_class is not None:
        if hand_rank_class == 6 and my_bet >= TRIPS_MAX_COMMIT and (flush_danger >= 1 or pair_danger >= 1):
            call_floor += HAND_CLASS_COMMIT_BUMP
            min_equity_to_call += HAND_CLASS_COMMIT_BUMP
    # Non-nut flush / dominated straight: tighten calling
    if continue_cost > 0 and street >= 1:
        if non_nut_flush:
            call_floor += NON_NUT_FLUSH_CALL_BUMP
            min_equity_to_call += NON_NUT_FLUSH_CALL_BUMP
        opp_kept_high_flush = bool(_info.get("opp_kept_high_flush", False))
        if (
            non_nut_flush
            and (opp_flush_sig >= 2 or opp_kept_high_flush)
            and continue_cost > 0
        ):
            call_floor += NON_NUT_FLUSH_VS_OPP_NUT_BUMP
            min_equity_to_call += NON_NUT_FLUSH_VS_OPP_NUT_BUMP
        if (
            non_nut_flush
            and opp_flush_sig >= 2
            and continue_cost > 0
            and street >= 1
        ):
            call_floor += NON_NUT_FLUSH_COMPOUND_STACK_BUMP
            min_equity_to_call += NON_NUT_FLUSH_COMPOUND_STACK_BUMP
        if (
            opp_kept_high_flush
            and opp_flush_sig >= 1
            and continue_cost > 0
            and street >= 1
        ):
            call_floor += OPP_KEPT_HIGH_FLUSH_SIG_BUMP
            min_equity_to_call += OPP_KEPT_HIGH_FLUSH_SIG_BUMP
        if straight_dominated:
            call_floor += STRAIGHT_DOMINATED_CALL_BUMP
            min_equity_to_call += STRAIGHT_DOMINATED_CALL_BUMP
    # Reactive: vs barrel bots, demand more equity to call flop raises.
    if continue_cost > 0 and street == 1:
        fcb = float(_sp.get("flop_call_floor_bump", 0.0))
        if fcb != 0.0:
            call_floor += fcb
            min_equity_to_call += fcb

    # --- Cap total bumps to prevent over-tightening ---
    base_call_floor = pot_odds if continue_cost > 0 else 0.0
    if call_floor > base_call_floor + MAX_CALL_FLOOR_BUMP:
        call_floor = base_call_floor + MAX_CALL_FLOOR_BUMP

    # --- Pot-odds sanity: tiny bet into large pot = always call with any hand ---
    if continue_cost > 0 and pot > 0:
        bet_frac_vs_pot = continue_cost / pot
        if bet_frac_vs_pot < TINY_BET_POT_FRAC:
            call_floor = min(call_floor, TINY_BET_MAX_EQUITY_REQ)
            min_equity_to_call = min(min_equity_to_call, TINY_BET_MAX_EQUITY_REQ)
        elif street >= 1 and continue_cost <= MICRO_BET_MAX_CONTINUE:
            # Opponent clicked min (e.g. 2): don't demand turn/river 58%+ to continue
            call_floor = min(call_floor, MICRO_BET_MAX_EQUITY_REQ)
            min_equity_to_call = min(min_equity_to_call, MICRO_BET_MAX_EQUITY_REQ)

    if adj_equity >= effective_medium:
        action = (0, 0, 0, 0)
        if continue_cost > 0:
            # Re-raise for value when equity is near the top of the medium band.
            # On preflop, only re-raise vs blind completion (opp_bet <= 2);
            # vs real raises, just call — pot control.
            can_reraise = adj_equity >= RERAISE_EQUITY_THRESHOLD and valid[1]
            if street == 0 and opp_bet > 2:
                can_reraise = False
            if preflop_raise_war and adj_equity < PREFLOP_RAISE_WAR_PREMIUM_EQUITY:
                can_reraise = False
            if street >= 1 and commit_band >= 1:
                can_reraise = False
            if lockout_reraise and adj_equity < POSTFLOP_RAISEBACK_PREMIUM_EQUITY:
                can_reraise = False
            if second_raise_this_street and adj_equity < SECOND_RAISE_STREET_PREMIUM_EQUITY:
                can_reraise = False
            if street >= 1 and my_bet >= COMMIT50_BET and adj_equity < POSTFLOP_RAISEBACK_PREMIUM_EQUITY:
                can_reraise = False
            if street >= 1 and my_raises_this_hand >= 2 and adj_equity < POSTFLOP_RAISEBACK_PREMIUM_EQUITY:
                can_reraise = False
            if early_phase and street >= 1 and adj_equity < EARLY_RERAISE_PREMIUM_EQUITY:
                can_reraise = False
            # Nut-awareness: don't re-raise with non-nut hands on dangerous boards
            if can_reraise and street >= 1 and hand_rank_class is not None:
                if flush_danger >= 1 and hand_rank_class > NUT_RAISE_MIN_ON_FLUSH_BOARD:
                    can_reraise = False
                if pair_danger >= 2 and hand_rank_class > NUT_RAISE_MIN_ON_PAIRED_BOARD:
                    can_reraise = False
                if opp_flush_sig >= 2 and hand_rank_class > NUT_RAISE_MIN_ON_FLUSH_BOARD:
                    can_reraise = False
                if non_nut_flush:
                    can_reraise = False
                if straight_dominated:
                    can_reraise = False
            # Value-raise thin bets with trips+ (overrides commit_band / 78% reraise bar / dominated straight).
            if (
                continue_cost > 0
                and _eligible_value_reraise_vs_small_probe(
                    street=street,
                    continue_cost=continue_cost,
                    pot=pot,
                    hand_rank_class=hand_rank_class,
                    adj_equity=adj_equity,
                )
            ):
                can_reraise = True
            # Mixed strategy on the border of a value re-raise (raise ~20%, call ~80%).
            if (
                can_reraise
                and street >= 1
                and continue_cost > 0
                and (RERAISE_EQUITY_THRESHOLD - 0.04)
                <= adj_equity
                < RERAISE_EQUITY_THRESHOLD
                and random.random() > 0.20
            ):
                can_reraise = False
            if can_reraise:
                if street == 0:
                    raw = int(2 * _open_raise_mult)
                else:
                    raw = int(pot * _raise_frac_with_polarization(False))
                amount = max(min_raise, min(raw, max_raise))
                action = (1, amount, 0, 0)
            elif adj_equity >= max(min_equity_to_call, call_floor) and valid[3]:
                action = (3, 0, 0, 0)
        else:
            if street == 0 and valid[1]:
                # Preflop open-raise with medium hands too
                allow_open_raise = True
                if opp_postflop_pressure >= 1 and adj_equity < 0.74:
                    allow_open_raise = False
                if allow_open_raise:
                    raw = int(2 * _open_raise_mult)
                    amount = max(min_raise, min(raw, max_raise))
                    action = (1, amount, 0, 0)
                elif valid[2]:
                    action = (2, 0, 0, 0)
                elif valid[3]:
                    action = (3, 0, 0, 0)
            elif valid[1]:
                vbet_threshold = _vbet_eq
                if flush_danger == 2:
                    vbet_threshold += FLUSH_DANGER_HIGH_EQUITY_BUMP
                elif flush_danger == 1:
                    vbet_threshold += FLUSH_DANGER_MODERATE_EQUITY_BUMP
                if pair_danger == 2:
                    vbet_threshold += PAIRED_BOARD_HIGH_EQUITY_BUMP
                elif pair_danger == 1:
                    vbet_threshold += PAIRED_BOARD_MODERATE_EQUITY_BUMP
                if flush_danger >= 1 and pair_danger >= 1 and street >= 2 and commit_band >= 1:
                    if street == 3:
                        vbet_threshold += COMPOUND_DANGER_RIVER_BUMP
                    else:
                        vbet_threshold += COMPOUND_DANGER_TURN_BUMP
                if flush_danger >= 1 and pair_danger >= 1 and street >= 1 and adapted and opp_agg >= AGG_DANGER_THRESHOLD:
                    vbet_threshold += AGG_DANGER_EXTRA_BUMP
                if commit_band == 1:
                    vbet_threshold += SOFT_COMMIT_CALL_BUMP
                elif commit_band == 2:
                    vbet_threshold += HIGH_COMMIT_CALL_BUMP
                elif commit_band >= 3:
                    vbet_threshold += NEAR_ALLIN_CALL_BUMP
                if early_phase and street >= 1:
                    vbet_threshold += EARLY_POSTFLOP_CALL_BUMP
                if street == 3:
                    vbet_threshold += float(_sp.get("river_vbet_eq_discount", 0.0))
                # Nut-awareness: suppress value-betting non-nut hands on scary boards
                nut_blocked = False
                if hand_rank_class is not None and street >= 1:
                    if flush_danger >= 1 and hand_rank_class > NUT_VBET_MIN_ON_FLUSH_BOARD:
                        nut_blocked = True
                    if pair_danger >= 2 and hand_rank_class > NUT_VBET_MIN_ON_PAIRED_BOARD:
                        nut_blocked = True
                    if opp_flush_sig >= 2 and hand_rank_class > NUT_VBET_MIN_ON_FLUSH_BOARD:
                        nut_blocked = True
                    if non_nut_flush:
                        nut_blocked = True
                    if straight_dominated:
                        nut_blocked = True
                if adj_equity >= vbet_threshold and not nut_blocked:
                    raw = int(pot * _raise_frac_with_polarization(False))
                    amount = max(min_raise, min(raw, max_raise))
                    action = (1, amount, 0, 0)
            elif valid[2]:
                action = (2, 0, 0, 0)
            elif valid[3]:
                action = (3, 0, 0, 0)
        return _choose_final(action)

    # 2.5. CONTINUATION BET: on the flop when in position, checked to, and we raised preflop (repping strong)
    if (street == 1
            and continue_cost == 0
            and valid[1]
            and in_position
            and my_raises_this_hand >= 1
            and adj_equity >= _cbet_eq
            and adj_equity >= CBET_GIVE_UP_EQUITY):
        raw = int(pot * CBET_FRAC)
        amount = max(min_raise, min(raw, max_raise))
        return _choose_final((1, amount, 0, 0))

    # 3. MARGINAL HAND: base requirement = pot odds (no street floors)
    marginal_min_eq = pot_odds if continue_cost > 0 else 0.0
    marginal_floor = pot_odds if continue_cost > 0 else 0.0
    # Commitment tightening: avoid thin calls when heavily invested.
    if commit_band == 1:
        marginal_floor += SOFT_COMMIT_CALL_BUMP
        marginal_min_eq += SOFT_COMMIT_POT_ODDS_BUMP
    elif commit_band == 2:
        marginal_floor += HIGH_COMMIT_CALL_BUMP
        marginal_min_eq += HIGH_COMMIT_POT_ODDS_BUMP
    elif commit_band >= 3:
        marginal_floor += NEAR_ALLIN_CALL_BUMP
        marginal_min_eq += NEAR_ALLIN_POT_ODDS_BUMP
    if street >= 1 and continue_cost > 0 and my_bet >= COMMIT50_BET:
        marginal_floor += COMMIT50_CALL_BUMP
        marginal_min_eq += COMMIT50_POTODDS_BUMP
    if street >= 1 and continue_cost > 0 and my_bet >= NEAR_ALLIN_BET:
        marginal_floor += COMMIT80_CALL_BUMP_EXTRA
        marginal_min_eq += COMMIT80_POTODDS_BUMP_EXTRA
    if early_phase and street >= 1 and continue_cost > 0:
        marginal_floor += EARLY_POSTFLOP_CALL_BUMP
        marginal_min_eq += EARLY_POSTFLOP_CALL_BUMP
    if continue_cost > 0 and street >= 1:
        if opp_postflop_pressure == 1:
            marginal_floor += OPP_POSTFLOP_PRESSURE_MILD_CALL_BUMP
            marginal_min_eq += OPP_POSTFLOP_PRESSURE_MILD_CALL_BUMP
        elif opp_postflop_pressure >= 2:
            marginal_floor += OPP_POSTFLOP_PRESSURE_HIGH_CALL_BUMP
            marginal_min_eq += OPP_POSTFLOP_PRESSURE_HIGH_CALL_BUMP
        if opp_reraise_pressure == 1:
            marginal_floor += OPP_POSTFLOP_PRESSURE_MILD_CALL_BUMP
            marginal_min_eq += OPP_POSTFLOP_PRESSURE_MILD_CALL_BUMP
        elif opp_reraise_pressure >= 2:
            marginal_floor += OPP_POSTFLOP_PRESSURE_HIGH_CALL_BUMP
            marginal_min_eq += OPP_POSTFLOP_PRESSURE_HIGH_CALL_BUMP
        if opp_high_commit_pressure == 1:
            marginal_floor += OPP_POSTFLOP_PRESSURE_MILD_CALL_BUMP
            marginal_min_eq += OPP_POSTFLOP_PRESSURE_MILD_CALL_BUMP
        elif opp_high_commit_pressure >= 2:
            marginal_floor += OPP_POSTFLOP_PRESSURE_HIGH_CALL_BUMP
            marginal_min_eq += OPP_POSTFLOP_PRESSURE_HIGH_CALL_BUMP
    if continue_cost > 0 and street >= 2 and opp_postflop_pressure >= 2:
        marginal_floor += TURN_RIVER_VS_BARREL_BUMP
        marginal_min_eq += TURN_RIVER_VS_BARREL_BUMP
    if continue_cost > 0 and street >= 1 and multi_street_raise_pressure:
        marginal_floor += MULTISTREET_RAISE_PATTERN_BUMP
        marginal_min_eq += MULTISTREET_RAISE_PATTERN_BUMP
    if second_raise_this_street:
        marginal_floor += SECOND_RAISE_STREET_CALL_BUMP
        marginal_min_eq += SECOND_RAISE_STREET_CALL_BUMP
    if (
        continue_cost > 0
        and street >= 1
        and MEDIUM_POT_MIN <= pot <= MEDIUM_POT_MAX
        and opp_postflop_pressure >= 1
    ):
        marginal_floor += MEDIUM_POT_SPIRAL_BUMP
        marginal_min_eq += MEDIUM_POT_SPIRAL_BUMP
        if opp_postflop_pressure >= 2:
            marginal_floor += MEDIUM_POT_SPIRAL_EXTRA_BUMP
            marginal_min_eq += MEDIUM_POT_SPIRAL_EXTRA_BUMP
    # Large pot tightening
    if street >= 2 and pot >= LARGE_POT_THRESHOLD:
        marginal_floor += LARGE_POT_EQUITY_BUMP
    if street >= 2 and pot >= HUGE_POT_THRESHOLD:
        marginal_floor += HUGE_POT_EQUITY_BUMP
    # Flush-heavy board: demand more equity when we don't hold the flush
    if flush_danger == 2:
        marginal_floor += FLUSH_DANGER_HIGH_EQUITY_BUMP
    elif flush_danger == 1:
        marginal_floor += FLUSH_DANGER_MODERATE_EQUITY_BUMP
    # Paired board: demand more equity when we don't connect with the pair
    if pair_danger == 2:
        marginal_floor += PAIRED_BOARD_HIGH_EQUITY_BUMP
    elif pair_danger == 1:
        marginal_floor += PAIRED_BOARD_MODERATE_EQUITY_BUMP
    if flush_danger >= 1 and pair_danger >= 1 and street >= 2 and commit_band >= 1:
        if street == 3:
            marginal_floor += COMPOUND_DANGER_RIVER_BUMP
        else:
            marginal_floor += COMPOUND_DANGER_TURN_BUMP
    if street >= 1 and continue_cost > 0 and my_bet >= COMMIT50_BET and (flush_danger >= 1 or pair_danger >= 1):
        marginal_floor += HIGH_COMMIT_DANGER_EXTRA_BUMP
        marginal_min_eq += HIGH_COMMIT_DANGER_EXTRA_BUMP
    if flush_danger >= 1 and pair_danger >= 1 and street >= 1 and adapted and opp_agg >= AGG_DANGER_THRESHOLD:
        marginal_floor += AGG_DANGER_EXTRA_BUMP
    # Opponent discard signal: if opponent likely has a flush, tighten calling
    if opp_flush_sig >= 2 and street >= 1 and continue_cost > 0:
        marginal_floor += FLUSH_DANGER_HIGH_EQUITY_BUMP
        marginal_min_eq += FLUSH_DANGER_HIGH_EQUITY_BUMP
    elif opp_flush_sig >= 1 and street >= 1 and continue_cost > 0:
        marginal_floor += FLUSH_DANGER_MODERATE_EQUITY_BUMP
        marginal_min_eq += FLUSH_DANGER_MODERATE_EQUITY_BUMP
    if opp_discarded_pair and street >= 1 and continue_cost > 0:
        marginal_floor += OPP_DISCARDED_PAIR_CALL_BUMP
        marginal_min_eq += OPP_DISCARDED_PAIR_CALL_BUMP
    if street >= 1 and continue_cost > 0 and my_raises_this_hand >= 2:
        marginal_floor += INVESTED_THEN_PRESSURED_CALL_BUMP
        marginal_min_eq += INVESTED_THEN_PRESSURED_CALL_BUMP
    if street == 3 and continue_cost > 0 and commit_band >= 2:
        marginal_min_eq += RIVER_OVERCOMMIT_CALL_BUMP
    # Hand-class pot-size gates (marginal branch; pair/two-pair use pot odds)
    if continue_cost > 0 and street >= 1 and hand_rank_class is not None:
        if hand_rank_class == 6 and my_bet >= TRIPS_MAX_COMMIT and (flush_danger >= 1 or pair_danger >= 1):
            marginal_floor += HAND_CLASS_COMMIT_BUMP
            marginal_min_eq += HAND_CLASS_COMMIT_BUMP
    # Non-nut flush / dominated straight (marginal branch)
    if continue_cost > 0 and street >= 1:
        if non_nut_flush:
            marginal_floor += NON_NUT_FLUSH_CALL_BUMP
            marginal_min_eq += NON_NUT_FLUSH_CALL_BUMP
        opp_kept_hf_m = bool(_info.get("opp_kept_high_flush", False))
        if non_nut_flush and (opp_flush_sig >= 2 or opp_kept_hf_m) and continue_cost > 0:
            marginal_floor += NON_NUT_FLUSH_VS_OPP_NUT_BUMP
            marginal_min_eq += NON_NUT_FLUSH_VS_OPP_NUT_BUMP
        if (
            non_nut_flush
            and opp_flush_sig >= 2
            and continue_cost > 0
            and street >= 1
        ):
            marginal_floor += NON_NUT_FLUSH_COMPOUND_STACK_BUMP
            marginal_min_eq += NON_NUT_FLUSH_COMPOUND_STACK_BUMP
        if (
            opp_kept_hf_m
            and opp_flush_sig >= 1
            and continue_cost > 0
            and street >= 1
        ):
            marginal_floor += OPP_KEPT_HIGH_FLUSH_SIG_BUMP
            marginal_min_eq += OPP_KEPT_HIGH_FLUSH_SIG_BUMP
        if straight_dominated:
            marginal_floor += STRAIGHT_DOMINATED_CALL_BUMP
            marginal_min_eq += STRAIGHT_DOMINATED_CALL_BUMP

    # --- Cap total bumps to prevent over-tightening ---
    base_marginal_floor = pot_odds if continue_cost > 0 else 0.0
    if marginal_floor > base_marginal_floor + MAX_CALL_FLOOR_BUMP:
        marginal_floor = base_marginal_floor + MAX_CALL_FLOOR_BUMP

    # --- Pot-odds sanity: tiny bet into large pot = always call with any hand ---
    if continue_cost > 0 and pot > 0:
        m_bet_frac_vs_pot = continue_cost / pot
        if m_bet_frac_vs_pot < TINY_BET_POT_FRAC:
            marginal_floor = min(marginal_floor, TINY_BET_MAX_EQUITY_REQ)
            marginal_min_eq = min(marginal_min_eq, TINY_BET_MAX_EQUITY_REQ)
        elif street >= 1 and continue_cost <= MICRO_BET_MAX_CONTINUE:
            marginal_floor = min(marginal_floor, MICRO_BET_MAX_EQUITY_REQ)
            marginal_min_eq = min(marginal_min_eq, MICRO_BET_MAX_EQUITY_REQ)

    # --- Pot-odds override: call when equity beats pot odds even if below floor ---
    # if (
    #     continue_cost > 0
    #     and pot > 0
    #     and street <= 2
    #     and adj_equity >= pot_odds + POT_ODDS_OVERRIDE_MARGIN
    #     and adj_equity >= POT_ODDS_OVERRIDE_MIN_EQUITY
    #     and valid[3]
    # ):
    #     return _choose_final((3, 0, 0, 0))

    if continue_cost > 0 and adj_equity >= marginal_min_eq and adj_equity >= marginal_floor:
        if valid[3]:
            action = (3, 0, 0, 0)
            return _choose_final(action)

    # Preflop with marginal hand and no cost: still open-raise (don't limp)
    if street == 0 and continue_cost <= 0 and valid[1]:
        if opp_postflop_pressure >= 1 and adj_equity < 0.74:
            if valid[2]:
                return _choose_final((2, 0, 0, 0))
            if valid[3]:
                return _choose_final((3, 0, 0, 0))
        raw = int(2 * _open_raise_mult)
        amount = max(min_raise, min(raw, max_raise))
        return _choose_final((1, amount, 0, 0))

    def _local_bluff_freq() -> float:
        if adapted and opp_model.is_calling_station_postflop():
            return 0.0
        freq = _bluff_freq_base + (opp_fold_rate - 0.30) * 0.4
        return max(0.0, min(freq, _bluff_freq_cap))

    if valid[2]:
        # Check, but sometimes bluff-raise (only vs very foldy opponents)
        bluff_freq = _local_bluff_freq()
        if early_phase:
            bluff_freq *= 0.75
        if opp_postflop_pressure >= 1 or opp_reraise_pressure >= 1:
            bluff_freq *= 0.60
        if opp_high_commit_pressure >= 1:
            bluff_freq *= 0.50
        if my_raises_this_hand >= 2:
            bluff_freq *= 0.50
        if (
            valid[1]
            and _semi_bluff_min <= adj_equity <= _semi_bluff_max
            and random.random() < bluff_freq
            and opp_fold_rate > 0.50
            and commit_band == 0
        ):
            raw = int(pot * _raise_frac_with_polarization(True))
            amount = max(min_raise, min(raw, max_raise))
            action = (1, amount, 0, 0)
            return _choose_final(action)
        action = (2, 0, 0, 0)
        return _choose_final(action)

    # 4. WEAK HAND facing a bet: semi-bluff raise selectively, else fold
    if (
        valid[1]
        and _semi_bluff_min <= adj_equity <= _semi_bluff_max
        and commit_band == 0
    ):
        bluff_freq = _local_bluff_freq()
        if early_phase:
            bluff_freq *= 0.75
        if opp_postflop_pressure >= 1 or opp_reraise_pressure >= 1:
            bluff_freq *= 0.60
        if opp_high_commit_pressure >= 1:
            bluff_freq *= 0.50
        if my_raises_this_hand >= 2:
            bluff_freq *= 0.50
        if random.random() < bluff_freq and opp_fold_rate > 0.55:
            raw = int(pot * _raise_frac_with_polarization(True))
            amount = max(min_raise, min(raw, max_raise))
            action = (1, amount, 0, 0)
            return _choose_final(action)

    # Default: fold (blended_action is also allowed to be fold)
    return _choose_final((0, 0, 0, 0))


def _bluff_frequency(opp_fold_rate: float) -> float:
    """Scale bluff frequency based on how often the opponent folds."""
    freq = BASE_BLUFF_FREQ + (opp_fold_rate - 0.30) * 0.4
    return max(0.0, min(freq, MAX_BLUFF_FREQ))


def _abstract_state_key(
    *,
    street: int,
    in_position: bool,
    pot: int,
    continue_cost: int,
    equity: float,
) -> str:
    """
    Build a compact discrete state key for strategy-table lookup.
    Keep this in sync with the offline trainer.
    """
    pos = 1 if in_position else 0

    if pot < 10:
        pot_band = 0
    elif pot < 50:
        pot_band = 1
    else:
        pot_band = 2

    if continue_cost <= 0:
        cost_band = 0
    elif continue_cost <= 5:
        cost_band = 1
    else:
        cost_band = 2

    # Equity bands: 0..4
    if equity < 0.20:
        eq_band = 0
    elif equity < 0.40:
        eq_band = 1
    elif equity < 0.60:
        eq_band = 2
    elif equity < 0.78:
        eq_band = 3
    else:
        eq_band = 4

    return f"s{street}_p{pos}_pb{pot_band}_cb{cost_band}_eb{eq_band}"


def _pick_action_from_table(
    *,
    probs: dict[str, float],
    valid_actions: list[bool],
    pot: int,
    my_bet: int,
    min_raise: int,
    max_raise: int,
    equity: float,
    opp_fold_rate: float,
) -> tuple[int, int, int, int] | None:
    """
    Map abstract action probabilities to a legal engine action.
    """
    # Start from table probabilities; lightly tilt raises upward vs foldy opponents.
    raise_tilt = max(-0.15, min(0.15, (opp_fold_rate - 0.30) * 0.3))

    p_fold = max(0.0, float(probs.get("fold", 0.0)))
    p_call = max(0.0, float(probs.get("call", 0.0)))
    p_rs = max(0.0, float(probs.get("raise_small", 0.0))) * (1.0 + raise_tilt)
    p_rb = max(0.0, float(probs.get("raise_big", 0.0))) * (1.0 + raise_tilt)

    # Remove illegal actions.
    if not valid_actions[0]:
        p_fold = 0.0
    if not valid_actions[3] and not valid_actions[2]:
        p_call = 0.0
    if not valid_actions[1]:
        p_rs = 0.0
        p_rb = 0.0

    total = p_fold + p_call + p_rs + p_rb
    if total <= 0.0:
        return None

    r = random.random() * total
    if r < p_fold:
        return (0, 0, 0, 0)
    r -= p_fold

    if r < p_call:
        # Use CALL if facing a bet, else CHECK.
        if valid_actions[3]:
            return (3, 0, 0, 0)
        if valid_actions[2]:
            return (2, 0, 0, 0)
        return None
    r -= p_call

    # Raises: translate abstract sizing into concrete raise_to.
    # Use pot fractions but clamp to engine limits.
    if r < p_rs:
        frac = random.uniform(0.50, 0.68)
    else:
        frac = random.uniform(0.68, 0.85) if equity >= 0.60 else random.uniform(0.55, 0.72)

    raw = int(pot * frac)
    amount = max(min_raise, min(raw, max_raise))
    if amount <= 0:
        return None
    return (1, amount, 0, 0)
