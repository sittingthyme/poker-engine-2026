"""
Betting strategy for the hybrid poker bot.

Position-aware, equity-driven decisions with adaptive bluffing
tuned to the opponent model collected over the match.
"""

from __future__ import annotations

import random
from submission.opponent_model import OpponentModel

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

# Base equity bands (adjusted dynamically based on opponent)
BASE_STRONG_EQUITY = 0.78        # base value-bet / raise territory
BASE_MEDIUM_EQUITY = 0.55        # base call territory
BASE_BLUFF_EQUITY_CEIL = 0.20    # base bluff range ceiling

# Dynamic threshold adjustments
TIGHT_OPPONENT_STRONG_ADJ = -0.05   # Lower threshold vs tight (more aggressive)
TIGHT_OPPONENT_MEDIUM_ADJ = -0.05   # Lower threshold vs tight
LOOSE_OPPONENT_STRONG_ADJ = +0.03   # Higher threshold vs loose (tighter)
LOOSE_OPPONENT_MEDIUM_ADJ = +0.03   # Higher threshold vs loose

# Raise sizing (fraction of pot)
VALUE_RAISE_FRAC = 0.60     # standard value raise
STRONG_RAISE_FRAC = 0.80    # big value raise for very strong hands
BLUFF_RAISE_FRAC = 0.40     # bluff sizing (smaller to risk less)

# Bluff parameters – nearly eliminated vs math-based opponents
BASE_BLUFF_FREQ = 0.02      # base bluff frequency (very conservative)
MAX_BLUFF_FREQ = 0.08       # cap even vs very foldy opponents
MIN_HANDS_FOR_ADAPT = 20    # hands before trusting opponent model

# Position bonus: being in position (acting last) is an advantage
IP_EQUITY_BONUS = 0.03      # small bonus when in position post-flop

# Calling thresholds when facing aggression
AGG_CALL_DISCOUNT = 0.05    # widen call range vs hyper-aggressive opponents


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


def decide_action(
    equity: float,
    observation: dict,
    opp_model: OpponentModel,
) -> tuple[int, int, int, int]:
    """
    Choose (action_type, raise_amount, keep1, keep2) for a betting decision.

    action_type values: FOLD=0, RAISE=1, CHECK=2, CALL=3
    """
    valid = observation["valid_actions"]
    street = observation["street"]
    my_bet = observation["my_bet"]
    opp_bet = observation["opp_bet"]
    min_raise = observation["min_raise"]
    max_raise = observation["max_raise"]
    # Compute pot and blind position from available fields
    # (pot_size and blind_position may be stripped by Pydantic in API mode)
    pot = observation.get("pot_size", my_bet + opp_bet)
    blind_pos = observation.get("blind_position", blind_position_from_obs(observation))

    # Position: who acts LAST has position advantage.
    #   Pre-flop (street 0):  SB acts first → BB (blind_pos=1) is in position.
    #   Post-flop (street≥1): BB acts first → SB (blind_pos=0) is in position.
    in_position = (street == 0 and blind_pos == 1) or (street >= 1 and blind_pos == 0)

    # Adjust equity for position
    adj_equity = equity + (IP_EQUITY_BONUS if in_position else 0.0)

    # Cost to continue
    continue_cost = opp_bet - my_bet
    pot_odds = continue_cost / (continue_cost + pot) if continue_cost > 0 else 0.0

    # Opponent adaptation
    adapted = opp_model.hands_seen >= MIN_HANDS_FOR_ADAPT
    opp_fold_rate = opp_model.fold_rate(street) if adapted else 0.30
    opp_agg = opp_model.aggression(street) if adapted else 1.0
    
    # Use recent trends if available (weighted toward recent hands)
    bucket = opp_model.streets[street] if street is not None else opp_model.overall
    if adapted and bucket.recent_actions > 5:
        opp_fold_rate = opp_model.recent_fold_rate(street)
        opp_agg = opp_model.recent_aggression(street)

    # Dynamic threshold adjustment based on opponent type
    strong_equity = BASE_STRONG_EQUITY
    medium_equity = BASE_MEDIUM_EQUITY
    bluff_equity_ceil = BASE_BLUFF_EQUITY_CEIL
    
    if adapted:
        if opp_model.is_tight():
            # Tight opponent: lower thresholds (more aggressive)
            # They fold more, so we can bet/raise with weaker hands
            strong_equity += TIGHT_OPPONENT_STRONG_ADJ
            medium_equity += TIGHT_OPPONENT_MEDIUM_ADJ
        elif opp_model.is_loose():
            # Loose opponent: higher thresholds (tighter)
            # They call more, so we need stronger hands to value bet
            strong_equity += LOOSE_OPPONENT_STRONG_ADJ
            medium_equity += LOOSE_OPPONENT_MEDIUM_ADJ
    
    # Street-specific adjustments (can be more aggressive on later streets)
    if street == 3:  # River
        # More information, can be slightly more aggressive
        strong_equity -= 0.02
        medium_equity -= 0.02
    elif street == 0:  # Pre-flop
        # Less information, be slightly tighter
        strong_equity += 0.02
        medium_equity += 0.02

    # Against hyper-aggressive opponents, widen our calling range
    call_threshold_adj = 0.0
    if adapted and opp_agg > 2.0:
        call_threshold_adj = -AGG_CALL_DISCOUNT

    # ---- Decision tree ----

    # 1. STRONG HAND: value bet / raise
    if adj_equity >= strong_equity:
        if valid[1]:  # RAISE
            frac = STRONG_RAISE_FRAC if adj_equity >= 0.88 else VALUE_RAISE_FRAC
            raw = int(pot * frac)
            amount = max(min_raise, min(raw, max_raise))
            return (1, amount, 0, 0)
        if valid[3]:  # CALL
            return (3, 0, 0, 0)
        if valid[2]:  # CHECK
            return (2, 0, 0, 0)

    # 2. MEDIUM HAND: call if pot odds are right, play passively
    effective_medium = medium_equity + call_threshold_adj
    if adj_equity >= effective_medium:
        if continue_cost > 0 and adj_equity >= pot_odds and valid[3]:
            return (3, 0, 0, 0)
        if valid[2]:
            return (2, 0, 0, 0)
        if valid[3]:
            return (3, 0, 0, 0)

    # 3. MARGINAL HAND: call if cheap enough relative to equity
    if continue_cost > 0 and adj_equity >= pot_odds:
        if valid[3]:
            return (3, 0, 0, 0)

    if valid[2]:
        # Check, but sometimes bluff-raise (only vs very foldy opponents)
        bluff_freq = _bluff_frequency(opp_fold_rate)
        if (valid[1] and adj_equity < bluff_equity_ceil
                and random.random() < bluff_freq
                and opp_fold_rate > 0.50):
            raw = int(pot * BLUFF_RAISE_FRAC)
            amount = max(min_raise, min(raw, max_raise))
            return (1, amount, 0, 0)
        return (2, 0, 0, 0)

    # 4. WEAK HAND facing a bet: bluff-raise very selectively, else fold
    if valid[1] and adj_equity < bluff_equity_ceil:
        bluff_freq = _bluff_frequency(opp_fold_rate)
        if random.random() < bluff_freq and opp_fold_rate > 0.55:
            raw = int(pot * BLUFF_RAISE_FRAC)
            amount = max(min_raise, min(raw, max_raise))
            return (1, amount, 0, 0)

    # Default: fold
    return (0, 0, 0, 0)


def _bluff_frequency(opp_fold_rate: float) -> float:
    """Scale bluff frequency based on how often the opponent folds."""
    freq = BASE_BLUFF_FREQ + (opp_fold_rate - 0.30) * 0.4
    return max(0.0, min(freq, MAX_BLUFF_FREQ))
