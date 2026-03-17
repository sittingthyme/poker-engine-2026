"""
Betting strategy for the hybrid poker bot.

Position-aware, equity-driven decisions with adaptive bluffing
tuned to the opponent model collected over the match.
"""

from __future__ import annotations

import random
from submission.opponent_model import OpponentModel
from submission.strategy_table import StrategyTable

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

# Table strategy mode: "off" | "simple" | "conf"
# - off: heuristic only, table never used
# - simple: fixed 30% blend when table has entry
# - conf: confidence-weighted blend + EV safety check
TABLE_MODE = "conf"
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


def decide_action(
    equity: float,
    observation: dict,
    opp_model: OpponentModel,
    strategy_table: StrategyTable | None = None,
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

    # ---- Optional baseline policy via StrategyTable (blended with heuristic) ----
    blended_action: tuple[int, int, int, int] | None = None
    use_table = strategy_table is not None and TABLE_MODE in ("simple", "conf")
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
            return heuristic_action
        if TABLE_MODE == "conf":
            ev_h = _approx_ev_for_action(heuristic_action, **_ev_params())
            ev_t = _approx_ev_for_action(blended_action, **_ev_params())
            if ev_t < ev_h - TABLE_EV_EPSILON:
                return heuristic_action
        return blended_action

    # ---- Heuristic decision tree (existing logic) ----

    # 1. STRONG HAND: value bet / raise
    if adj_equity >= strong_equity:
        if valid[1]:  # RAISE
            frac = STRONG_RAISE_FRAC if adj_equity >= 0.88 else VALUE_RAISE_FRAC
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

    # 2. MEDIUM HAND: call if pot odds are right, play passively
    effective_medium = medium_equity + call_threshold_adj
    if adj_equity >= effective_medium:
        action = (0, 0, 0, 0)
        if continue_cost > 0 and adj_equity >= pot_odds and valid[3]:
            action = (3, 0, 0, 0)
        if valid[2]:
            action = (2, 0, 0, 0)
        if valid[3]:
            action = (3, 0, 0, 0)
        return _choose_final(action)

    # 3. MARGINAL HAND: call if cheap enough relative to equity
    if continue_cost > 0 and adj_equity >= pot_odds:
        if valid[3]:
            action = (3, 0, 0, 0)
            return _choose_final(action)

    if valid[2]:
        # Check, but sometimes bluff-raise (only vs very foldy opponents)
        bluff_freq = _bluff_frequency(opp_fold_rate)
        if (valid[1] and adj_equity < bluff_equity_ceil
                and random.random() < bluff_freq
                and opp_fold_rate > 0.50):
            raw = int(pot * BLUFF_RAISE_FRAC)
            amount = max(min_raise, min(raw, max_raise))
            action = (1, amount, 0, 0)
            return _choose_final(action)
        action = (2, 0, 0, 0)
        return _choose_final(action)

    # 4. WEAK HAND facing a bet: bluff-raise very selectively, else fold
    if valid[1] and adj_equity < bluff_equity_ceil:
        bluff_freq = _bluff_frequency(opp_fold_rate)
        if random.random() < bluff_freq and opp_fold_rate > 0.55:
            raw = int(pot * BLUFF_RAISE_FRAC)
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
        frac = 0.50
    else:
        frac = 0.80 if equity >= 0.60 else 0.65

    raw = int(pot * frac)
    amount = max(min_raise, min(raw, max_raise))
    if amount <= 0:
        return None
    return (1, amount, 0, 0)
