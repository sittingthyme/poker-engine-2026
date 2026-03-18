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
# 27-card deck: straights/flushes/full house are common; two pair is vulnerable – play tight
BASE_STRONG_EQUITY = 0.85       # value-bet/raise only with very strong hands (was 0.82)
BASE_MEDIUM_EQUITY = 0.65       # call only with solid equity (was 0.60)
BASE_BLUFF_EQUITY_CEIL = 0.20   # base bluff range ceiling

# Extra equity margin when calling (short deck: avoid thin calls vs draws)
CALL_MARGIN_ABOVE_POT_ODDS = 0.10   # require equity >= pot_odds + this to call (was 0.06)
# Minimum equity to call in marginal branch (avoid calling with weak draws)
MIN_EQUITY_TO_CALL_MARGINAL = 0.55
# Street-specific call floors: later streets need stronger hands (no cards to come)
TURN_MIN_EQUITY_CALL = 0.58
RIVER_MIN_EQUITY_CALL = 0.62
# Extra margin when facing a big bet (>=50% pot)
TURN_BIG_BET_EXTRA_MARGIN = 0.06
RIVER_BIG_BET_EXTRA_MARGIN = 0.08

# Pot-size tightening: large pots on turn/river signal multi-street aggression
LARGE_POT_THRESHOLD = 40     # pot >= 40 chips
LARGE_POT_EQUITY_BUMP = 0.04
HUGE_POT_THRESHOLD = 80      # pot >= 80 chips
HUGE_POT_EQUITY_BUMP = 0.04  # cumulative with above → +0.08 for huge pots

# Board flush danger: 3-suit 27-card deck makes flushes very common.
# When 3+ community cards share a suit, tighten calling if we don't hold the flush.
FLUSH_DANGER_HIGH_EQUITY_BUMP = 0.10    # 3+ of one suit on board, we hold 0 of that suit
FLUSH_DANGER_MODERATE_EQUITY_BUMP = 0.04  # 3+ of one suit on board, we hold 1 of that suit

# Paired-board danger: when board has a pair/trips, full houses and quads are common.
# Tighten if we don't connect with the paired rank ourselves.
PAIRED_BOARD_HIGH_EQUITY_BUMP = 0.08   # board paired/trips and we don't match that rank at all
PAIRED_BOARD_MODERATE_EQUITY_BUMP = 0.03  # board paired/trips but we match with 1 card

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

# Raise-back lockout: when facing post-flop aggression in committed pots,
# default to call/fold and only re-raise with very premium equity.
POSTFLOP_RAISEBACK_PREMIUM_EQUITY = 0.90

# Danger compounding: when both flush and paired-board danger are present.
COMPOUND_DANGER_TURN_BUMP = 0.05
COMPOUND_DANGER_RIVER_BUMP = 0.08

# River overcommit guard when facing a raise in large commitments.
RIVER_OVERCOMMIT_CALL_BUMP = 0.08

# Dynamic threshold adjustments
TIGHT_OPPONENT_STRONG_ADJ = -0.05   # Lower threshold vs tight (more aggressive)
TIGHT_OPPONENT_MEDIUM_ADJ = -0.05   # Lower threshold vs tight
LOOSE_OPPONENT_STRONG_ADJ = +0.03   # Higher threshold vs loose (tighter)
LOOSE_OPPONENT_MEDIUM_ADJ = +0.03   # Higher threshold vs loose

# Raise sizing (fraction of pot)
VALUE_RAISE_FRAC = 0.60     # standard value raise
STRONG_RAISE_FRAC = 0.80    # big value raise for very strong hands
BLUFF_RAISE_FRAC = 0.40     # bluff sizing (smaller to risk less)

# Preflop open-raise sizing (multiples of BB=2)
PREFLOP_OPEN_RAISE_MULTIPLIER = 3.5  # raise to 3.5x BB = 7 chips (top bots use 5-9x)
PREFLOP_3BET_MULTIPLIER = 2.5        # 3-bet to 2.5x the incoming raise (down from 3.0)

# Facing a 3-bet: tighter ranges to avoid bloated pots with marginal hands
PREFLOP_3BET_CALL_MIN_EQUITY = 0.62  # need 62%+ equity to call a 3-bet
PREFLOP_4BET_MIN_EQUITY = 0.80       # need 80%+ equity to 4-bet (very premium only)

# Re-raise threshold: when facing a bet, re-raise for value if equity >= this
RERAISE_EQUITY_THRESHOLD = 0.78

# Value betting: when checked to, bet with medium-strong hands instead of checking
VALUE_BET_EQUITY = 0.72

# Continuation bet (c-bet): bet the flop when we raised preflop, even without a strong hand
CBET_EQUITY_THRESHOLD = 0.45   # c-bet with 45%+ equity on the flop
CBET_FRAC = 0.55               # c-bet size: 55% of pot
CBET_GIVE_UP_EQUITY = 0.30     # don't c-bet with very weak hands (below 30%)

# Preflop escalation cap: stop re-raising if already committed this many chips
PREFLOP_RERAISE_CAP = 10
PREFLOP_PREMIUM_EQUITY = 0.75  # only keep re-raising with top equity

# Hand strength floor: never fold trips or better on the river (treys rank_class <= 6)
TRIPS_OR_BETTER_RANK_CLASS = 6

# Bluff parameters – nearly eliminated vs math-based opponents
BASE_BLUFF_FREQ = 0.02      # base bluff frequency (very conservative)
MAX_BLUFF_FREQ = 0.08       # cap even vs very foldy opponents
MIN_HANDS_FOR_ADAPT = 20    # hands before trusting opponent model

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


def decide_action(
    equity: float,
    observation: dict,
    opp_model: OpponentModel,
    info: dict | None = None,
    hand_rank_class: int | None = None,
) -> tuple[int, int, int, int]:
    """
    Choose (action_type, raise_amount, keep1, keep2) for a betting decision.

    action_type values: FOLD=0, RAISE=1, CHECK=2, CALL=3

    hand_rank_class: optional treys rank class (1=SF..9=high card).
        When provided on the river, used to enforce a strength floor.
    """
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
    # Need lead > 1 chip per remaining hand so folding every hand still leaves us ahead
    min_lead_to_fold = 2 * hands_left
    if valid[0] and hands_left > 0 and my_bankroll > min_lead_to_fold:
        return (0, 0, 0, 0)  # FOLD

    # Cost to continue (computed early for preflop cap check)
    continue_cost = opp_bet - my_bet

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
        if equity < PREFLOP_3BET_CALL_MIN_EQUITY:
            if valid[0]:
                return (0, 0, 0, 0)  # FOLD — not strong enough vs a real raise
        elif equity < PREFLOP_4BET_MIN_EQUITY:
            if valid[3]:
                return (3, 0, 0, 0)  # CALL — see a flop, don't escalate

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
    lockout_reraise = facing_postflop_raise and commit_band >= 1

    # Position: who acts LAST has position advantage.
    #   Pre-flop (street 0):  SB acts first → BB (blind_pos=1) is in position.
    #   Post-flop (street≥1): BB acts first → SB (blind_pos=0) is in position.
    in_position = (street == 0 and blind_pos == 1) or (street >= 1 and blind_pos == 0)

    # Adjust equity for position
    adj_equity = equity + (IP_EQUITY_BONUS if in_position else 0.0)

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

    # ---- Hand-strength floor: never fold trips or better on the river ----
    if (
        street == 3
        and hand_rank_class is not None
        and hand_rank_class <= TRIPS_OR_BETTER_RANK_CLASS
        and continue_cost > 0
    ):
        # We have trips+.  Always at least call; raise if equity is high.
        river_raise_floor = RERAISE_EQUITY_THRESHOLD
        if commit_band >= 2:
            river_raise_floor = max(river_raise_floor, POSTFLOP_RAISEBACK_PREMIUM_EQUITY)
        if lockout_reraise:
            river_raise_floor = max(river_raise_floor, POSTFLOP_RAISEBACK_PREMIUM_EQUITY)
        if adj_equity >= river_raise_floor and valid[1]:
            frac = VALUE_RAISE_FRAC
            raw = int(pot * frac)
            amount = max(min_raise, min(raw, max_raise))
            return (1, amount, 0, 0)
        if valid[3]:
            return (3, 0, 0, 0)

    # ---- Heuristic decision tree (existing logic) ----

    # 1. STRONG HAND: value bet / raise (high bar in short deck)
    if adj_equity >= strong_equity:
        can_strong_raise = valid[1]
        if lockout_reraise and adj_equity < POSTFLOP_RAISEBACK_PREMIUM_EQUITY:
            can_strong_raise = False
        if can_strong_raise:  # RAISE
            if street == 0:
                # Preflop: use BB-based sizing instead of pot-fraction
                if continue_cost <= 0 or my_bet <= 2:
                    # Open-raise or first raise: 3.5x BB
                    raw = int(2 * PREFLOP_OPEN_RAISE_MULTIPLIER)
                else:
                    # 3-bet: 3x the incoming raise
                    raw = int(opp_bet * PREFLOP_3BET_MULTIPLIER)
            else:
                frac = STRONG_RAISE_FRAC if adj_equity >= 0.92 else VALUE_RAISE_FRAC
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
    min_equity_to_call = pot_odds + CALL_MARGIN_ABOVE_POT_ODDS if continue_cost > 0 else 0.0
    # Tighten call floor on later streets (no more cards to improve)
    if street >= 3:
        call_floor = RIVER_MIN_EQUITY_CALL
    elif street >= 2:
        call_floor = TURN_MIN_EQUITY_CALL
    else:
        call_floor = MIN_EQUITY_TO_CALL_MARGINAL
    # Facing a big bet (>=50% pot): demand extra margin
    if continue_cost > 0 and pot > 0:
        bet_frac = continue_cost / pot
        if street == 3 and bet_frac >= 0.50:
            min_equity_to_call += RIVER_BIG_BET_EXTRA_MARGIN
        elif street == 2 and bet_frac >= 0.50:
            min_equity_to_call += TURN_BIG_BET_EXTRA_MARGIN
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
    # River overcommit guard: when already deep and facing aggression, demand more.
    if street == 3 and continue_cost > 0 and commit_band >= 2:
        min_equity_to_call += RIVER_OVERCOMMIT_CALL_BUMP

    if adj_equity >= effective_medium:
        action = (0, 0, 0, 0)
        if continue_cost > 0:
            # Re-raise for value when equity is near the top of the medium band.
            # On preflop, only re-raise vs blind completion (opp_bet <= 2);
            # vs real raises, just call — pot control.
            can_reraise = adj_equity >= RERAISE_EQUITY_THRESHOLD and valid[1]
            if street == 0 and opp_bet > 2:
                can_reraise = False
            if street >= 1 and commit_band >= 1:
                can_reraise = False
            if lockout_reraise and adj_equity < POSTFLOP_RAISEBACK_PREMIUM_EQUITY:
                can_reraise = False
            if can_reraise:
                if street == 0:
                    raw = int(2 * PREFLOP_OPEN_RAISE_MULTIPLIER)
                else:
                    raw = int(pot * VALUE_RAISE_FRAC)
                amount = max(min_raise, min(raw, max_raise))
                action = (1, amount, 0, 0)
            elif adj_equity >= max(min_equity_to_call, call_floor) and valid[3]:
                action = (3, 0, 0, 0)
        else:
            if street == 0 and valid[1]:
                # Preflop open-raise with medium hands too
                raw = int(2 * PREFLOP_OPEN_RAISE_MULTIPLIER)
                amount = max(min_raise, min(raw, max_raise))
                action = (1, amount, 0, 0)
            elif valid[1]:
                vbet_threshold = VALUE_BET_EQUITY
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
                if commit_band == 1:
                    vbet_threshold += SOFT_COMMIT_CALL_BUMP
                elif commit_band == 2:
                    vbet_threshold += HIGH_COMMIT_CALL_BUMP
                elif commit_band >= 3:
                    vbet_threshold += NEAR_ALLIN_CALL_BUMP
                if adj_equity >= vbet_threshold:
                    raw = int(pot * VALUE_RAISE_FRAC)
                    amount = max(min_raise, min(raw, max_raise))
                    action = (1, amount, 0, 0)
            elif valid[2]:
                action = (2, 0, 0, 0)
            elif valid[3]:
                action = (3, 0, 0, 0)
        return _choose_final(action)

    # 2.5. CONTINUATION BET: on the flop when checked to, bet with decent equity
    if (street == 1
            and continue_cost == 0
            and valid[1]
            and adj_equity >= CBET_EQUITY_THRESHOLD
            and adj_equity >= CBET_GIVE_UP_EQUITY):
        raw = int(pot * CBET_FRAC)
        amount = max(min_raise, min(raw, max_raise))
        return _choose_final((1, amount, 0, 0))

    # 3. MARGINAL HAND: call only if equity clearly above pot odds and above floor (avoid weak draws)
    marginal_min_eq = pot_odds + CALL_MARGIN_ABOVE_POT_ODDS
    # Use street-specific floor (river/turn require stronger hands)
    if street >= 3:
        marginal_floor = RIVER_MIN_EQUITY_CALL
    elif street >= 2:
        marginal_floor = TURN_MIN_EQUITY_CALL
    else:
        marginal_floor = MIN_EQUITY_TO_CALL_MARGINAL
    # Facing a big bet: extra margin
    if continue_cost > 0 and pot > 0:
        m_bet_frac = continue_cost / pot
        if street == 3 and m_bet_frac >= 0.50:
            marginal_min_eq += RIVER_BIG_BET_EXTRA_MARGIN
        elif street == 2 and m_bet_frac >= 0.50:
            marginal_min_eq += TURN_BIG_BET_EXTRA_MARGIN
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
    if street == 3 and continue_cost > 0 and commit_band >= 2:
        marginal_min_eq += RIVER_OVERCOMMIT_CALL_BUMP
    if continue_cost > 0 and adj_equity >= marginal_min_eq and adj_equity >= marginal_floor:
        if valid[3]:
            action = (3, 0, 0, 0)
            return _choose_final(action)

    # Preflop with marginal hand and no cost: still open-raise (don't limp)
    if street == 0 and continue_cost <= 0 and valid[1]:
        raw = int(2 * PREFLOP_OPEN_RAISE_MULTIPLIER)
        amount = max(min_raise, min(raw, max_raise))
        return _choose_final((1, amount, 0, 0))

    if valid[2]:
        # Check, but sometimes bluff-raise (only vs very foldy opponents)
        bluff_freq = _bluff_frequency(opp_fold_rate)
        if (valid[1] and adj_equity < bluff_equity_ceil
                and random.random() < bluff_freq
                and opp_fold_rate > 0.50
                and commit_band == 0):
            raw = int(pot * BLUFF_RAISE_FRAC)
            amount = max(min_raise, min(raw, max_raise))
            action = (1, amount, 0, 0)
            return _choose_final(action)
        action = (2, 0, 0, 0)
        return _choose_final(action)

    # 4. WEAK HAND facing a bet: bluff-raise very selectively, else fold
    if valid[1] and adj_equity < bluff_equity_ceil and commit_band == 0:
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
