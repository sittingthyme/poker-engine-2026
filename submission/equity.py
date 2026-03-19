"""
Monte Carlo equity calculator for the 27-card poker variant.

Computes win probability by simulating random opponent hands and board run-outs,
taking into account all known/revealed cards (own cards, community cards,
opponent discards, own discards).
"""

import random
from treys import Card, Evaluator

# ---------------------------------------------------------------------------
# Constants mirroring the engine
# ---------------------------------------------------------------------------
RANKS = "23456789A"
SUITS = "dhs"
DECK_SIZE = 27  # 9 ranks * 3 suits

# ---------------------------------------------------------------------------
# Card conversion helpers (self-contained so submission/ has no gym_env import)
# ---------------------------------------------------------------------------

def int_to_treys(card_int: int) -> int:
    """Convert our 0-26 card encoding to treys internal integer."""
    rank = RANKS[card_int % len(RANKS)]
    suit = SUITS[card_int // len(RANKS)]
    return Card.new(rank + suit)


def _ace_to_ten(treys_card: int) -> int:
    """Convert treys Ace to treys Ten for high-straight detection."""
    s = Card.int_to_str(treys_card)
    return Card.new(s.replace("A", "T"))


# Singleton evaluator -- constructed once, reused everywhere
_evaluator = Evaluator()


def evaluate_hand(hand_treys: list[int], board_treys: list[int]) -> int:
    """
    Evaluate a hand (2 cards) + board (5 cards) using the tournament's
    Ace-can-be-high (above 9) rule.  Lower score = better hand.
    """
    reg = _evaluator.evaluate(hand_treys, board_treys)
    alt_hand = list(map(_ace_to_ten, hand_treys))
    alt_board = list(map(_ace_to_ten, board_treys))
    alt = _evaluator.evaluate(alt_hand, alt_board)
    return min(reg, alt)


def get_hand_rank_class(hole_cards: list[int], community_cards: list[int]) -> int:
    """
    Return the treys rank class for our hand (1=SF .. 6=Trips .. 9=High).

    Requires exactly 2 hole cards and 5 community cards.
    """
    hand_treys = [int_to_treys(c) for c in hole_cards[:2]]
    board_treys = [int_to_treys(c) for c in community_cards[:5]]
    rank = evaluate_hand(hand_treys, board_treys)
    return _evaluator.get_rank_class(rank)


def count_dominating_hands(
    hole_cards: list[int],
    community_cards: list[int],
    opp_discarded: list[int] | None = None,
    my_discarded: list[int] | None = None,
) -> float:
    """
    Count the fraction of possible 2-card opponent hands that beat ours.

    Uses exact enumeration over all remaining 2-card combos to determine
    how dominated our current hand is.  Returns a float in [0, 1] where
    higher = more dominated.
    """
    board = [c for c in community_cards if c != -1]
    if len(board) < 5 or len(hole_cards) < 2:
        return 0.0

    hand_treys = [int_to_treys(c) for c in hole_cards[:2]]
    board_treys = [int_to_treys(c) for c in board]
    my_rank = evaluate_hand(hand_treys, board_treys)

    known = set(hole_cards[:2]) | set(board) | set(opp_discarded or []) | set(my_discarded or [])
    known.discard(-1)
    remaining = [c for c in range(DECK_SIZE) if c not in known]

    n = len(remaining)
    if n < 2:
        return 0.0

    beats_us = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            opp_hand = [int_to_treys(remaining[i]), int_to_treys(remaining[j])]
            opp_rank = evaluate_hand(opp_hand, board_treys)
            total += 1
            if opp_rank < my_rank:
                beats_us += 1

    return beats_us / total if total > 0 else 0.0


def get_hand_rank_class_partial(hole_cards: list[int], community_cards: list[int]) -> int | None:
    """
    Like get_hand_rank_class but works with 3-5 community cards.

    For fewer than 5 community cards, pads with random remaining cards
    and returns a conservative (pessimistic) rank class using simple
    rank counting rather than treys evaluation.

    Returns None if insufficient cards.
    """
    board = [c for c in community_cards if c != -1]
    if len(hole_cards) < 2 or len(board) < 3:
        return None

    if len(board) == 5:
        return get_hand_rank_class(hole_cards, community_cards)

    # For 3-4 board cards: count ranks to detect trips+ conservatively.
    rank_counts: dict[int, int] = {}
    for c in list(hole_cards[:2]) + board:
        r = c % 9
        rank_counts[r] = rank_counts.get(r, 0) + 1

    max_count = max(rank_counts.values())
    num_pairs = sum(1 for cnt in rank_counts.values() if cnt >= 2)

    if max_count >= 4:
        return 2  # Four of a kind
    if max_count >= 3 and num_pairs >= 2:
        return 3  # Full house
    if max_count >= 3:
        return 6  # Trips
    if num_pairs >= 2:
        return 7  # Two pair
    if num_pairs >= 1:
        return 8  # Pair
    return 9  # High card


# ---------------------------------------------------------------------------
# Core equity function
# ---------------------------------------------------------------------------

def compute_equity(
    my_cards: list[int],
    community_cards: list[int],
    opp_discarded: list[int] | None = None,
    my_discarded: list[int] | None = None,
    num_simulations: int = 300,
    return_nut_fraction: bool = False,
) -> float | tuple[float, float]:
    """
    Monte Carlo equity (win-rate) for *my_cards* vs a random opponent.

    Parameters
    ----------
    my_cards : list[int]
        Our hole cards (2 cards, 0-26 encoding).
    community_cards : list[int]
        Visible board cards (0-5 cards, -1 entries ignored).
    opp_discarded : list[int] | None
        Opponent's revealed discards (-1 entries ignored).
    my_discarded : list[int] | None
        Our own discards (-1 entries ignored).
    num_simulations : int
        Number of random roll-outs.
    return_nut_fraction : bool
        If True, also return the fraction of wins that came from
        flush or better (rank_class <= 4).  Used by discard logic
        to prefer keeps with higher nut potential.

    Returns
    -------
    float  in [0, 1]  when return_nut_fraction is False.
    (float, float)    when return_nut_fraction is True — (equity, nut_frac).
    """
    if opp_discarded is None:
        opp_discarded = []
    if my_discarded is None:
        my_discarded = []

    # Build the set of all known / dead cards
    known: set[int] = set(my_cards)
    board: list[int] = []
    for c in community_cards:
        if c != -1:
            known.add(c)
            board.append(c)
    for c in opp_discarded:
        if c != -1:
            known.add(c)
    for c in my_discarded:
        if c != -1:
            known.add(c)

    remaining = [i for i in range(DECK_SIZE) if i not in known]

    opp_needed = 2
    board_needed = 5 - len(board)
    sample_size = opp_needed + board_needed

    if sample_size > len(remaining):
        if return_nut_fraction:
            return 0.5, 0.0
        return 0.5

    # Pre-convert our cards once
    my_treys = [int_to_treys(c) for c in my_cards]

    wins = 0
    nut_wins = 0
    total = 0

    for _ in range(num_simulations):
        sample = random.sample(remaining, sample_size)
        opp_cards = sample[:opp_needed]
        full_board = board + sample[opp_needed:]

        opp_treys = [int_to_treys(c) for c in opp_cards]
        board_treys = [int_to_treys(c) for c in full_board]

        my_rank = evaluate_hand(my_treys, board_treys)
        opp_rank = evaluate_hand(opp_treys, board_treys)

        if my_rank < opp_rank:
            wins += 1
            if return_nut_fraction:
                rc = _evaluator.get_rank_class(my_rank)
                if rc <= 4:  # flush or better
                    nut_wins += 1
        total += 1

    if total == 0:
        if return_nut_fraction:
            return 0.5, 0.0
        return 0.5

    equity = wins / total
    if return_nut_fraction:
        nut_frac = nut_wins / wins if wins > 0 else 0.0
        return equity, nut_frac
    return equity


def compute_equity_best2_of5(
    five_cards: list[int],
    num_simulations: int = 300,
) -> float:
    """
    Preflop equity: best 2 of 5 cards vs random opponent.

    For each trial we sample one random board and one random opponent hand.
    We evaluate all 10 possible 2-card hands from our 5; we count a win if
    the best of those 10 beats the opponent on that board.

    Parameters
    ----------
    five_cards : list[int]
        Our 5 hole cards (0-26 encoding).
    num_simulations : int
        Number of random roll-outs.

    Returns
    -------
    float in [0, 1] – estimated probability of winning with best 2 of 5.
    """
    if len(five_cards) != 5:
        return 0.5
    known = set(five_cards)
    remaining = [i for i in range(DECK_SIZE) if i not in known]
    if len(remaining) < 7:
        return 0.5
    # Pre-convert all 5 to treys; build list of (i,j) pairs for best-2
    treys_5 = [int_to_treys(c) for c in five_cards]
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    wins = 0
    total = 0
    for _ in range(num_simulations):
        sample = random.sample(remaining, 7)
        opp_cards = sample[:2]
        full_board = sample[2:]
        opp_treys = [int_to_treys(c) for c in opp_cards]
        board_treys = [int_to_treys(c) for c in full_board]
        opp_rank = evaluate_hand(opp_treys, board_treys)
        best_my_rank = min(
            evaluate_hand([treys_5[i], treys_5[j]], board_treys) for i, j in pairs
        )
        if best_my_rank < opp_rank:
            wins += 1
        total += 1
    if total == 0:
        return 0.5
    return wins / total


def _rank_index(card_int: int) -> int:
    return card_int % len(RANKS)


def _straight_draw_strength_with_board(keep: list[int], board: list[int]) -> int:
    """
    Rank straight potential from kept cards + board ranks.

    Returns
    -------
    int
        0 = no straight draw
        1 = gutshot draw
        2 = open-ended draw
        3 = made straight
    """
    ranks = {_rank_index(c) for c in (keep + board)}
    straight_windows = [
        {0, 1, 2, 3, 4},
        {1, 2, 3, 4, 5},
        {2, 3, 4, 5, 6},
        {3, 4, 5, 6, 7},
        {4, 5, 6, 7, 8},
    ]
    best = 0
    for window in straight_windows:
        present = ranks & window
        if len(present) == 5:
            return 3
        if len(present) == 4:
            missing = (window - present).pop()
            lo, hi = min(window), max(window)
            if missing == lo or missing == hi:
                best = max(best, 2)  # open-ended
            else:
                best = max(best, 1)  # gutshot
    return best


def _has_straight_draw_with_board(keep: list[int], board: list[int]) -> bool:
    return _straight_draw_strength_with_board(keep, board) >= 1


def _has_pair_or_better(keep: list[int], board: list[int]) -> bool:
    """Whether current keep+board already has at least one pair."""
    rank_counts: dict[int, int] = {}
    for c in keep + board:
        r = _rank_index(c)
        rank_counts[r] = rank_counts.get(r, 0) + 1
    return any(cnt >= 2 for cnt in rank_counts.values())


def _keep_priority(keep: list[int], board: list[int]) -> int:
    """
    Priority tier for discard choice: higher = prefer this keep.
    - 5: made straight
    - 4: trips+, two pair, pocket pair (any), or board-connecting pair
    - 3: open-ended straight draw  (same tier as low board pairs — equity decides)
    - 1: gutshot straight draw only
    - 0: no draw, no pair
    """
    draw_strength = _straight_draw_strength_with_board(keep, board)

    if draw_strength == 3:
        return 5

    rank_counts: dict[int, int] = {}
    for c in keep + board:
        r = _rank_index(c)
        rank_counts[r] = rank_counts.get(r, 0) + 1

    has_trips = any(cnt >= 3 for cnt in rank_counts.values())
    has_two_pair = sum(1 for cnt in rank_counts.values() if cnt >= 2) >= 2
    has_pair = any(cnt >= 2 for cnt in rank_counts.values())

    if has_trips or has_two_pair:
        return 4

    # Pocket pairs: in a 27-card deck, any pocket pair is strong.
    # Never discard a pocket pair for a straight draw.
    if len(keep) == 2:
        r0, r1 = _rank_index(keep[0]), _rank_index(keep[1])
        if r0 == r1:
            return 4

    # Board-connecting pair (one of our cards matches a board card)
    if has_pair:
        return 4

    if draw_strength >= 2:
        return 3
    if draw_strength == 1:
        return 1
    return 0


def _is_low_pair(keep: list[int]) -> bool:
    """Treat pairs up to rank 5 as low pairs in this short deck."""
    if len(keep) != 2:
        return False
    r0 = _rank_index(keep[0])
    r1 = _rank_index(keep[1])
    return r0 == r1 and r0 <= 3


def _keep_rank_key(keep: list[int]) -> tuple[int, int]:
    """
    Deterministic tie-break key for 2-card keeps.
    Higher key means stronger raw ranks (e.g. 9-8 beats 9-3).
    """
    r = sorted((_rank_index(c) for c in keep), reverse=True)
    return r[0], r[1]


# Nut-fraction bonus: when two keeps have close equity, prefer the one
# whose wins come more often from flushes or better (nut potential).
NUT_FRAC_BONUS = 0.04  # effective equity bonus per 100% nut fraction

# Discard mixing: when top keeps are within this score margin, randomly
# select among them weighted by score to reduce predictability.
DISCARD_MIX_MARGIN = 0.02   # keeps within 2% composite score are candidates (tighter)
DISCARD_MIX_TEMPERATURE = 8.0  # higher = more weight to the best keep

def best_discard(
    hole_cards: list[int],
    community_cards: list[int],
    opp_discarded: list[int] | None = None,
    sims_per_pair: int = 800,
) -> tuple[int, int, float]:
    """
    Evaluate all C(5,2)=10 ways to keep 2 of 5 hole cards.

    Selection uses a composite score: raw equity plus a small bonus for
    nut potential (fraction of wins from flush or better).  When multiple
    keeps have near-equal scores, randomly select among them weighted by
    score to reduce predictability (opponents see our discards).

    Returns (keep_idx_1, keep_idx_2, best_equity).
    """
    candidates: list[tuple[int, int, float, float, tuple[int, int]]] = []

    for i in range(5):
        for j in range(i + 1, 5):
            keep = [hole_cards[i], hole_cards[j]]
            discards = [hole_cards[k] for k in range(5) if k != i and k != j]
            eq, nut_frac = compute_equity(
                keep,
                community_cards,
                opp_discarded=opp_discarded,
                my_discarded=discards,
                num_simulations=sims_per_pair,
                return_nut_fraction=True,
            )
            score = eq + NUT_FRAC_BONUS * nut_frac
            rank_key = _keep_rank_key(keep)
            candidates.append((i, j, eq, score, rank_key))

    if not candidates:
        return 0, 1, 0.0

    best_score = max(c[3] for c in candidates)

    # Find the deterministic best (highest score, tie-break by rank key)
    det_best = max(candidates, key=lambda c: (c[3], c[4]))

    # Safety: if the best keep is a pocket pair or has an Ace, don't mix it away
    det_keep = [hole_cards[det_best[0]], hole_cards[det_best[1]]]
    det_is_pair = (_rank_index(det_keep[0]) == _rank_index(det_keep[1]))
    det_has_ace = any(_rank_index(c) == 8 for c in det_keep)

    if det_is_pair or det_has_ace:
        return det_best[0], det_best[1], det_best[2]

    # Safety: if any candidate is a pocket pair and scores within 0.05 of best,
    # always prefer the pair (Monte Carlo noise can mask its true value).
    PAIR_PROTECTION_MARGIN = 0.05
    for i, j, eq, sc, rk in candidates:
        keep = [hole_cards[i], hole_cards[j]]
        if _rank_index(keep[0]) == _rank_index(keep[1]) and sc >= best_score - PAIR_PROTECTION_MARGIN:
            return i, j, eq

    # Gather keeps within the mix margin of the best
    finalists = [(i, j, eq, sc, rk) for i, j, eq, sc, rk in candidates
                 if sc >= best_score - DISCARD_MIX_MARGIN]

    if len(finalists) == 1:
        f = finalists[0]
        return f[0], f[1], f[2]

    # Weighted random selection among finalists
    import math
    weights = [math.exp(DISCARD_MIX_TEMPERATURE * (sc - best_score)) for _, _, _, sc, _ in finalists]
    total_w = sum(weights)
    r = random.random() * total_w
    cumul = 0.0
    for idx, (i, j, eq, sc, rk) in enumerate(finalists):
        cumul += weights[idx]
        if r <= cumul:
            return i, j, eq

    f = finalists[-1]
    return f[0], f[1], f[2]
