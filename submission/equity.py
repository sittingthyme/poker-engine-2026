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


# ---------------------------------------------------------------------------
# Core equity function
# ---------------------------------------------------------------------------

def compute_equity(
    my_cards: list[int],
    community_cards: list[int],
    opp_discarded: list[int] | None = None,
    my_discarded: list[int] | None = None,
    num_simulations: int = 300,
) -> float:
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

    Returns
    -------
    float  in [0, 1]  –  estimated probability of winning.
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
        return 0.5  # not enough unknowns to simulate

    # Pre-convert our cards once
    my_treys = [int_to_treys(c) for c in my_cards]

    wins = 0
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
        total += 1

    if total == 0:
        return 0.5
    return wins / total


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
    - 3: open-ended straight draw or made straight (prioritize over everything)
    - 2: trips, pair, or two-pair (prioritize over gutshot)
    - 1: gutshot straight draw only
    - 0: no draw, no pair
    """
    draw_strength = _straight_draw_strength_with_board(keep, board)
    has_pair = _has_pair_or_better(keep, board)
    if draw_strength >= 2:  # open-ended or made straight
        return 3
    if has_pair:  # trips, pair, two-pair
        return 2
    if draw_strength == 1:  # gutshot only
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


def best_discard(
    hole_cards: list[int],
    community_cards: list[int],
    opp_discarded: list[int] | None = None,
    sims_per_pair: int = 550,
) -> tuple[int, int, float]:
    """
    Evaluate all C(5,2)=10 ways to keep 2 of 5 hole cards.
    Favors straight draws over low pairs in 27-card deck.

    Returns (keep_idx_1, keep_idx_2, best_equity).
    """
    board = [c for c in community_cards if c != -1]
    best_i, best_j = 0, 1
    best_eq = -1.0
    best_priority = -1
    best_has_pair_or_better = False
    best_is_low_pair = False
    best_rank_key = (-1, -1)

    for i in range(5):
        for j in range(i + 1, 5):
            keep = [hole_cards[i], hole_cards[j]]
            discards = [hole_cards[k] for k in range(5) if k != i and k != j]
            eq = compute_equity(
                keep,
                community_cards,
                opp_discarded=opp_discarded,
                my_discarded=discards,
                num_simulations=sims_per_pair,
            )
            priority = _keep_priority(keep, board)
            has_pair_or_better = _has_pair_or_better(keep, board)
            is_low = _is_low_pair(keep)
            rank_key = _keep_rank_key(keep)

            # Priority order: open-ended/made straight (3) > trips/pair/two-pair (2) > gutshot (1) > nothing (0).
            if priority > best_priority:
                best_eq = eq
                best_i, best_j = i, j
                best_priority = priority
                best_has_pair_or_better = has_pair_or_better
                best_is_low_pair = is_low
                best_rank_key = rank_key
                continue
            if priority < best_priority:
                continue

            # Same priority tier: prefer higher equity.
            if eq > best_eq + 1e-9:
                best_eq = eq
                best_i, best_j = i, j
                best_has_pair_or_better = has_pair_or_better
                best_is_low_pair = is_low
                best_rank_key = rank_key
                continue

            # Tie-break for near-equal equity: higher rank keep (e.g. 9-8 > 9-3).
            if abs(eq - best_eq) <= 0.020 and rank_key > best_rank_key:
                best_eq = eq
                best_i, best_j = i, j
                best_has_pair_or_better = has_pair_or_better
                best_is_low_pair = is_low
                best_rank_key = rank_key

    return best_i, best_j, best_eq
