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


def compute_equity_vs_flush_draw(
    my_cards: list[int],
    community_cards: list[int],
    flush_suit: int,
    opp_discarded: list[int] | None = None,
    my_discarded: list[int] | None = None,
    num_simulations: int = 300,
) -> float:
    """
    Equity vs opponent who has 2 cards of flush_suit (flush draw).
    Used when discard signals indicate opponent likely kept suited cards.

    Parameters
    ----------
    my_cards : list[int]
        Our hole cards (2 cards).
    community_cards : list[int]
        Visible board cards.
    flush_suit : int
        Suit index (0=d, 1=h, 2=s). Opponent has 2 cards of this suit.
    opp_discarded, my_discarded : list[int] | None
        Known discards.
    num_simulations : int
        Number of Monte Carlo trials.

    Returns
    -------
    float in [0, 1] – estimated win rate vs flush-draw opponent.
    """
    if opp_discarded is None:
        opp_discarded = []
    if my_discarded is None:
        my_discarded = []

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
    suit_cards = [c for c in range(flush_suit * 9, flush_suit * 9 + 9) if c not in known]
    board_needed = 5 - len(board)

    if len(suit_cards) < 2 or len(remaining) < 2 + board_needed:
        return 0.5

    my_treys = [int_to_treys(c) for c in my_cards]
    wins = 0
    total = 0

    for _ in range(num_simulations):
        opp_cards = random.sample(suit_cards, 2)
        opp_set = set(opp_cards)
        remaining_for_board = [c for c in remaining if c not in opp_set]
        if len(remaining_for_board) < board_needed:
            continue
        board_completion = random.sample(remaining_for_board, board_needed)
        full_board = board + board_completion

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


def compute_equity_vs_board_pair(
    my_cards: list[int],
    community_cards: list[int],
    opp_discarded: list[int] | None = None,
    my_discarded: list[int] | None = None,
    num_simulations: int = 300,
) -> float:
    """
    Equity vs opponent who has at least one card matching a board rank.
    Used when discard signals indicate opponent likely paired the board.

    Sampling: pick one card from remaining board-rank cards, then one
    from the rest of the deck.
    """
    if opp_discarded is None:
        opp_discarded = []
    if my_discarded is None:
        my_discarded = []

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

    board_ranks = {c % 9 for c in board}
    remaining = [i for i in range(DECK_SIZE) if i not in known]
    board_rank_cards = [c for c in remaining if c % 9 in board_ranks]
    other_cards = [c for c in remaining if c % 9 not in board_ranks]
    board_needed = 5 - len(board)

    if not board_rank_cards or len(remaining) < 2 + board_needed:
        return 0.5

    my_treys = [int_to_treys(c) for c in my_cards]
    wins = 0
    total = 0

    for _ in range(num_simulations):
        c1 = random.choice(board_rank_cards)
        pool = [c for c in remaining if c != c1]
        if not pool:
            continue
        c2 = random.choice(pool)
        opp_cards = [c1, c2]
        opp_set = set(opp_cards)
        remaining_for_board = [c for c in remaining if c not in opp_set]
        if len(remaining_for_board) < board_needed:
            continue
        board_completion = random.sample(remaining_for_board, board_needed)
        full_board = board + board_completion

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


def compute_equity_vs_straight_draw(
    my_cards: list[int],
    community_cards: list[int],
    straight_ranks: set[int],
    opp_discarded: list[int] | None = None,
    my_discarded: list[int] | None = None,
    num_simulations: int = 300,
) -> float:
    """
    Equity vs opponent who has at least one card in straight_ranks.
    Used when discard signals indicate opponent likely kept straight connectors.

    Sampling: pick one card from remaining straight-rank cards, then one
    from the rest of the deck.
    """
    if opp_discarded is None:
        opp_discarded = []
    if my_discarded is None:
        my_discarded = []

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
    straight_cards = [c for c in remaining if c % 9 in straight_ranks]
    board_needed = 5 - len(board)

    if not straight_cards or len(remaining) < 2 + board_needed:
        return 0.5

    my_treys = [int_to_treys(c) for c in my_cards]
    wins = 0
    total = 0

    for _ in range(num_simulations):
        c1 = random.choice(straight_cards)
        pool = [c for c in remaining if c != c1]
        if not pool:
            continue
        c2 = random.choice(pool)
        opp_cards = [c1, c2]
        opp_set = set(opp_cards)
        remaining_for_board = [c for c in remaining if c not in opp_set]
        if len(remaining_for_board) < board_needed:
            continue
        board_completion = random.sample(remaining_for_board, board_needed)
        full_board = board + board_completion

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


def _plausible_preflop_raise_hand(c1: int, c2: int) -> bool:
    """
    Crude filter: hands that often open/raise in short-deck HU (not full GTO).

    Keeps pairs, suited cards, or at least one high card (6+ in 23456789A).
    """
    r1, r2 = c1 % len(RANKS), c2 % len(RANKS)
    s1, s2 = c1 // len(RANKS), c2 // len(RANKS)
    if r1 == r2:
        return True
    if s1 == s2:
        return True
    if max(r1, r2) >= 4:  # 6 or better
        return True
    return False


def compute_equity_best2_of5_vs_raise_shape(
    five_cards: list[int],
    num_simulations: int = 300,
    *,
    max_resample: int = 35,
) -> float:
    """
    Like compute_equity_best2_of5, but the opponent's two cards are drawn from a
    biased distribution: only hands that pass _plausible_preflop_raise_hand,
    with rejection sampling (falls back to uniform if sampling fails).

    Use when facing a preflop raise: raw best-2-of-5 vs uniform villain
    overestimates how often we are ahead.
    """
    if len(five_cards) != 5:
        return 0.5
    known = set(five_cards)
    remaining = [i for i in range(DECK_SIZE) if i not in known]
    if len(remaining) < 7:
        return 0.5
    treys_5 = [int_to_treys(c) for c in five_cards]
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    wins = 0
    total = 0
    for _ in range(num_simulations):
        rest = list(remaining)
        opp_cards: list[int] | None = None
        board_sample: list[int] = []
        for _try in range(max_resample):
            cand = random.sample(rest, 7)
            o0, o1 = cand[0], cand[1]
            if _plausible_preflop_raise_hand(o0, o1):
                opp_cards = [o0, o1]
                board_sample = cand[2:]
                break
        if opp_cards is None:
            cand = random.sample(rest, 7)
            opp_cards = cand[:2]
            board_sample = cand[2:]
        opp_treys = [int_to_treys(c) for c in opp_cards]
        board_treys = [int_to_treys(c) for c in board_sample]
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


def _get_suit(card_int: int) -> int:
    return card_int // len(RANKS)


def _flush_draw_strength(keep: list[int], board: list[int]) -> tuple[int, bool]:
    """
    In a 27-card deck (9 per suit), flush potential from keep + board.
    Returns (base_strength, is_double_suited):
      base: 0 = none, 1 = 3 of suit, 2 = 4 of suit, 3 = made flush
      is_double_suited: both keep cards are in the flush suit
    """
    suits = [_get_suit(c) for c in (keep + board)]
    if not suits:
        return 0, False
    best_suit = max(set(suits), key=lambda s: suits.count(s))
    max_count = suits.count(best_suit)
    base = 0
    if max_count >= 5:
        base = 3
    elif max_count >= 4:
        base = 2
    elif max_count >= 3:
        base = 1

    keep_suits = [_get_suit(c) for c in keep]
    is_double = base >= 1 and len(keep) == 2 and keep_suits[0] == best_suit and keep_suits[1] == best_suit
    return base, is_double


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
# Flush-draw bonus: prefer keeps with flush draws when equity is close.
# In a 3-suit deck, flush draws are valuable; Monte Carlo variance can
# miss this with limited sims.
FLUSH_DRAW_BONUS = 0.02  # per level (1=3 of suit, 2=4 of suit, 3=made)
DOUBLE_SUITED_BONUS = 0.04  # extra when both keep cards are in the flush suit

def best_discard(
    hole_cards: list[int],
    community_cards: list[int],
    opp_discarded: list[int] | None = None,
    sims_per_pair: int = 800,
) -> tuple[int, int, float]:
    """
    Evaluate all C(5,2)=10 ways to keep 2 of 5 hole cards.

    Selection uses a composite score: raw equity plus a small bonus for
    nut potential (fraction of wins from flush or better).  This favors
    keeps that win big pots rather than keeps that win marginally.

    Returns (keep_idx_1, keep_idx_2, best_equity).
    """
    best_i, best_j = 0, 1
    best_score = -1.0
    best_eq = -1.0
    best_rank_key = (-1, -1)
    board = [c for c in community_cards if c != -1]

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
            flush_base, is_double_suited = _flush_draw_strength(keep, board)
            flush_bonus = FLUSH_DRAW_BONUS * flush_base + (DOUBLE_SUITED_BONUS if is_double_suited else 0)
            score = eq + NUT_FRAC_BONUS * nut_frac + flush_bonus
            rank_key = _keep_rank_key(keep)

            if score > best_score + 0.01:
                best_score = score
                best_eq = eq
                best_i, best_j = i, j
                best_rank_key = rank_key
            elif abs(score - best_score) <= 0.01 and rank_key > best_rank_key:
                best_score = score
                best_eq = eq
                best_i, best_j = i, j
                best_rank_key = rank_key

    return best_i, best_j, best_eq
