"""
Monte Carlo equity calculator for the 27-card poker variant.

Computes win probability by simulating random opponent hands and board run-outs,
taking into account all known/revealed cards (own cards, community cards,
opponent discards, own discards).
"""

import random
from functools import lru_cache
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

# Tournament hand-type order (docs/rules.md): SF > FH > Flush > Straight > Trips > 2P > Pair > High.
# Treys rank_class uses the same relative ordering for these (quads class 2 is impossible here).
# Composite key makes category dominate treys' 52-card absolute rank table when comparing hands.
_TOURNAMENT_TIER_FROM_TREYS_CLASS: dict[int, int] = {
    1: 1,  # straight flush
    2: 2,  # quads — impossible in 27-card; keep slot if treys ever returns it
    3: 2,  # full house
    4: 3,  # flush
    5: 4,  # straight
    6: 5,  # three of a kind
    7: 6,  # two pair
    8: 7,  # pair
    9: 8,  # high card
}


def _treys_raw_score(hand_treys: list[int], board_treys: list[int]) -> int:
    """Treys 5-card rank id (Ace straight aware). Valid input to Evaluator.get_rank_class."""
    reg = _evaluator.evaluate(hand_treys, board_treys)
    alt_hand = list(map(_ace_to_ten, hand_treys))
    alt_board = list(map(_ace_to_ten, board_treys))
    alt = _evaluator.evaluate(alt_hand, alt_board)
    return min(reg, alt)


def evaluate_short_deck_hand(hand_treys: list[int], board_treys: list[int]) -> int:
    """
    27-card tournament evaluation: Ace high/low straight handling + category-first ordering.

    Treys' raw scores are tuned for 52-card frequencies; we combine ``rank_class`` tier
    (per official hand ranking) with raw score so cross-category comparisons match the
    engine (see ``docs/rules.md`` Hand Rankings).

    **Not** a treys hand-rank id — use :func:`evaluate_hand` / :func:`_treys_raw_score`
    for :meth:`Evaluator.get_rank_class`.
    """
    raw = _treys_raw_score(hand_treys, board_treys)
    rc = _evaluator.get_rank_class(raw)
    tier = _TOURNAMENT_TIER_FROM_TREYS_CLASS.get(rc, rc)
    # Lower return value = better hand (same convention as treys raw).
    return tier * 100_000 + raw


def evaluate_hand(hand_treys: list[int], board_treys: list[int]) -> int:
    """
    Treys raw hand strength (lower = better). Ace-can-be-high (above 9) rule.

    Use this with ``Evaluator.get_rank_class`` and APIs that expect treys rank ids.
    For comparing two showdown hands under tournament category order, use
    :func:`evaluate_short_deck_hand` instead.
    """
    return _treys_raw_score(hand_treys, board_treys)


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

    For 3-4 board cards, uses treys on the visible cards (best 5 of 5 on the flop,
    best 5 of 6 on the turn). This correctly classifies made straights and flushes;
    the old rank-count-only path returned class 9 for those and allowed folding the nuts.

    Returns None if insufficient cards.
    """
    board = [c for c in community_cards if c != -1]
    if len(hole_cards) < 2 or len(board) < 3:
        return None

    if len(board) == 5:
        return get_hand_rank_class(hole_cards, community_cards)

    hand_treys = [int_to_treys(c) for c in hole_cards[:2]]
    board_treys = [int_to_treys(c) for c in board]
    rank = evaluate_hand(hand_treys, board_treys)
    return _evaluator.get_rank_class(rank)


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


def _strong_preflop_raise_hand(c1: int, c2: int) -> bool:
    """Tighter range for heavy aggression (e.g. 3-bet lines)."""
    r1, r2 = c1 % len(RANKS), c2 % len(RANKS)
    s1, s2 = c1 // len(RANKS), c2 // len(RANKS)
    if r1 == r2:
        return r1 >= 2  # 4+ pair
    if s1 == s2:
        return max(r1, r2) >= 4  # 6+ with suited
    return max(r1, r2) >= 5  # 7+ offsuit


def _shove_top15_preflop_hand(c1: int, c2: int) -> bool:
    """Approximate top ~15% of HU shoving hands (pairs, strong broadways)."""
    r1, r2 = c1 % len(RANKS), c2 % len(RANKS)
    s1, s2 = c1 // len(RANKS), c2 // len(RANKS)
    if r1 == r2:
        return r1 >= 3  # 55+
    if s1 == s2:
        return max(r1, r2) >= 5  # 7+ suited
    return max(r1, r2) >= 6  # 8+ offsuit


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
    *,
    opponent_range_bias: float = 0.0,
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
    opponent_range_bias : float
        In [0, 1]. Fraction of trials where the opponent's two cards are drawn
        from a plausible/strong raising range (vs uniform dead cards).

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

    bias = max(0.0, min(1.0, opponent_range_bias))
    use_strong = bias >= 0.55
    hand_fn = _strong_preflop_raise_hand if use_strong else _plausible_preflop_raise_hand

    for _ in range(num_simulations):
        if bias > 0.0 and random.random() < bias:
            cand: list[int] | None = None
            for _try in range(40):
                c = random.sample(remaining, sample_size)
                if hand_fn(c[0], c[1]):
                    cand = c
                    break
            if cand is None:
                cand = random.sample(remaining, sample_size)
            opp_cards = cand[:opp_needed]
            full_board = board + cand[opp_needed:]
        else:
            sample = random.sample(remaining, sample_size)
            opp_cards = sample[:opp_needed]
            full_board = board + sample[opp_needed:]

        opp_treys = [int_to_treys(c) for c in opp_cards]
        board_treys = [int_to_treys(c) for c in full_board]

        my_rank = evaluate_short_deck_hand(my_treys, board_treys)
        opp_rank = evaluate_short_deck_hand(opp_treys, board_treys)

        if my_rank < opp_rank:
            wins += 1
            if return_nut_fraction:
                rc = _evaluator.get_rank_class(evaluate_hand(my_treys, board_treys))
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

        my_rank = evaluate_short_deck_hand(my_treys, board_treys)
        opp_rank = evaluate_short_deck_hand(opp_treys, board_treys)

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

        my_rank = evaluate_short_deck_hand(my_treys, board_treys)
        opp_rank = evaluate_short_deck_hand(opp_treys, board_treys)

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

        my_rank = evaluate_short_deck_hand(my_treys, board_treys)
        opp_rank = evaluate_short_deck_hand(opp_treys, board_treys)

        if my_rank < opp_rank:
            wins += 1
        total += 1

    if total == 0:
        return 0.5
    return wins / total


def _equity_best2_of5_impl(five_cards: tuple[int, ...], num_simulations: int) -> float:
    fc = list(five_cards)
    known = set(fc)
    remaining = [i for i in range(DECK_SIZE) if i not in known]
    if len(remaining) < 7:
        return 0.5
    treys_5 = [int_to_treys(c) for c in fc]
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    wins = 0
    total = 0
    for _ in range(num_simulations):
        sample = random.sample(remaining, 7)
        opp_cards = sample[:2]
        full_board = sample[2:]
        opp_treys = [int_to_treys(c) for c in opp_cards]
        board_treys = [int_to_treys(c) for c in full_board]
        opp_rank = evaluate_short_deck_hand(opp_treys, board_treys)
        best_my_rank = min(
            evaluate_short_deck_hand([treys_5[i], treys_5[j]], board_treys) for i, j in pairs
        )
        if best_my_rank < opp_rank:
            wins += 1
        total += 1
    if total == 0:
        return 0.5
    return wins / total


@lru_cache(maxsize=100_000)
def _preflop_best2_equity_cached(cards_key: tuple[int, ...]) -> float:
    """Fixed 450-sim preflop equity; keyed by sorted 5-card set."""
    return _equity_best2_of5_impl(cards_key, 450)


def compute_equity_best2_of5(
    five_cards: list[int],
    num_simulations: int = 300,
) -> float:
    """
    Preflop equity: best 2 of 5 cards vs random opponent.

    Hot path uses an LRU cache (450 simulations) so repeated preflop spots
    do not re-run Monte Carlo. ``num_simulations`` is only used when it
    differs from the cached canonical run (rare / testing).
    """
    if len(five_cards) != 5:
        return 0.5
    key = tuple(sorted(five_cards))
    if num_simulations == 450:
        return _preflop_best2_equity_cached(key)
    return _equity_best2_of5_impl(key, num_simulations)


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
        opp_rank = evaluate_short_deck_hand(opp_treys, board_treys)
        best_my_rank = min(
            evaluate_short_deck_hand([treys_5[i], treys_5[j]], board_treys) for i, j in pairs
        )
        if best_my_rank < opp_rank:
            wins += 1
        total += 1
    if total == 0:
        return 0.5
    return wins / total


def compute_equity_best2_of5_vs_shove_top15(
    five_cards: list[int],
    num_simulations: int = 300,
    *,
    max_resample: int = 40,
) -> float:
    """
    Best 2 of 5 vs opponent on a narrow shoving range (~top 15% of hands).

    Used with pot odds when facing large preflop jams (replaces linear per-chip penalties).
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
        opp_cards: list[int] | None = None
        board_sample: list[int] = []
        for _try in range(max_resample):
            cand = random.sample(remaining, 7)
            o0, o1 = cand[0], cand[1]
            if _shove_top15_preflop_hand(o0, o1):
                opp_cards = [o0, o1]
                board_sample = cand[2:]
                break
        if opp_cards is None:
            cand = random.sample(remaining, 7)
            opp_cards = cand[:2]
            board_sample = cand[2:]
        opp_treys = [int_to_treys(c) for c in opp_cards]
        board_treys = [int_to_treys(c) for c in board_sample]
        opp_rank = evaluate_short_deck_hand(opp_treys, board_treys)
        best_my_rank = min(
            evaluate_short_deck_hand([treys_5[i], treys_5[j]], board_treys) for i, j in pairs
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
