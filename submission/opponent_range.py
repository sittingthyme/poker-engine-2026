"""
Bayesian range inference from opponent discards.

When opponent discards 3 cards, infer their likely hand strength from
discard quality and structured pattern recognition (pairs, straights,
flush draws) relative to the board.

Card encoding: card_int in [0, 27), rank = card_int % 9, suit = card_int // 9
RANKS = "23456789A", SUITS = "dhs" (3 suits, 9 cards each)
"""

from __future__ import annotations

from dataclasses import dataclass, field

_RANKS = "23456789A"
_NUM_RANKS = 9


def _rank(c: int) -> int:
    return c % _NUM_RANKS


def _suit(c: int) -> int:
    return c // _NUM_RANKS


# ---------------------------------------------------------------------------
# Structured discard signals
# ---------------------------------------------------------------------------

@dataclass
class DiscardSignals:
    """Structured signals inferred from opponent's 3 discards + board."""

    # Flush signals (0 = no signal, 1 = possible, 2 = likely)
    opp_flush_signal: int = 0
    flush_suit: int = -1                # which suit is the flush danger suit

    # Straight signals (0 = no signal, 1 = possible, 2 = likely)
    opp_straight_signal: int = 0
    straight_helping_ranks: set = field(default_factory=set)  # ranks that help complete a straight

    # Pair signals
    opp_discarded_pair: bool = False    # a pair among the 3 discards
    opp_likely_has_pair: bool = False   # kept a card matching a board rank

    # Strength signals
    opp_kept_high_cards: bool = False   # discarded low cards → kept high
    discard_quality: float = 0.5        # 0-1, higher = discarded strong cards


def analyze_opponent_discards(
    opp_discards: list[int],
    board: list[int],
) -> DiscardSignals:
    """
    Analyze opponent's 3 revealed discards relative to the current board.

    Parameters
    ----------
    opp_discards : list[int]  – 3 discarded cards (0-26 encoding)
    board        : list[int]  – visible community cards (filter out -1 before calling)
    """
    sig = DiscardSignals()

    discards = [c for c in opp_discards if c != -1]
    if len(discards) != 3:
        return sig

    d_ranks = [_rank(c) for c in discards]
    d_suits = [_suit(c) for c in discards]
    b_ranks = [_rank(c) for c in board]
    b_suits = [_suit(c) for c in board]

    # --- Discard quality (generic) ---
    rank_sum = sum(d_ranks) / (3 * 8.0)
    suited_bonus = 1.0 if len(set(d_suits)) < 3 else 0.0
    d_sorted = sorted(d_ranks)
    gap = d_sorted[-1] - d_sorted[0]
    connected_bonus = 1.0 if gap <= 2 else 0.0
    sig.discard_quality = 0.6 * rank_sum + 0.2 * suited_bonus + 0.2 * connected_bonus

    # --- Pair in discards ---
    from collections import Counter
    d_rank_counts = Counter(d_ranks)
    sig.opp_discarded_pair = any(cnt >= 2 for cnt in d_rank_counts.values())

    # --- Opponent likely has a pair (kept card matching board rank) ---
    board_rank_set = set(b_ranks)
    discarded_board_matches = sum(1 for r in d_ranks if r in board_rank_set)
    if board_rank_set and discarded_board_matches == 0:
        sig.opp_likely_has_pair = True

    # --- Flush signals (board-relative) ---
    # In a 3-suit deck, flush draws are very common and powerful.
    # Key insight: if the board has N cards of suit S, and the opponent
    # discarded 0 of suit S, they likely kept 2 of suit S (flush draw).
    board_suit_counts: dict[int, int] = {}
    for s in b_suits:
        board_suit_counts[s] = board_suit_counts.get(s, 0) + 1

    danger_suit = -1
    for s, cnt in board_suit_counts.items():
        if cnt >= 2:
            if danger_suit == -1 or cnt > board_suit_counts.get(danger_suit, 0):
                danger_suit = s

    if danger_suit >= 0:
        sig.flush_suit = danger_suit
        danger_in_discards = sum(1 for s in d_suits if s == danger_suit)
        board_danger_count = board_suit_counts[danger_suit]

        if danger_in_discards == 0:
            if board_danger_count >= 3:
                sig.opp_flush_signal = 2  # very likely kept suited
            else:
                sig.opp_flush_signal = 1  # possible
        # If they discarded 2+ of the danger suit, they gave up on flush
        # (not very useful on its own, but good combined with other signals)

    # --- Straight signals ---
    # Check if kept cards likely connect to the board for a straight.
    # Two complementary approaches:
    #  1. Board is connected AND discards don't contain straight-helpers
    #  2. Discards are all far from board ranks (opponent kept connectors)
    if len(board) >= 3:
        board_sorted = sorted(set(b_ranks))
        board_span = board_sorted[-1] - board_sorted[0]

        # Ranks that help make a straight with the board
        straight_helping_ranks = set()
        for start in range(max(0, min(b_ranks) - 4), min(_NUM_RANKS - 4, max(b_ranks)) + 1):
            window = set(range(start, start + 5))
            board_in_window = set(b_ranks) & window
            if len(board_in_window) >= 2:
                straight_helping_ranks |= (window - set(b_ranks))
        straight_helping_ranks = {r for r in straight_helping_ranks if 0 <= r < _NUM_RANKS}

        if straight_helping_ranks:
            sig.straight_helping_ranks = straight_helping_ranks
            helpers_discarded = sum(1 for r in d_ranks if r in straight_helping_ranks)
            if helpers_discarded == 0 and len(straight_helping_ranks) >= 2:
                sig.opp_straight_signal = 2
            elif helpers_discarded <= 1 and board_span <= 4:
                sig.opp_straight_signal = max(sig.opp_straight_signal, 1)

    # --- High/low card signals ---
    avg_discard_rank = sum(d_ranks) / 3.0
    max_discard_rank = max(d_ranks)
    if avg_discard_rank <= 2.0 and max_discard_rank <= 4:
        sig.opp_kept_high_cards = True

    return sig


# ---------------------------------------------------------------------------
# Legacy quality score (kept for backward compatibility)
# ---------------------------------------------------------------------------

def discard_quality(discards: list[int]) -> float:
    """
    Quality of 3 discarded cards (0-1).
    Higher = they discarded strong cards -> likely kept weak.
    Lower = they discarded weak cards -> likely kept strong.
    """
    if len(discards) != 3:
        return 0.5
    ranks = [c % 9 for c in discards]
    rank_sum = sum(ranks) / (3 * 8.0)
    suits = [c // 9 for c in discards]
    suited = 1.0 if len(set(suits)) < 3 else 0.0
    ranks_sorted = sorted(ranks)
    gap = max(ranks_sorted) - min(ranks_sorted)
    connected = 1.0 if gap <= 2 else 0.0
    return 0.6 * rank_sum + 0.2 * suited + 0.2 * connected


class OpponentRangeModel:
    """
    Tracks opponent range strength inferred from discard patterns.
    Per-hand: when we see their 3 discards, compute strength and confidence.
    """

    def __init__(self) -> None:
        self._current_strength: float = 0.5
        self._current_confidence: float = 0.0
        self._history: list[float] = []

    def update_from_discards(
        self,
        discards: list[int],
        preflop_action_strength: float = 0.5,
    ) -> None:
        """
        Call when we see opponent's 3 discards (e.g. after discard phase).

        ``preflop_action_strength`` in [0, 1]: higher = they showed more aggression
        pre-flop (opens, 3-bets). Blended with discard quality so a 3-bet line that
        dumps high cards still reads as a strong kept range (e.g. pairs), not a
        sudden switch to trash.
        """
        if len(discards) != 3:
            return
        quality = discard_quality(discards)
        self._history.append(quality)
        pa = max(0.0, min(1.0, preflop_action_strength))
        self._current_strength = (pa * 0.7) + ((1.0 - quality) * 0.3)
        self._current_confidence = min(1.0, len(self._history) * 0.15)

    def new_hand(self) -> None:
        """Reset per-hand state; history persists for confidence."""
        self._current_strength = 0.5
        self._current_confidence = 0.0

    @property
    def range_strength(self) -> float:
        """0-1: higher = opponent range is stronger."""
        return self._current_strength

    @property
    def range_confidence(self) -> float:
        """0-1: how confident we are in the inference."""
        return self._current_confidence
