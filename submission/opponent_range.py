"""
Bayesian range inference from opponent discards.

When opponent discards 3 cards, infer their likely hand strength from
discard quality: low-quality discards (rags) -> they kept strong; high-quality
discards -> they kept weak.
"""

from __future__ import annotations


def discard_quality(discards: list[int]) -> float:
    """
    Quality of 3 discarded cards (0-1).
    Higher = they discarded strong cards -> likely kept weak.
    Lower = they discarded weak cards -> likely kept strong.
    Uses rank (0-8 for 2-A) and simple suitedness/connectivity.
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

    def update_from_discards(self, discards: list[int]) -> None:
        """
        Call when we see opponent's 3 discards (e.g. after discard phase).
        Low quality -> strong range; high quality -> weak range.
        """
        if len(discards) != 3:
            return
        quality = discard_quality(discards)
        self._history.append(quality)
        # Range strength: 1 - quality (low quality discard = strong kept hand)
        self._current_strength = 1.0 - quality
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
