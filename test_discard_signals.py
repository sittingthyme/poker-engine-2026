"""
Test accuracy of opponent discard signal inference.

Simulates 10,000 random hands: deal 5 cards to opponent + 3–5 board cards,
opponent keeps best 2 (using our equity engine), we see the 3 discards.
Then we run analyze_opponent_discards and check each signal against ground truth.
"""

import random
import sys
from collections import Counter
from submission.opponent_range import analyze_opponent_discards, DiscardSignals
from submission.equity import (
    int_to_treys, evaluate_short_deck_hand, _flush_draw_strength,
    _straight_draw_strength_with_board, _rank_index,
)

RANKS = "23456789A"
SUITS = "dhs"
DECK_SIZE = 27
NUM_TRIALS = 10_000


def _rank(c: int) -> int:
    return c % 9


def _suit(c: int) -> int:
    return c // 9


def _card_str(c: int) -> str:
    return RANKS[c % 9] + SUITS[c // 9]


def _best_keep(hole5: list[int], board: list[int]) -> tuple[list[int], list[int]]:
    """Pick best 2 of 5 by quick equity (fewer sims for speed)."""
    best_pair = (0, 1)
    best_rank = 999999

    if len(board) < 5:
        remaining = [i for i in range(DECK_SIZE) if i not in set(hole5) | set(board)]
        need = 5 - len(board)
        if len(remaining) < need + 2:
            return hole5[:2], hole5[2:]
        sample_boards = []
        for _ in range(30):
            s = random.sample(remaining, need)
            sample_boards.append(board + s)
    else:
        sample_boards = [board]

    for i in range(5):
        for j in range(i + 1, 5):
            hand_treys = [int_to_treys(hole5[i]), int_to_treys(hole5[j])]
            total_rank = 0
            for sb in sample_boards:
                board_treys = [int_to_treys(c) for c in sb[:5]]
                total_rank += evaluate_short_deck_hand(hand_treys, board_treys)
            if total_rank < best_rank:
                best_rank = total_rank
                best_pair = (i, j)

    keep = [hole5[best_pair[0]], hole5[best_pair[1]]]
    discards = [hole5[k] for k in range(5) if k not in best_pair]
    return keep, discards


def _kept_has_pair(keep: list[int], board: list[int]) -> bool:
    all_ranks = [_rank(c) for c in keep + board]
    counts = Counter(all_ranks)
    return any(cnt >= 2 for cnt in counts.values())


def _kept_has_flush_draw(keep: list[int], board: list[int]) -> bool:
    all_suits = [_suit(c) for c in keep + board]
    counts = Counter(all_suits)
    return any(cnt >= 3 for cnt in counts.values())


def _kept_has_straight_draw(keep: list[int], board: list[int]) -> bool:
    return _straight_draw_strength_with_board(keep, board) >= 1


def _kept_both_suited(keep: list[int]) -> bool:
    return len(keep) == 2 and _suit(keep[0]) == _suit(keep[1])


def _kept_has_board_pair(keep: list[int], board: list[int]) -> bool:
    """At least one kept card matches a board rank."""
    board_ranks = {_rank(c) for c in board}
    return any(_rank(c) in board_ranks for c in keep)


def run_test():
    random.seed(42)

    # Counters: (signal_true, actual_true, both_true, total)
    metrics = {
        "flush_signal>=1": [0, 0, 0, 0],
        "flush_signal>=2": [0, 0, 0, 0],
        "straight_signal>=1": [0, 0, 0, 0],
        "straight_signal>=2": [0, 0, 0, 0],
        "opp_discarded_pair": [0, 0, 0, 0],
        "opp_likely_has_pair": [0, 0, 0, 0],
        "opp_kept_high_cards": [0, 0, 0, 0],
        "discarded_pair→stronger": [0, 0, 0, 0],
    }

    for trial in range(NUM_TRIALS):
        deck = list(range(DECK_SIZE))
        random.shuffle(deck)

        # Deal: 5 to opponent, 3-5 for board
        board_size = random.choice([3, 4, 5])
        opp_hole = deck[:5]
        board = deck[5:5 + board_size]

        keep, discards = _best_keep(opp_hole, board)
        sig = analyze_opponent_discards(discards, board)

        # Ground truth for kept hand
        gt_has_flush = _kept_has_flush_draw(keep, board)
        gt_has_straight = _kept_has_straight_draw(keep, board)
        gt_has_pair = _kept_has_pair(keep, board)
        gt_has_board_pair = _kept_has_board_pair(keep, board)
        gt_both_suited = _kept_both_suited(keep)
        gt_high = sum(_rank(c) for c in keep) / 2.0 >= 5.5
        gt_discards_have_pair = len(set(_rank(c) for c in discards)) < 3

        # -- flush_signal >= 1 (opponent may have flush draw) --
        m = metrics["flush_signal>=1"]
        m[3] += 1
        if sig.opp_flush_signal >= 1:
            m[0] += 1
        if gt_has_flush and gt_both_suited:
            m[1] += 1
        if sig.opp_flush_signal >= 1 and gt_has_flush and gt_both_suited:
            m[2] += 1

        # -- flush_signal >= 2 (opponent likely has flush draw) --
        m = metrics["flush_signal>=2"]
        m[3] += 1
        if sig.opp_flush_signal >= 2:
            m[0] += 1
        if gt_has_flush and gt_both_suited:
            m[1] += 1
        if sig.opp_flush_signal >= 2 and gt_has_flush and gt_both_suited:
            m[2] += 1

        # -- straight_signal >= 1 --
        m = metrics["straight_signal>=1"]
        m[3] += 1
        if sig.opp_straight_signal >= 1:
            m[0] += 1
        if gt_has_straight:
            m[1] += 1
        if sig.opp_straight_signal >= 1 and gt_has_straight:
            m[2] += 1

        # -- straight_signal >= 2 --
        m = metrics["straight_signal>=2"]
        m[3] += 1
        if sig.opp_straight_signal >= 2:
            m[0] += 1
        if gt_has_straight:
            m[1] += 1
        if sig.opp_straight_signal >= 2 and gt_has_straight:
            m[2] += 1

        # -- opp_discarded_pair --
        m = metrics["opp_discarded_pair"]
        m[3] += 1
        if sig.opp_discarded_pair:
            m[0] += 1
        if gt_discards_have_pair:
            m[1] += 1
        if sig.opp_discarded_pair and gt_discards_have_pair:
            m[2] += 1

        # -- opp_likely_has_pair --
        m = metrics["opp_likely_has_pair"]
        m[3] += 1
        if sig.opp_likely_has_pair:
            m[0] += 1
        if gt_has_board_pair:
            m[1] += 1
        if sig.opp_likely_has_pair and gt_has_board_pair:
            m[2] += 1

        # -- opp_kept_high_cards --
        m = metrics["opp_kept_high_cards"]
        m[3] += 1
        if sig.opp_kept_high_cards:
            m[0] += 1
        if gt_high:
            m[1] += 1
        if sig.opp_kept_high_cards and gt_high:
            m[2] += 1

        # -- discarded_pair → kept hand is stronger (trips, two pair, higher pair) --
        # When opponent discards a pair, they likely kept something better
        m = metrics["discarded_pair→stronger"]
        m[3] += 1
        if sig.opp_discarded_pair:
            m[0] += 1
        keep_rank_counts = Counter(_rank(c) for c in keep + board)
        gt_stronger = any(cnt >= 3 for cnt in keep_rank_counts.values()) or \
                      sum(1 for cnt in keep_rank_counts.values() if cnt >= 2) >= 2
        if gt_stronger:
            m[1] += 1
        if sig.opp_discarded_pair and gt_stronger:
            m[2] += 1

    # Print results
    print(f"\n{'Signal':<30} {'Predicted':>10} {'Actual':>10} {'Precision':>10} {'Recall':>10} {'Lift':>10} {'Base Rate':>10}")
    print("-" * 92)
    for name, (predicted, actual, both, total) in metrics.items():
        precision = both / predicted if predicted > 0 else 0.0
        recall = both / actual if actual > 0 else 0.0
        base_rate = actual / total if total > 0 else 0.0
        lift = (precision / base_rate) if base_rate > 0 else 0.0
        print(f"{name:<30} {predicted:>10} {actual:>10} {precision:>10.1%} {recall:>10.1%} {lift:>10.2f}x {base_rate:>10.1%}")

    print(f"\nTotal trials: {NUM_TRIALS}")
    print("\nLift = Precision / Base Rate (>1 = better than random guess)")


if __name__ == "__main__":
    run_test()
