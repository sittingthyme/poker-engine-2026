#!/usr/bin/env python3
"""Summarize match.csv for *strategy* review vs official chip accounting.

Many matches end with a long **preflop-only** suffix (no flop dealt): e.g. folding
every hand once you are far enough ahead to **lock the win**. That segment is
correct for the tournament score but **must not** drive fold rates, VPIP, or
per-hand EV interpretation.

This script detects a longest suffix of hands with no Flop/Turn/River rows and
treats **everything before that** as the default **analysis segment**. Full-log
stats are optional (--full-log).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

POSTFLOP = {"Flop", "Turn", "River"}


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        first = f.readline()
        if not first.lstrip().startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def streets_by_hand(rows: list[dict[str, str]]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        out[int(r["hand_number"])].add(r["street"])
    return dict(out)


def preflop_only_tail_len(streets_map: dict[int, set[str]], num_hands: int) -> int:
    """Longest suffix of hands with no Flop/Turn/River rows in the log."""
    n = 0
    for h in range(num_hands - 1, -1, -1):
        st = streets_map.get(h, set())
        if st & POSTFLOP:
            break
        n += 1
    return n


def fold_rate_stats(
    rows: list[dict[str, str]],
) -> tuple[dict[int, Counter[str]], dict[int, Counter[str]]]:
    pf_pre = {0: Counter(), 1: Counter()}
    pf_post = {0: Counter(), 1: Counter()}
    for r in rows:
        act = r["action_type"]
        if act == "DISCARD":
            continue
        at = int(r["active_team"])
        st = r["street"]
        if st == "Pre-Flop":
            pf_pre[at][act] += 1
        elif st in POSTFLOP:
            pf_post[at][act] += 1
    return pf_pre, pf_post


def action_aggregates(
    rows: list[dict[str, str]],
) -> tuple[Counter[tuple[int, str]], dict[tuple[int, str], Counter[str]]]:
    by_team: Counter[tuple[int, str]] = Counter()
    by_team_street: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    for r in rows:
        at = int(r["active_team"])
        st = r["street"]
        act = r["action_type"]
        if act == "DISCARD":
            continue
        by_team_street[(at, st)][act] += 1
        by_team[at, act] += 1
    return by_team, by_team_street


def print_fold_block(pf_pre: dict[int, Counter], pf_post: dict[int, Counter], label: str) -> None:
    print(f"\n=== FOLD RATE ({label}) ===")
    for team, name in [(0, "PlayerAgent"), (1, "ProbabilityAgent")]:
        c = pf_pre[team]
        denom = c["FOLD"] + c["CALL"] + c["RAISE"] + c.get("CHECK", 0)
        fr = c["FOLD"] / denom if denom else 0.0
        print(f"  Preflop {name}: FOLD {c['FOLD']}/{denom} = {fr:.1%}")
    for team, name in [(0, "PlayerAgent"), (1, "ProbabilityAgent")]:
        c = pf_post[team]
        denom = c["FOLD"] + c["CALL"] + c["RAISE"] + c["CHECK"]
        fr = c["FOLD"] / denom if denom else 0.0
        print(f"  Postflop {name}: FOLD {c['FOLD']}/{denom} = {fr:.1%}")


def print_action_sections(
    by_team: Counter[tuple[int, str]],
    by_team_street: dict[tuple[int, str], Counter[str]],
    label: str,
) -> None:
    print(f"\n=== ACTIONS (excluding DISCARD) — {label} ===")
    for team, name in [(0, "PlayerAgent"), (1, "ProbabilityAgent")]:
        print(f"\n--- {name} (Team {team}) ---")
        sub = Counter()
        for (t, act), n in by_team.items():
            if t == team:
                sub[act] += n
        for act, n in sub.most_common():
            print(f"  {act:8s} {n:6d}")

    print(f"\n=== STREET × ACTION (Team 0 / Team 1) — {label} ===")
    for st in ["Pre-Flop", "Flop", "Turn", "River"]:
        print(f"\n{st}:")
        for team, label_t in [(0, "T0"), (1, "T1")]:
            c = by_team_street[(team, st)]
            if not c:
                continue
            tot = sum(c.values())
            parts = ", ".join(f"{a}={c[a]}" for a in sorted(c.keys()))
            print(f"  {label_t}: {parts}  (n={tot})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "csv_path",
        nargs="?",
        default="match.csv",
        type=Path,
        help="Path to match log (default: match.csv)",
    )
    ap.add_argument(
        "--full-log",
        action="store_true",
        help="Also print duplicate sections for the entire CSV (includes lockout tail).",
    )
    args = ap.parse_args()
    path = args.csv_path
    rows = load_rows(path)
    if not rows:
        raise SystemExit("No data rows")

    last = rows[-1]
    br0_end = int(float(last["team_0_bankroll"]))
    br1_end = int(float(last["team_1_bankroll"]))
    hands = max(int(r["hand_number"]) for r in rows) + 1

    hand_end: dict[int, tuple[int, int]] = {}
    for r in rows:
        h = int(r["hand_number"])
        hand_end[h] = (int(float(r["team_0_bankroll"])), int(float(r["team_1_bankroll"])))

    smap = streets_by_hand(rows)
    tail = preflop_only_tail_len(smap, hands)
    first_tail_h = hands - tail if tail else hands

    rows_analysis = (
        [r for r in rows if int(r["hand_number"]) < first_tail_h]
        if tail > 0
        else rows
    )
    analysis_hands = first_tail_h if tail > 0 else hands
    if first_tail_h > 0:
        br0_analysis, br1_analysis = hand_end[first_tail_h - 1]
    else:
        # Entire log is preflop-only suffix (degenerate)
        br0_analysis = br1_analysis = 0

    by_team_a, by_team_street_a = action_aggregates(rows_analysis)
    by_team_f, by_team_street_f = action_aggregates(rows)

    hands_with_river_analysis = {
        int(r["hand_number"])
        for r in rows_analysis
        if r["street"] == "River" and r["action_type"] != "DISCARD"
    }

    deltas_analysis = [
        (hand_end[h][0] - hand_end[h - 1][0], h) for h in range(1, first_tail_h if tail else hands)
    ]
    deltas_analysis.sort()

    print("=== PRIMARY: ANALYSIS SEGMENT (strategy / tendencies) ===")
    print(f"File: {path}")
    if tail > 0:
        print(
            f"Excluded preflop-only lockout suffix: hands {first_tail_h}..{hands - 1} "
            f"({tail} hands; e.g. chip-secure auto-fold — not used below)."
        )
        print(f"Analysis covers hands 0..{first_tail_h - 1} ({analysis_hands} hands).")
    else:
        print(f"Hands: {hands} (no preflop-only suffix detected; full log = analysis segment).")

    print(
        f"Bankroll at END of analysis segment — Team0: {br0_analysis:+d}  |  "
        f"Team1: {br1_analysis:+d}"
    )
    if analysis_hands:
        print(f"Per-hand avg over analysis segment (Team0): {br0_analysis / analysis_hands:+.4f} chips/hand")

    print_action_sections(by_team_a, by_team_street_a, "analysis segment only")

    pf_pre, pf_post = fold_rate_stats(rows_analysis)
    print_fold_block(pf_pre, pf_post, "analysis segment only")

    pct_r = (
        100 * len(hands_with_river_analysis) / analysis_hands
        if analysis_hands
        else 0.0
    )
    print(
        f"\nHands with ≥1 River betting action (analysis): "
        f"{len(hands_with_river_analysis)} / {analysis_hands} ({pct_r:.1f}%)"
    )

    print("\n=== TEAM 0 WORST / BEST HANDS (chip delta, analysis segment) ===")
    for d, h in deltas_analysis[:5]:
        print(f"  worst hand {h}: {d:+d} chips")
    for d, h in deltas_analysis[-5:][::-1]:
        print(f"  best  hand {h}: {d:+d} chips")

    print("\n=== POSTFLOP RAISE COUNTS (analysis segment) ===")
    for st in sorted(POSTFLOP):
        r0 = by_team_street_a[(0, st)]["RAISE"]
        r1 = by_team_street_a[(1, st)]["RAISE"]
        print(f"  {st}: Team0={r0}  Team1={r1}")

    print("\n=== OFFICIAL MATCH RESULT (includes lockout tail; scorekeeping only) ===")
    print(f"Hands logged: {hands}")
    print(
        f"Final bankrolls — Team0: {br0_end:+d}  |  Team1: {br1_end:+d}"
    )
    if tail > 0:
        print(
            f"Chip movement on lockout suffix only (Team0): {br0_end - br0_analysis:+d} "
            f"(ignore for strategy; included in official result above)."
        )

    if args.full_log:
        print("\n" + "=" * 60)
        print("OPTIONAL: FULL LOG (includes lockout — use for debugging only)")
        print_action_sections(by_team_f, by_team_street_f, "full log")
        pf2, po2 = fold_rate_stats(rows)
        print_fold_block(pf2, po2, "full log")
        deltas_full = [(hand_end[h][0] - hand_end[h - 1][0], h) for h in range(1, hands)]
        deltas_full.sort()
        print("\n=== TEAM 0 WORST 5 / BEST 5 (full log) ===")
        for d, h in deltas_full[:5]:
            print(f"  worst hand {h}: {d:+d} chips")
        for d, h in deltas_full[-5:][::-1]:
            print(f"  best  hand {h}: {d:+d} chips")


if __name__ == "__main__":
    main()
