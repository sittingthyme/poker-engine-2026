"""
Offline trainer for an abstract strategy lookup table (Option A).

This is intentionally lightweight and self-contained (numpy only). It does NOT
try to perfectly solve the full 27-card game tree. Instead, it learns a stable,
hard-to-exploit baseline policy over a coarse abstraction:
  - street (0..3)
  - position (IP/OOP)
  - pot band (small/med/large)
  - continue-cost band (none/small/large)
  - equity band (very weak .. very strong)

It outputs `submission/strategy_table.json`, which the bot can load at startup.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


# ---- Abstraction (must match submission/strategy.py) ----


def state_key(street: int, pos: int, pot_band: int, cost_band: int, eq_band: int) -> str:
    return f"s{street}_p{pos}_pb{pot_band}_cb{cost_band}_eb{eq_band}"


def equity_mid(eq_band: int) -> float:
    # Midpoints consistent with bands in submission/strategy.py
    mids = [0.10, 0.30, 0.50, 0.69, 0.87]
    return float(mids[eq_band])


def pot_mid(pot_band: int) -> int:
    return [6, 25, 90][pot_band]


def cost_mid(cost_band: int) -> int:
    return [0, 3, 12][cost_band]


@dataclass
class OpponentResponseModel:
    """
    Simple fixed response model for training. This is NOT opponent-specific;
    it is just used to create a baseline policy that isn't obviously bad.
    """

    def fold_prob(self, pot: int, raise_frac: float, street: int) -> float:
        # Bigger raises induce more folds; later streets slightly more sticky.
        base = 0.22 + 0.35 * (raise_frac - 0.5)
        street_adj = -0.03 * street
        pot_adj = -0.02 if pot > 50 else 0.0
        return float(min(0.75, max(0.05, base + street_adj + pot_adj)))


def ev_call(equity: float, pot: int, continue_cost: int) -> float:
    if continue_cost <= 0:
        return 0.0
    final_pot = pot + continue_cost
    return equity * final_pot - (1.0 - equity) * continue_cost


def ev_raise(equity: float, pot: int, delta: int, p_fold: float) -> float:
    # Same simplified branch model as submission/strategy.py.
    ev_fold_branch = pot + delta
    ev_call_branch = equity * (pot + 2 * delta) - (1.0 - equity) * delta
    return p_fold * ev_fold_branch + (1.0 - p_fold) * ev_call_branch


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    pos = [max(r, 0.0) for r in regrets]
    s = sum(pos)
    if s <= 1e-12:
        return [1.0 / len(pos)] * len(pos)
    return [p / s for p in pos]


def main() -> None:
    rng = random.Random(7)
    opp = OpponentResponseModel()

    # Actions: fold, call, raise_small, raise_big
    A = 4

    # State space sizes
    streets = 4
    pos = 2
    pot_bands = 3
    cost_bands = 3
    eq_bands = 5

    # Regrets and average strategy accumulator
    regrets = [[[[[[0.0 for _ in range(A)] for _ in range(eq_bands)] for _ in range(cost_bands)] for _ in range(pot_bands)] for _ in range(pos)] for _ in range(streets)]
    strat_sum = [[[[[[0.0 for _ in range(A)] for _ in range(eq_bands)] for _ in range(cost_bands)] for _ in range(pot_bands)] for _ in range(pos)] for _ in range(streets)]

    # Training iterations
    iters = 4000
    for t in range(1, iters + 1):
        # Sample a random abstract state each iteration (MCCFR-ish sampling)
        s = rng.randrange(streets)
        p = rng.randrange(pos)
        pb = rng.randrange(pot_bands)
        cb = rng.randrange(cost_bands)
        eb = rng.randrange(eq_bands)

        pot = pot_mid(pb)
        cost = cost_mid(cb)
        equity = equity_mid(eb)

        sigma = regret_matching(regrets[s][p][pb][cb][eb])
        for i in range(A):
            strat_sum[s][p][pb][cb][eb][i] += sigma[i]

        # Compute action values under a fixed response model.
        # Fold baseline = 0 (only meaningful when cost>0; still harmless elsewhere).
        v_fold = 0.0

        # Call/check: if no cost, treat as check with neutral EV in this local model.
        v_call = ev_call(equity, pot, cost) if cost > 0 else 0.0

        # Raises: choose delta as pot fractions. If cost==0, this is a bet.
        # We approximate delta relative to current commitment; here we only need delta.
        delta_small = max(2, int(pot * 0.50))
        delta_big = max(2, int(pot * 0.80))
        p_fold_small = opp.fold_prob(pot, 0.50, s)
        p_fold_big = opp.fold_prob(pot, 0.80, s)
        v_rs = ev_raise(equity, pot, delta_small, p_fold_small)
        v_rb = ev_raise(equity, pot, delta_big, p_fold_big)

        vals = [v_fold, v_call, v_rs, v_rb]

        # Expected value under current strategy
        v_sigma = sum(sigma[i] * vals[i] for i in range(A))

        # Regret update
        for i in range(A):
            regrets[s][p][pb][cb][eb][i] += vals[i] - v_sigma

        # Small diminishing noise encourages exploration early, vanishes late
        if t < 800:
            for i in range(A):
                # Approximate gaussian with Box-Muller
                u1 = max(1e-12, rng.random())
                u2 = rng.random()
                z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                regrets[s][p][pb][cb][eb][i] += z * 0.01

    # Build final average strategy table (normalized)
    out: dict[str, dict[str, float]] = {}
    for s in range(streets):
        for p in range(pos):
            for pb in range(pot_bands):
                for cb in range(cost_bands):
                    for eb in range(eq_bands):
                        avg = strat_sum[s][p][pb][cb][eb]
                        total = sum(avg)
                        if total <= 0:
                            avg = [1.0] * A
                            total = float(A)
                        avg = [x / total for x in avg]
                        out[state_key(s, p, pb, cb, eb)] = {
                            "fold": float(avg[0]),
                            "call": float(avg[1]),
                            "raise_small": float(avg[2]),
                            "raise_big": float(avg[3]),
                        }

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "submission" / "strategy_table.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path} with {len(out)} states.")


if __name__ == "__main__":
    main()

