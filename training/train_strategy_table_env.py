"""
Offline self-play trainer for the strategy lookup table (Option A).

Goal
----
Generate `submission/strategy_table.json` by learning a robust baseline policy
through *self-play rollouts* in the real `PokerEnv` environment.

This is intentionally approximate:
- We keep a coarse state abstraction (same key format as runtime).
- We use a small discrete action abstraction (fold/call/raise_small/raise_big).
- We use Monte Carlo rollouts (few sims) to estimate counterfactual values.

Design constraints
------------------
- Must not rely on non-allowed tournament libraries at runtime (this is offline).
- Output is a compact JSON table that the submission bot can load at init.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Ensure repo root is on sys.path when running as `python training/...py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gym_env import PokerEnv  # noqa: E402

from submission.equity import best_discard, compute_equity

_DEBUG_LOG_PATH = "/Users/nicholasng/Documents/GitHub/poker-engine-2026/.cursor/debug-a2906f.log"
_DEBUG_SESSION_ID = "a2906f"


def _dlog(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


AbstractAction = Literal["fold", "call", "raise_small", "raise_big"]


class OpponentType:
    """Fixed opponent policies for diversity in training."""
    SELF_PLAY = "self_play"
    PROB_LIKE = "prob_like"
    TIGHT = "tight"
    LOOSE = "loose"
    RANDOM = "random"


def get_opponent_action(
    *,
    opp_type: str,
    obs: dict,
    equity: float,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    """
    Return an engine action tuple for a fixed opponent policy.
    Used when opp_type != SELF_PLAY.
    """
    valid = obs["valid_actions"]
    pot = obs.get("pot_size", obs["my_bet"] + obs["opp_bet"])
    continue_cost = obs.get("opp_bet", 0) - obs.get("my_bet", 0)
    pot_odds = continue_cost / (continue_cost + pot) if continue_cost > 0 else 0.0

    if opp_type == OpponentType.PROB_LIKE:
        # ProbabilityAgent-like: raise if equity > 0.75, call if equity >= pot_odds
        if equity > 0.75 and valid[PokerEnv.ActionType.RAISE.value]:
            amount = max(obs["min_raise"], min(int(pot * 0.75), obs["max_raise"]))
            return (PokerEnv.ActionType.RAISE.value, amount, 0, 0)
        if equity >= pot_odds:
            if valid[PokerEnv.ActionType.CALL.value]:
                return (PokerEnv.ActionType.CALL.value, 0, 0, 0)
            if valid[PokerEnv.ActionType.CHECK.value]:
                return (PokerEnv.ActionType.CHECK.value, 0, 0, 0)
        if valid[PokerEnv.ActionType.FOLD.value]:
            return (PokerEnv.ActionType.FOLD.value, 0, 0, 0)
        if valid[PokerEnv.ActionType.CHECK.value]:
            return (PokerEnv.ActionType.CHECK.value, 0, 0, 0)
        return (PokerEnv.ActionType.CALL.value, 0, 0, 0)

    if opp_type == OpponentType.TIGHT:
        # Tight/passive: raise only if equity > 0.85, call more, bluff less
        if equity > 0.85 and valid[PokerEnv.ActionType.RAISE.value]:
            amount = max(obs["min_raise"], min(int(pot * 0.65), obs["max_raise"]))
            return (PokerEnv.ActionType.RAISE.value, amount, 0, 0)
        if equity >= pot_odds - 0.02:
            if valid[PokerEnv.ActionType.CALL.value]:
                return (PokerEnv.ActionType.CALL.value, 0, 0, 0)
            if valid[PokerEnv.ActionType.CHECK.value]:
                return (PokerEnv.ActionType.CHECK.value, 0, 0, 0)
        if valid[PokerEnv.ActionType.FOLD.value]:
            return (PokerEnv.ActionType.FOLD.value, 0, 0, 0)
        if valid[PokerEnv.ActionType.CHECK.value]:
            return (PokerEnv.ActionType.CHECK.value, 0, 0, 0)
        return (PokerEnv.ActionType.CALL.value, 0, 0, 0)

    if opp_type == OpponentType.LOOSE:
        # Loose/aggressive: raise if equity > 0.55, bluff ~15% when facing a bet
        if equity > 0.55 and valid[PokerEnv.ActionType.RAISE.value]:
            amount = max(obs["min_raise"], min(int(pot * 0.80), obs["max_raise"]))
            return (PokerEnv.ActionType.RAISE.value, amount, 0, 0)
        if continue_cost > 0 and 0.25 <= equity <= 0.55 and rng.random() < 0.15:
            if valid[PokerEnv.ActionType.RAISE.value]:
                amount = max(obs["min_raise"], min(int(pot * 0.50), obs["max_raise"]))
                return (PokerEnv.ActionType.RAISE.value, amount, 0, 0)
        if equity >= pot_odds - 0.05:
            if valid[PokerEnv.ActionType.CALL.value]:
                return (PokerEnv.ActionType.CALL.value, 0, 0, 0)
            if valid[PokerEnv.ActionType.CHECK.value]:
                return (PokerEnv.ActionType.CHECK.value, 0, 0, 0)
        if valid[PokerEnv.ActionType.FOLD.value]:
            return (PokerEnv.ActionType.FOLD.value, 0, 0, 0)
        if valid[PokerEnv.ActionType.CHECK.value]:
            return (PokerEnv.ActionType.CHECK.value, 0, 0, 0)
        return (PokerEnv.ActionType.CALL.value, 0, 0, 0)

    if opp_type == OpponentType.RANDOM:
        # Uniform over valid abstract actions
        choices: list[AbstractAction] = []
        if valid[PokerEnv.ActionType.FOLD.value]:
            choices.append("fold")
        if valid[PokerEnv.ActionType.CALL.value] or valid[PokerEnv.ActionType.CHECK.value]:
            choices.append("call")
        if valid[PokerEnv.ActionType.RAISE.value]:
            choices.extend(["raise_small", "raise_big"])
        if not choices:
            return (PokerEnv.ActionType.CALL.value, 0, 0, 0)
        a = rng.choice(choices)
        return action_to_env(a=a, obs=obs, equity=equity)

    # Fallback (e.g. SELF_PLAY should not call this)
    return action_to_env(a="call", obs=obs, equity=equity)


def regret_matching(regrets: dict[AbstractAction, float]) -> dict[AbstractAction, float]:
    pos = {a: max(0.0, r) for a, r in regrets.items()}
    s = sum(pos.values())
    if s <= 1e-12:
        n = float(len(pos))
        return {a: 1.0 / n for a in pos}
    return {a: v / s for a, v in pos.items()}


def sample_action(rng: random.Random, probs: dict[AbstractAction, float]) -> AbstractAction:
    total = sum(max(0.0, p) for p in probs.values())
    if total <= 0:
        return "call"
    r = rng.random() * total
    for a, p in probs.items():
        r -= max(0.0, p)
        if r <= 0:
            return a
    return "call"


def abstract_state_key(*, street: int, in_position: bool, pot: int, continue_cost: int, equity: float) -> str:
    """
    Must match `_abstract_state_key` in `submission/strategy.py`.
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


def box_muller_gauss(rng: random.Random, sigma: float) -> float:
    u1 = max(1e-12, rng.random())
    u2 = rng.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z * sigma


def in_position(*, street: int, blind_position: int) -> bool:
    # Same logic used in submission/strategy.py.
    return (street == 0 and blind_position == 1) or (street >= 1 and blind_position == 0)


def action_to_env(
    *,
    a: AbstractAction,
    obs: dict,
    equity: float,
) -> tuple[int, int, int, int]:
    """
    Convert an abstract action into a concrete engine action tuple.
    We never emit DISCARD from here.
    """
    valid = obs["valid_actions"]
    pot = obs.get("pot_size", obs["my_bet"] + obs["opp_bet"])
    min_raise = obs["min_raise"]
    max_raise = obs["max_raise"]

    # FOLD
    if a == "fold":
        if valid[PokerEnv.ActionType.FOLD.value]:
            return (PokerEnv.ActionType.FOLD.value, 0, 0, 0)
        # If fold isn't legal (should be), fall back.
        a = "call"

    # CALL / CHECK
    if a == "call":
        if valid[PokerEnv.ActionType.CALL.value]:
            return (PokerEnv.ActionType.CALL.value, 0, 0, 0)
        if valid[PokerEnv.ActionType.CHECK.value]:
            return (PokerEnv.ActionType.CHECK.value, 0, 0, 0)
        return (PokerEnv.ActionType.FOLD.value, 0, 0, 0)

    # RAISE
    if not valid[PokerEnv.ActionType.RAISE.value]:
        # Not legal, fallback.
        return action_to_env(a="call", obs=obs, equity=equity)

    # Engine invariant: when raising, opponent must have bet >= our bet.
    # If we're ahead somehow, treat the raise as a call/check.
    if obs.get("my_bet", 0) > obs.get("opp_bet", 0):
        _dlog(
            run_id="pre-fix",
            hypothesis_id="H1_raise_when_ahead",
            location="training/train_strategy_table_env.py:action_to_env",
            message="Attempted raise when my_bet > opp_bet; fallback to call/check",
            data={
                "a": a,
                "street": obs.get("street"),
                "my_bet": obs.get("my_bet"),
                "opp_bet": obs.get("opp_bet"),
                "pot": pot,
                "min_raise": min_raise,
                "max_raise": max_raise,
                "equity": equity,
                "valid_actions": list(valid),
            },
        )
        return action_to_env(a="call", obs=obs, equity=equity)

    frac = 0.50 if a == "raise_small" else (0.80 if equity >= 0.60 else 0.65)
    raw = int(pot * frac)
    amount = max(min_raise, min(raw, max_raise))
    if amount <= 0:
        return action_to_env(a="call", obs=obs, equity=equity)
    return (PokerEnv.ActionType.RAISE.value, amount, 0, 0)


@dataclass
class TrainConfig:
    hands: int = 3000
    # Rollouts used to estimate each counterfactual action value at the sampled state.
    rollouts_per_action: int = 12
    equity_sims: int = 80
    discard_sims_per_pair: int = 80
    # Exploration noise early in training
    regret_noise_sigma: float = 0.02
    regret_noise_hands: int = 250

    # Opponent diversity: weights for sampling opponent type per hand
    # (self_play, prob_like, tight, loose, random)
    opponent_weights: tuple[float, ...] = (0.40, 0.25, 0.15, 0.15, 0.05)


class TableTrainer:
    ACTIONS: tuple[AbstractAction, ...] = ("fold", "call", "raise_small", "raise_big")

    def __init__(self, cfg: TrainConfig, seed: int = 7) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        # regrets/state: state_key -> action -> regret
        self.regrets: dict[str, dict[AbstractAction, float]] = {}
        self.strat_sum: dict[str, dict[AbstractAction, float]] = {}
        self.visit_counts: dict[str, int] = {}

    def _ensure_state(self, k: str) -> None:
        if k not in self.regrets:
            self.regrets[k] = {a: 0.0 for a in self.ACTIONS}
            self.strat_sum[k] = {a: 0.0 for a in self.ACTIONS}
            self.visit_counts[k] = 0

    def current_policy(self, k: str) -> dict[AbstractAction, float]:
        self._ensure_state(k)
        return regret_matching(self.regrets[k])

    def _accumulate_avg(self, k: str, sigma: dict[AbstractAction, float]) -> None:
        for a, p in sigma.items():
            self.strat_sum[k][a] += float(p)

    def export_table(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for k, sums in self.strat_sum.items():
            total = sum(sums.values())
            visits = self.visit_counts.get(k, 0)
            if total <= 0:
                probs = {"fold": 0.25, "call": 0.25, "raise_small": 0.25, "raise_big": 0.25}
            else:
                probs = {a: float(v / total) for a, v in sums.items()}
            out[k] = {"probs": probs, "visits": visits}
        return out

    # -----------------------
    # Rollout / simulation
    # -----------------------

    def _random_deck(self) -> list[int]:
        deck = list(range(27))
        self.rng.shuffle(deck)
        return deck

    def _play_hand_collect_history(
        self,
        *,
        deck: list[int],
        small_blind_player: int,
    ) -> tuple[list[tuple[int, int, int, int]], list[dict], list[int], int]:
        """
        Play a hand using table policy for the table player and optionally a fixed
        opponent for diversity.

        Returns:
          - action_history: list of action tuples applied sequentially
          - obs_history: observations for the acting player *before* each action
          - acting_player_history: acting player index before each action
          - table_player: player index whose policy we train (-1 if self-play, both use table)
        """
        opp_types = [
            OpponentType.SELF_PLAY,
            OpponentType.PROB_LIKE,
            OpponentType.TIGHT,
            OpponentType.LOOSE,
            OpponentType.RANDOM,
        ]
        opp_type = self.rng.choices(opp_types, weights=self.cfg.opponent_weights, k=1)[0]
        table_player: int = self.rng.randrange(2) if opp_type != OpponentType.SELF_PLAY else -1

        env = PokerEnv(num_hands=1)
        (obs0, obs1), _ = env.reset(options={"cards": deck, "small_blind_player": small_blind_player})
        obs = [obs0, obs1]

        action_history: list[tuple[int, int, int, int]] = []
        obs_history: list[dict] = []
        acting_hist: list[int] = []

        terminated = False
        truncated = False
        info = {}
        reward = (0, 0)

        # Track discards so equity calls can condition on them.
        my_discards = [[], []]

        while not terminated and not truncated:
            acting = obs[0]["acting_agent"]
            o = obs[acting]
            valid = o["valid_actions"]

            # Discard round (mandatory at street 1 if not completed).
            if valid[PokerEnv.ActionType.DISCARD.value]:
                my_cards = [c for c in o["my_cards"] if c != -1]
                community = [c for c in o["community_cards"] if c != -1]
                opp_disc = [c for c in o.get("opp_discarded_cards", [-1, -1, -1]) if c != -1]

                keep_i, keep_j, _ = best_discard(
                    my_cards,
                    community,
                    opp_discarded=opp_disc if opp_disc else None,
                    sims_per_pair=self.cfg.discard_sims_per_pair,
                )
                my_discards[acting] = [my_cards[k] for k in range(5) if k not in (keep_i, keep_j)]
                a_env = (PokerEnv.ActionType.DISCARD.value, 0, keep_i, keep_j)
            else:
                # Betting decision
                street = o["street"]
                pot = o.get("pot_size", o["my_bet"] + o["opp_bet"])
                continue_cost = o["opp_bet"] - o["my_bet"]
                blind_pos = o.get("blind_position", 0)
                ip = in_position(street=street, blind_position=blind_pos)
                equity = compute_equity(
                    [c for c in o["my_cards"] if c != -1][:2],
                    [c for c in o["community_cards"] if c != -1],
                    opp_discarded=[c for c in o.get("opp_discarded_cards", [-1, -1, -1]) if c != -1] or None,
                    my_discarded=my_discards[acting] or None,
                    num_simulations=self.cfg.equity_sims,
                )
                k = abstract_state_key(
                    street=street,
                    in_position=ip,
                    pot=pot,
                    continue_cost=continue_cost,
                    equity=equity,
                )

                use_table = (
                    opp_type == OpponentType.SELF_PLAY
                    or acting == table_player
                )
                if use_table:
                    sigma = self.current_policy(k)
                    self._accumulate_avg(k, sigma)
                    a_abs = sample_action(self.rng, sigma)
                    a_env = action_to_env(a=a_abs, obs=o, equity=equity)
                else:
                    a_env = get_opponent_action(
                        opp_type=opp_type,
                        obs=o,
                        equity=equity,
                        rng=self.rng,
                    )

            # Record decision point for possible counterfactual training.
            obs_history.append(dict(o))
            acting_hist.append(acting)
            action_history.append(a_env)

            try:
                (obs0, obs1), reward, terminated, truncated, info = env.step(a_env)
            except AssertionError as e:
                _dlog(
                    run_id="pre-fix",
                    hypothesis_id="H2_engine_assert",
                    location="training/train_strategy_table_env.py:_finish_hand_from_state",
                    message="Engine assertion during env.step",
                    data={
                        "err": str(e),
                        "a_env": list(a_env),
                        "street": o.get("street"),
                        "my_bet": o.get("my_bet"),
                        "opp_bet": o.get("opp_bet"),
                        "min_raise": o.get("min_raise"),
                        "max_raise": o.get("max_raise"),
                        "pot": o.get("pot_size", o.get("my_bet", 0) + o.get("opp_bet", 0)),
                        "valid_actions": list(o.get("valid_actions", [])),
                    },
                )
                raise
            obs = [obs0, obs1]

        # reward is per-hand reward; env uses min(bets) as chip delta
        # reward[0] is from perspective of player 0.
        return action_history, obs_history, acting_hist, table_player

    def _replay_to_step(
        self,
        *,
        deck: list[int],
        small_blind_player: int,
        action_history: list[tuple[int, int, int, int]],
        stop_idx: int,
    ) -> tuple[PokerEnv, list[dict], list[list[int]]]:
        """
        Reset env with the same deck and replay actions up to stop_idx (exclusive).
        Returns env plus current obs list and tracked discards.
        """
        env = PokerEnv(num_hands=1)
        (obs0, obs1), _ = env.reset(options={"cards": deck, "small_blind_player": small_blind_player})
        obs = [obs0, obs1]
        my_discards = [[], []]

        terminated = False
        truncated = False
        reward = (0, 0)
        info = {}

        for i in range(stop_idx):
            acting = obs[0]["acting_agent"]
            o = obs[acting]
            a_env = action_history[i]
            if a_env[0] == PokerEnv.ActionType.DISCARD.value:
                # reconstruct discards for equity later
                keep_i, keep_j = a_env[2], a_env[3]
                my_cards = [c for c in o["my_cards"] if c != -1]
                my_discards[acting] = [my_cards[k] for k in range(5) if k not in (keep_i, keep_j)]

            (obs0, obs1), reward, terminated, truncated, info = env.step(a_env)
            obs = [obs0, obs1]
            if terminated or truncated:
                break

        return env, obs, my_discards

    def _finish_hand_from_state(
        self,
        *,
        env: PokerEnv,
        obs: list[dict],
        my_discards: list[list[int]],
    ) -> tuple[int, int]:
        """
        Finish the current hand using current policy for both players.
        Returns final reward tuple.
        """
        terminated = False
        truncated = False
        info = {}
        reward = (0, 0)

        while not terminated and not truncated:
            acting = obs[0]["acting_agent"]
            o = obs[acting]
            valid = o["valid_actions"]

            if valid[PokerEnv.ActionType.DISCARD.value]:
                my_cards = [c for c in o["my_cards"] if c != -1]
                community = [c for c in o["community_cards"] if c != -1]
                opp_disc = [c for c in o.get("opp_discarded_cards", [-1, -1, -1]) if c != -1]
                keep_i, keep_j, _ = best_discard(
                    my_cards,
                    community,
                    opp_discarded=opp_disc if opp_disc else None,
                    sims_per_pair=self.cfg.discard_sims_per_pair,
                )
                my_discards[acting] = [my_cards[k] for k in range(5) if k not in (keep_i, keep_j)]
                a_env = (PokerEnv.ActionType.DISCARD.value, 0, keep_i, keep_j)
            else:
                street = o["street"]
                pot = o.get("pot_size", o["my_bet"] + o["opp_bet"])
                continue_cost = o["opp_bet"] - o["my_bet"]
                blind_pos = o.get("blind_position", 0)
                ip = in_position(street=street, blind_position=blind_pos)
                equity = compute_equity(
                    [c for c in o["my_cards"] if c != -1][:2],
                    [c for c in o["community_cards"] if c != -1],
                    opp_discarded=[c for c in o.get("opp_discarded_cards", [-1, -1, -1]) if c != -1] or None,
                    my_discarded=my_discards[acting] or None,
                    num_simulations=self.cfg.equity_sims,
                )
                k = abstract_state_key(
                    street=street,
                    in_position=ip,
                    pot=pot,
                    continue_cost=continue_cost,
                    equity=equity,
                )
                sigma = self.current_policy(k)
                a_abs = sample_action(self.rng, sigma)
                a_env = action_to_env(a=a_abs, obs=o, equity=equity)

            (obs0, obs1), reward, terminated, truncated, info = env.step(a_env)
            obs = [obs0, obs1]

        return int(reward[0]), int(reward[1])

    def train(self) -> None:
        """
        Outcome-sampling style training:
        - For each hand, play once with current policy and record history.
        - Sample a single betting decision point from that hand.
        - Estimate counterfactual values for all abstract actions by rollouts.
        - Regret-update that state for the acting player.
        """
        for h in range(self.cfg.hands):
            # Lightweight progress indicator so long runs are visible.
            if h % 50 == 0:
                print(f"[strategy-train] hand {h}/{self.cfg.hands}", flush=True)
            deck = self._random_deck()
            sb = self.rng.randrange(2)

            action_hist, obs_hist, acting_hist, table_player = self._play_hand_collect_history(
                deck=deck, small_blind_player=sb
            )

            # Choose a random betting decision point (exclude discard-only entries).
            candidate_idxs = [
                i for i, o in enumerate(obs_hist)
                if not o["valid_actions"][PokerEnv.ActionType.DISCARD.value]
            ]
            if not candidate_idxs:
                continue
            idx = self.rng.choice(candidate_idxs)

            # When using opponent diversity, only update regrets for table player's decisions.
            if table_player >= 0 and acting_hist[idx] != table_player:
                continue

            # Replay to that point, then evaluate each action via rollouts.
            env, obs, discards = self._replay_to_step(
                deck=deck,
                small_blind_player=sb,
                action_history=action_hist,
                stop_idx=idx,
            )
            acting = obs[0]["acting_agent"]
            o = obs[acting]

            street = o["street"]
            pot = o.get("pot_size", o["my_bet"] + o["opp_bet"])
            continue_cost = o["opp_bet"] - o["my_bet"]
            blind_pos = o.get("blind_position", 0)
            ip = in_position(street=street, blind_position=blind_pos)
            equity = compute_equity(
                [c for c in o["my_cards"] if c != -1][:2],
                [c for c in o["community_cards"] if c != -1],
                opp_discarded=[c for c in o.get("opp_discarded_cards", [-1, -1, -1]) if c != -1] or None,
                my_discarded=discards[acting] or None,
                num_simulations=self.cfg.equity_sims,
            )
            k = abstract_state_key(
                street=street,
                in_position=ip,
                pot=pot,
                continue_cost=continue_cost,
                equity=equity,
            )

            self._ensure_state(k)
            self.visit_counts[k] = self.visit_counts.get(k, 0) + 1
            sigma = self.current_policy(k)
            # (We already accumulated average strategy during play; doing it here as well is ok.)
            self._accumulate_avg(k, sigma)

            vals: dict[AbstractAction, float] = {}
            for a in self.ACTIONS:
                total = 0.0
                for _ in range(self.cfg.rollouts_per_action):
                    env2, obs2, disc2 = self._replay_to_step(
                        deck=deck,
                        small_blind_player=sb,
                        action_history=action_hist,
                        stop_idx=idx,
                    )
                    acting2 = obs2[0]["acting_agent"]
                    o2 = obs2[acting2]
                    a_env = action_to_env(a=a, obs=o2, equity=equity)
                    (obs0, obs1), reward, terminated, truncated, info = env2.step(a_env)
                    r0, r1 = self._finish_hand_from_state(env=env2, obs=[obs0, obs1], my_discards=disc2)
                    total += float(r0 if acting == 0 else r1)
                vals[a] = total / float(self.cfg.rollouts_per_action)

            v_sigma = sum(sigma[a] * vals[a] for a in self.ACTIONS)
            for a in self.ACTIONS:
                self.regrets[k][a] += vals[a] - v_sigma

            if h < self.cfg.regret_noise_hands:
                for a in self.ACTIONS:
                    self.regrets[k][a] += box_muller_gauss(self.rng, self.cfg.regret_noise_sigma)


def main() -> None:
    cfg = TrainConfig()
    trainer = TableTrainer(cfg)
    trainer.train()

    out = trainer.export_table()

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "submission" / "strategy_table.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path} with {len(out)} learned states.")


if __name__ == "__main__":
    main()

