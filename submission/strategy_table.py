"""
Lightweight strategy lookup table for Option A (table-first policy).

The table maps an abstracted game state to action probabilities.
It is designed to be:
  - fast to query during matches
  - easy to export from offline training
  - safe under tournament restrictions (no network, small files)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ActionProbs = dict[str, float]


@dataclass(frozen=True)
class StrategyEntry:
    """Probs + visit count for one abstract state."""

    probs: ActionProbs
    visits: int


@dataclass(frozen=True)
class StrategyTable:
    """
    Holds a mapping from state keys -> StrategyEntry (probs + visits).

    State key format (string):
        "s{street}_p{pos}_pb{potBand}_cb{costBand}_eb{equityBand}"

    Actions (string keys):
        - "fold"
        - "call"   (also used as "check" when no continue cost)
        - "raise_small"
        - "raise_big"
    """

    table: Mapping[str, StrategyEntry]

    @staticmethod
    def default_path() -> Path:
        # `submission/` is a Python package directory; keep the file alongside code.
        return Path(__file__).with_name("strategy_table.json")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "StrategyTable | None":
        p = Path(path) if path is not None else cls.default_path()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception:
            # If the file exists but is malformed, fail closed (no table).
            return None

        if not isinstance(raw, dict):
            return None

        table: dict[str, StrategyEntry] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            # New format: {"probs": {...}, "visits": N}
            if "probs" in v and "visits" in v:
                probs_raw = v["probs"]
                visits = int(v["visits"])
            else:
                # Legacy format: flat probs dict
                probs_raw = v
                visits = 0
            probs: ActionProbs = {}
            for ak, av in probs_raw.items():
                if isinstance(ak, str) and isinstance(av, (int, float)):
                    probs[ak] = float(av)
            if probs:
                table[k] = StrategyEntry(probs=probs, visits=visits)

        return cls(table=table) if table else None

    def get(self, state_key: str) -> ActionProbs | None:
        entry = self.table.get(state_key)
        return dict(entry.probs) if entry else None

    def get_with_confidence(
        self, state_key: str, visit_threshold: int = 200
    ) -> tuple[ActionProbs | None, float]:
        """Return (probs, confidence) where confidence = min(1, visits/visit_threshold)."""
        entry = self.table.get(state_key)
        if not entry:
            return None, 0.0
        conf = min(1.0, entry.visits / float(visit_threshold)) if visit_threshold > 0 else 1.0
        return dict(entry.probs), conf

