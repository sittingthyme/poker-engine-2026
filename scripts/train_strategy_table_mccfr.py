#!/usr/bin/env python3
"""
Offline strategy-table trainer (tabular regret / self-play).

Delegates to training/train_strategy_table_env.py — state keys come from
submission.abstract_state (must match runtime StrategyTable lookup).

Usage:
  python scripts/train_strategy_table_mccfr.py

Writes submission/strategy_table.json (see TrainConfig in the training module).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    from training.train_strategy_table_env import main as train_main

    train_main()


if __name__ == "__main__":
    main()
