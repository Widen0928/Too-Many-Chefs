"""Tunable choices of the controller, in one place.

The defaults are what the submission uses. For local experiments a JSON
object in the ``TMC_EXPERIMENT`` environment variable overrides any of them
(worker processes inherit the environment), so variants can be compared on
the same levels without editing code; see tools/bench.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Config:
    # Most orders one plan covers (None: all that can be made). Three did best
    # in local tests: smaller problems give lama-first more focused plans
    # (and the rush-hour ticket); the controller plans on as orders are served.
    horizon: int | None = 3
    # Replan after every delivery, not only when something unexpected happens.
    replan_on_delivery: bool = False
    # Queue for orders that may be served in any order: "deadline" or "reward"
    # (higher reward per remaining time first).
    order_sort: str = "deadline"
    # Which models to plan with: "both", "fast" or "safe".
    models: str = "both"
    # Seconds the tip-keeping model gets.
    tip_safe_seconds: float = 6.0
    # Dispatch variants to roll out: "full" or "small".
    candidates: str = "full"


def load() -> Config:
    config = Config()
    raw = os.environ.get("TMC_EXPERIMENT")
    if not raw:
        return config
    try:
        values = json.loads(raw)
        known = {f.name for f in fields(Config)}
        return replace(config, **{k: v for k, v in values.items() if k in known})
    except Exception:  # noqa: BLE001 - a bad override must not break a run
        return config


CONFIG = load()
