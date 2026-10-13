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
    # Horizons to choose between per level by playing the level out at the
    # start (strategy.py); empty to always use ``horizon``.
    select_horizons: tuple = (3, "3r", 2, None, "3w", 1, "2r", "3safe", "2w", 4, "1r", "r", "4r")
    # Seconds strategy selection may spend playing the level out. The
    # server stops a level after 60 s (Level 2) or 90 s (Level 3), and the
    # run itself needs time too, so this stays well inside.
    selection_seconds: float = 30.0
    # Once a full trial reaches a conservative ceiling on remaining points,
    # further strategies cannot improve the grade.
    stop_at_score_bound: bool = True
    # When the best plan lets an order expire, also try plans that leave the
    # most urgent order out or serve the next one first.
    # Experimental: a partial-plan rollout cannot account for the replan
    # triggered when an order expires. Keep the verified baseline by default.
    deadline_alternatives: bool = False
    deadline_kinds: tuple = ("swap",)
    # Compare swap-enabled policies only after the existing whole-level
    # candidates, within the same selection budget.
    select_deadline_alternatives: bool = False
    # Dispatch variants to roll out: "full" or "small".
    candidates: str = "full"
    # Compare grouping identical dishes as a separate whole-level policy
    # in timed kitchens with cooking chains. Keep deadline sorting available.
    select_order_batches: bool = True


def load() -> Config:
    config = Config()
    raw = os.environ.get("TMC_EXPERIMENT")
    if not raw:
        return config
    try:
        values = json.loads(raw)
        known = {f.name for f in fields(Config)}
        values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
        return replace(config, **{k: v for k, v in values.items() if k in known})
    except Exception:  # noqa: BLE001 - a bad override must not break a run
        return config


CONFIG = load()
