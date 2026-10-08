"""Estimate each level's best possible score (orders plus every tip).

The grader pays each order the best-tipping route: every distinct food on
the route that is fetched from a storage or prepared earns one tip.

Usage: uv run python tools/max_tips.py [glob ...]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.configuration.load import load, load_configuration  # noqa: E402
from simulator.entities import Food, OvercookedState, Storage  # noqa: E402


def best_route(state, storages, loose, dish):
    def routes(food, depth):
        out = []
        if food in loose:
            out.append(frozenset())
        if food in storages:
            out.append(frozenset() if food == dish else frozenset({food}))
        if depth > 10:
            return out or [None]
        for r in state.cook_recipes:
            if r.output == food:
                for sub in routes(r.ingredient, depth + 1):
                    if sub is not None:
                        out.append(sub | {food})
        for r in state.combine_recipes:
            if r.output == food:
                a, *rest = sorted(r.ingredients)
                b = rest[0] if rest else a
                for sa in routes(a, depth + 1):
                    for sb in routes(b, depth + 1):
                        if sa is not None and sb is not None:
                            out.append(sa | sb | {food})
        out = [o for o in out if o is not None]
        return sorted(set(out), key=len, reverse=True)[:20] or [None]

    best = routes(dish, 0)[0]
    return best


def main() -> None:
    patterns = sys.argv[1:] or ["levels/[012]_*/*.yaml"]
    for path in sorted({p for pat in patterns for p in glob.glob(pat)}):
        env = load(load_configuration(path))
        state = env.get_first_entity_of_type(OvercookedState)
        storages = {s.food_name for s in env.get_entities_of_type(Storage)}
        loose = {f.name for f in env.get_entities_of_type(Food)}
        total = 0
        detail = []
        for entry in [*state.order_queue.visible, *state.order_queue.pending]:
            route = best_route(state, storages, loose, entry.name)
            tips = len(route) if route is not None else 0
            total += entry.reward + tips
            detail.append(tips)
        print(f"{Path(path).parent.name[:2]}/{Path(path).stem:55s} max={total:4d} tips={detail}")


if __name__ == "__main__":
    main()
