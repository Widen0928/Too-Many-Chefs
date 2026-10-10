"""Time a plan, and reorder its steps so the chef finishes sooner.

lama-first ignores time: a plan may put the batter in the oven and then
have the chef stand there for eight ticks, although fetching the cherry
could fill the wait. ``ticks`` estimates how long a plan takes one chef:
walking (shortest paths between the tiles it acts on), one tick per action,
three for chopping, and waiting at a station until food that cooks by
itself is done. ``improve`` then moves steps (and short runs of steps)
earlier or later, keeping a change only if the plan still checks out
exactly (library.valid) and the estimate drops.
"""

from __future__ import annotations

import time

from controllers.pddl.interfaces import PDDLPlanStep

from .library import valid
from .planner.kitchen import Kitchen
from .routing import _options, _station_param, _target, walk

PUT_STEPS = {"put_in_station", "add_to_station", "plate_into_station", "plate_add_to_station"}
TAKE_STEPS = {
    "take_out",
    "combine_from_station",
    "combine_from_station_tended",
    "station_onto_plate",
    "station_onto_plate_tended",
}


def ticks(kitchen: Kitchen, chef, steps: list[PDDLPlanStep]) -> float:
    """Estimated ticks for ``chef`` to carry out ``steps``."""
    start = (chef.x, chef.y)
    reach = kitchen.reach.get(chef.id, set())
    at = None
    t = 0.0
    put: dict[str, float] = {}
    ready: dict[str, float] = {}
    for step in steps:
        if step.name == "cook_by_itself":
            symbol = step.parameters[0]
            station = kitchen.station_by_symbol.get(symbol)
            if station is not None and symbol in put:
                ready[symbol] = put[symbol] + station.entity.cook_time
            continue
        target = _target(step)
        if target is None:
            continue
        options = _options(kitchen, step, reach)
        if options:
            target = min(options, key=lambda o: (walk(kitchen, start, at, o), o))
        t += walk(kitchen, start, at, target)
        at = target
        index = _station_param(step)
        symbol = step.parameters[index] if index is not None else None
        if symbol is not None and step.name in TAKE_STEPS and symbol in ready:
            t = max(t, ready.pop(symbol))
        if symbol is not None and step.name in PUT_STEPS:
            put[symbol] = t
        duration = 1.0
        if step.name == "cook" and symbol is not None:
            station = kitchen.station_by_symbol.get(symbol)
            if station is not None:
                duration = float(max(1, station.entity.cook_time))
        t += duration
    return t


def exposure(kitchen: Kitchen, chef, steps: list[PDDLPlanStep]) -> float:
    """Ticks our food sits in a station while the chef is elsewhere.

    The stations down the middle are shared: food left in one while the
    chef fetches a plate can be gone when it returns (other chefs' plans
    happily use whatever is ready). Counts, for each station, the time from
    putting food in to taking it out when the chef walks off in between.
    """
    start = (chef.x, chef.y)
    reach = kitchen.reach.get(chef.id, set())
    at = None
    t = 0.0
    put: dict[str, tuple[float, int]] = {}
    moves = 0
    total = 0.0
    for step in steps:
        target = _target(step)
        if target is None:
            continue
        options = _options(kitchen, step, reach)
        if options:
            target = min(options, key=lambda o: (walk(kitchen, start, at, o), o))
        if at is not None and target != at:
            moves += 1
        t += walk(kitchen, start, at, target)
        at = target
        index = _station_param(step)
        symbol = step.parameters[index] if index is not None else None
        if symbol is not None and step.name in TAKE_STEPS and symbol in put:
            since, moved = put.pop(symbol)
            if moves > moved:
                total += t - since
        if symbol is not None and step.name in PUT_STEPS:
            put[symbol] = (t, moves)
        t += 1.0
    return total


def improve(
    kitchen: Kitchen,
    chef,
    steps: list[PDDLPlanStep],
    init: frozenset,
    goal: list,
    seconds: float = 0.2,
    longest_run: int = 4,
    guard: float = 0.0,
) -> list[PDDLPlanStep]:
    """``steps`` reordered to finish sooner, still valid from ``init``.
    With ``guard`` > 0, ticks food spends unattended count too (times
    ``guard``), which favours plans that wait beside the stove."""
    deadline = time.monotonic() + seconds

    def cost(plan):
        value = ticks(kitchen, chef, plan)
        return value + guard * exposure(kitchen, chef, plan) if guard else value

    best = list(steps)
    best_t = cost(best)
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(best)
        for length in range(1, longest_run + 1):
            for i in range(0, n - length + 1):
                run = best[i : i + length]
                rest = best[:i] + best[i + length :]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    trial = rest[:j] + run + rest[j:]
                    t = cost(trial)
                    if t >= best_t - 1e-9 or not valid(trial, init, goal):
                        continue
                    best, best_t, improved = trial, t, True
                    break
                if improved or time.monotonic() > deadline:
                    break
            if improved or time.monotonic() > deadline:
                break
    return best
