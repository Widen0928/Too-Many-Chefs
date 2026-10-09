"""Walking: which stations a plan uses, and how the chef gets around.

lama-first ignores costs, so a plan may bake the chicken in an oven at the
far end of the kitchen while the fryer it also needs is next to the chicken
storage. ``relocate_stations`` swaps each station a plan uses for an
equivalent empty one (same equipment, standing the same way) whenever that
shortens the chef's walk, and ``plan_ticks`` estimates how long a plan takes.

``route`` finds the chef's next step. Tiles nearer the other team's starting
places cost more, since their chefs walk there; and when another chef is in
the way the controller can compare the way round with the direct way and
wait for a moment rather than take a long detour.
"""

from __future__ import annotations

import heapq

from controllers.pddl.interfaces import PDDLPlanStep

from .planner.dispatch import BIN_STEPS, DELIVERY_STEPS, STORAGE_STEPS, _food_sym
from .planner.kitchen import Kitchen, loc_symbol, parse_loc

# Steps that pick a station up or set it down; such a station stays put.
MOVE_STATION_STEPS = {
    "lift_station_from_counter",
    "lift_station_from_support",
    "set_station_on_support",
    "set_station_on_counter",
}
# Extra cost of using a station another chef stands beside (they are
# probably about to use it).
CROWDED = 4.0
# Cost of a step onto a tile nearer the other team.
FOREIGN_COST = 3


def neighbours(tile):
    x, y = tile
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


# --- Plans ---------------------------------------------------------------------


def _station_param(step: PDDLPlanStep) -> int | None:
    """Index of the station symbol in a step's parameters, if any."""
    if step.name == "cook_by_itself":
        return 0
    if len(step.parameters) > 2 and step.parameters[1].startswith("s_"):
        return 1
    return None


def _target(step: PDDLPlanStep):
    for p in step.parameters:
        if p.startswith("loc_"):
            return parse_loc(p)
    return None


def _options(kitchen: Kitchen, step: PDDLPlanStep, reach) -> list:
    """Equivalent tiles for a storage, bin or hatch step."""
    if step.name in STORAGE_STEPS:
        food = step.parameters[2]
        found = [
            loc
            for loc, storage in kitchen.storages.items()
            if _food_sym(storage.food_name) == food
        ]
    elif step.name in BIN_STEPS:
        found = list(kitchen.bins)
    elif step.name in DELIVERY_STEPS:
        found = list(kitchen.deliveries)
    else:
        return []
    return [loc for loc in found if loc in reach]


def walk(kitchen: Kitchen, start, at, target) -> float:
    """Steps from beside ``at`` (or from ``start``) to beside ``target``."""
    dist = kitchen.distances.get(target, {})
    tiles = [start] if at is None else [n for n in neighbours(at) if n in kitchen.walkable]
    best = min((dist[t] for t in tiles if t in dist), default=None)
    return 30.0 if best is None else float(best)


def plan_cost(kitchen: Kitchen, chef, steps: list[PDDLPlanStep], crowded=frozenset()) -> float:
    """Steps the chef walks and acts for ``steps``, picking the nearest of
    equivalent storages as it goes (as the dispatcher will)."""
    start = (chef.x, chef.y)
    reach = kitchen.reach.get(chef.id, set())
    at = None
    total = 0.0
    for step in steps:
        target = _target(step)
        if target is None:
            continue
        options = _options(kitchen, step, reach)
        if options:
            target = min(options, key=lambda o: (walk(kitchen, start, at, o), o))
        total += walk(kitchen, start, at, target) + 1
        if target in crowded:
            total += CROWDED
        at = target
    return total


def plan_ticks(kitchen: Kitchen, chef, steps: list[PDDLPlanStep]) -> float:
    """Rough time to carry ``steps`` out: walking, acting and some waiting
    for food that cooks by itself."""
    ticks = plan_cost(kitchen, chef, steps)
    for step in steps:
        if step.name == "cook_by_itself":
            station = kitchen.station_by_symbol.get(step.parameters[0])
            if station is not None:
                ticks += station.entity.cook_time / 2
    return ticks


def relocate_stations(kitchen: Kitchen, chef, steps: list[PDDLPlanStep], environment):
    """``steps`` with its stations swapped for equivalent empty ones where
    that shortens the walk."""
    from simulator.entities import Agent

    used: list[str] = []
    for step in steps:
        index = _station_param(step)
        if index is not None and step.parameters[index] not in used:
            used.append(step.parameters[index])
    if not used:
        return steps
    fixed = {step.parameters[1] for step in steps if step.name in MOVE_STATION_STEPS}
    reach = kitchen.reach.get(chef.id, set())
    others = {
        (a.x, a.y)
        for a in environment.get_entities_of_type(Agent)
        if a.id != chef.id and a.x is not None
    }
    crowded = frozenset(
        st.location
        for st in kitchen.stations
        if st.location is not None and any(n in others for n in neighbours(st.location))
    )

    def alternatives(symbol: str) -> list[str]:
        station = kitchen.station_by_symbol.get(symbol)
        if (
            station is None
            or station.location is None
            or symbol in fixed
            or kitchen.station_food(station) is not None
        ):
            return []
        return [
            other.symbol
            for other in kitchen.stations
            if other.symbol not in used
            and other.location is not None
            and other.location in reach
            and other.name == station.name
            and other.supported == station.supported
            and other.entity.held_item_id is None
        ]

    def apply(mapping: dict[str, str]) -> list[PDDLPlanStep]:
        out = []
        for step in steps:
            index = _station_param(step)
            if index is None or mapping.get(step.parameters[index], step.parameters[index]) == step.parameters[index]:
                out.append(step)
                continue
            new = kitchen.station_by_symbol[mapping[step.parameters[index]]]
            params = list(step.parameters)
            params[index] = new.symbol
            if index == 1:
                params[2] = loc_symbol(new.location)
            out.append(PDDLPlanStep(step.name, tuple(params)))
        return out

    mapping = {symbol: symbol for symbol in used}
    best = plan_cost(kitchen, chef, steps, crowded)
    for _ in range(2):
        improved = False
        for symbol in used:
            for alt in alternatives(symbol):
                if alt in mapping.values():
                    continue
                trial = dict(mapping)
                trial[symbol] = alt
                cost = plan_cost(kitchen, chef, apply(trial), crowded)
                if cost < best - 0.5:
                    best, mapping, improved = cost, trial, True
        if not improved:
            break
    return apply(mapping)


# --- Moving --------------------------------------------------------------------


def foreign_tiles(floor: set, ours: list, theirs: list) -> frozenset:
    """Floor tiles nearer the other team's chefs than ours."""
    if not ours or not theirs:
        return frozenset()

    def spread(starts):
        dist = {s: 0 for s in starts if s in floor}
        frontier = list(dist)
        while frontier:
            nxt = []
            for tile in frontier:
                for n in neighbours(tile):
                    if n in floor and n not in dist:
                        dist[n] = dist[tile] + 1
                        nxt.append(n)
            frontier = nxt
        return dist

    near_us, near_them = spread(ours), spread(theirs)
    return frozenset(
        t for t in floor if near_them.get(t, 10**6) < near_us.get(t, 10**6)
    )


def route(floor: set, foreign: frozenset, start, target, blocked=frozenset()):
    """First step and cost of the cheapest walk from ``start`` to a free tile
    beside ``target``, never through ``blocked``. ``(None, None)`` if there
    is no way."""
    goals = {n for n in neighbours(target) if n in floor and n not in blocked}
    if not goals:
        return None, None
    best = {start: 0}
    first = {start: None}
    heap = [(0, start)]
    while heap:
        cost, tile = heapq.heappop(heap)
        if cost > best.get(tile, 10**9):
            continue
        if tile in goals and tile != start:
            return first[tile], cost
        for n in neighbours(tile):
            if n not in floor or n in blocked:
                continue
            step = FOREIGN_COST if n in foreign else 1
            if cost + step < best.get(n, 10**9):
                best[n] = cost + step
                first[n] = n if tile == start else first[tile]
                heapq.heappush(heap, (cost + step, n))
    return None, None
