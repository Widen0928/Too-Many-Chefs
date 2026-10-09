"""Describe a kitchen as a PDDL problem for ``overcooked.pddl``.

``DefaultPDDLController`` creates one ``DefaultProblemGenerator`` and calls
``to_pddl`` in a worker process whenever it needs a plan; ``should_replan``
runs in the simulator process every tick.

``build_problem`` reads the kitchen through ``Kitchen`` (kitchen.py) and adds
objects, initial facts and goals with one ``describe_*`` function per aspect
of the kitchen. Recipe rules are compiled into static facts here, following
the order in which the simulator itself resolves a Combine, so that every
action the planner can choose is one the simulator will accept.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

from controllers.pddl.interfaces import PDDLProblem, ProblemGenerator
from controllers.pddl.problem_builder import ProblemBuilder
from simulator.entities import Food, OvercookedState, Plate
from simulator.environment import Environment

from .config import CONFIG
from .kitchen import (
    DIRTY_PLATE,
    EMPTY_PLATE,
    Kitchen,
    chef_symbol,
    food_symbol,
    loc_symbol,
    plate_symbol,
)

DOMAIN_PATH = Path(__file__).with_name("overcooked.pddl")


class DefaultProblemGenerator(ProblemGenerator):
    """Build problems, and replan when the orders change or the kitchen stalls."""

    # Ticks without any change to the kitchen's items before replanning.
    STALL_TICKS = 40
    # Consecutive calls on the same timestep (nobody can act) before replanning.
    FROZEN_CALLS = 3
    # Replans in a row that make no progress before giving up on replanning.
    MAX_STUCK_REPLANS = 3

    def __init__(self) -> None:
        self._orders: list[str] | None = None
        self._expired = 0
        self._delivered = 0
        self._last_signature: tuple | None = None
        self._last_change_timestep = 0
        self._last_timestep: int | None = None
        self._frozen_calls = 0
        self._waste: int | None = None
        # Orders a plan may cover (None: all). Set before replanning when the
        # chefs keep getting stuck; the planning worker gets a copy.
        self.horizon: int | None = None
        # Set by strategy.py while playing a level out.
        self.forced_horizon = "auto"
        self._stuck_count = 0
        self._stuck_signature: tuple | None = None

    def to_pddl(
        self,
        environment: Environment,
        controlled_agent_ids: Collection[str] | None = None,
    ) -> PDDLProblem:
        kitchen = Kitchen(environment, controlled_agent_ids)
        domain = build_domain()
        last: PDDLProblem | None = None
        horizon = self.horizon or self._strategy_horizon(environment)
        tips = tip_problem(kitchen)
        if tips is not None:
            # No order left can be made: earn the tips the orders on display
            # still owe, instead of stopping.
            problem = PDDLProblem(domain=domain, problem=tips)
            if probe(problem, LATER_PROBE_SECONDS):
                return problem
        for entries, seconds in goal_sets(kitchen, horizon):
            if not entries:
                return PDDLProblem(domain=domain, problem=build_problem(kitchen, entries))
            best: tuple | None = None
            # Two models: one that keeps every tip, and one allowed quicker
            # but tip-losing moves. Each is planned and tried out, and the
            # one that earns more (orders first, then tips, then speed) wins.
            models = {"both": (True, False), "fast": (True,), "safe": (False,)}
            for fast in models.get(CONFIG.models, (True, False)):
                problem = PDDLProblem(
                    domain=domain, problem=build_problem(kitchen, entries, fast=fast)
                )
                last = last or problem
                # The tip-keeping model is much harder to plan in divided
                # kitchens, so it only gets a short try.
                limit = seconds if fast or CONFIG.models == "safe" else min(
                    seconds, CONFIG.tip_safe_seconds
                )
                steps = probe(problem, limit)
                if steps is None:
                    continue
                if not steps:
                    continue
                score = evaluate(environment, steps)
                if best is None or score >= best[0]:
                    best = (score, problem)
            if best is not None:
                return best[1]
        return last or PDDLProblem(domain=domain, problem=build_problem(kitchen, []))

    def _strategy_horizon(self, environment: Environment):
        """The planning horizon: fixed while playing a level out, otherwise
        chosen once per level by playing it out under each option."""
        if self.forced_horizon != "auto":
            return self.forced_horizon
        options = list(CONFIG.select_horizons)
        if len(options) < 2:
            return CONFIG.horizon
        from .strategy import choose_horizon

        try:
            return choose_horizon(environment, options)
        except Exception:  # noqa: BLE001
            return CONFIG.horizon

    def should_replan(self, environment: Environment) -> bool:
        state = environment.get_first_entity_of_type(OvercookedState)
        if state is None:
            return False
        timestep = environment.timestep

        # Expired or unexpected orders: the plan's goals are out of date. A
        # delivery the plan made itself (and the next ticket it reveals) is
        # expected, so the plan, including any food already being prepared
        # for later tickets, is kept.
        queue = state.order_queue
        orders = sorted(e.name for e in [*queue.visible, *queue.pending])
        if self._orders is None:
            self._orders = orders
            self._expired = queue.expired_count
            self._delivered = queue.delivered_count
        else:
            served = queue.delivered_count - self._delivered
            if served and CONFIG.replan_on_delivery:
                return self._reset()
            if (
                queue.expired_count != self._expired
                or not _sub_multiset(orders, self._orders)
                # A replacement ticket (infinite orders) that the plan
                # does not cover, even if it repeats a dish.
                or len(orders) > len(self._orders) - served
            ):
                return self._reset()

        # Nobody can act and nothing is cooking, so time is not passing.
        if timestep == self._last_timestep:
            self._frozen_calls += 1
            if self._frozen_calls >= self.FROZEN_CALLS:
                return self._stuck(environment)
        else:
            self._frozen_calls = 0
        self._last_timestep = timestep

        # Something burnt or spoilt appeared: the plan assumed it would not.
        waste = count_waste(environment, state)
        if self._waste is None:
            self._waste = waste
        elif waste > self._waste:
            return self._reset()

        # Time passes but nothing in the kitchen changes: chefs are stuck.
        signature = kitchen_signature(environment)
        if signature != self._last_signature:
            self._last_signature = signature
            self._last_change_timestep = timestep
        elif timestep - self._last_change_timestep >= self.STALL_TICKS:
            return self._stuck(environment)
        return False

    def _stuck(self, environment: Environment) -> bool:
        """Replan because the chefs are stuck, varying the plan if it recurs.

        If the last replan for being stuck made no progress at all, the same
        plan would most likely get stuck the same way, so the next one covers
        only the first order, which changes who does what. After a few
        fruitless attempts, stop spending planning time on it.
        """
        signature = (environment.timestep, kitchen_signature(environment))
        if signature == self._stuck_signature:
            self._stuck_count += 1
        else:
            self._stuck_count = 1
        self._stuck_signature = signature
        if self._stuck_count > self.MAX_STUCK_REPLANS:
            return False
        stuck_count = self._stuck_count
        self._reset()
        self._stuck_count = stuck_count
        self.horizon = 1 if stuck_count >= 2 else None
        return True

    def _reset(self) -> bool:
        self.horizon = None
        self._stuck_count = 0
        self._orders = None
        self._last_signature = None
        self._last_timestep = None
        self._frozen_calls = 0
        self._waste = None
        return True


def _sub_multiset(small: list[str], big: list[str]) -> bool:
    remaining = list(big)
    for item in small:
        if item not in remaining:
            return False
        remaining.remove(item)
    return True


def count_waste(environment: Environment, state: OvercookedState) -> int:
    """Food that no order, shown or still to come, can use: burnt meat,
    garbage. Food prepared early for a ticket not yet shown is not waste."""
    useful = set(state.visible_orders) | {e.name for e in state.order_queue.pending}
    frontier = list(useful)
    while frontier:
        food = frontier.pop()
        for recipe in state.cook_recipes:
            if recipe.output == food and recipe.ingredient not in useful:
                useful.add(recipe.ingredient)
                frontier.append(recipe.ingredient)
        for recipe in state.combine_recipes:
            if recipe.output == food:
                for ingredient in recipe.ingredients:
                    if ingredient not in useful:
                        useful.add(ingredient)
                        frontier.append(ingredient)
    return sum(
        1 for food in environment.get_entities_of_type(Food) if food.name not in useful
    )


def kitchen_signature(environment: Environment) -> tuple:
    """Where every held item is, ignoring chef positions."""
    from simulator.entities import Agent, Counter, Equipment, Food

    items = []
    for cls in (Agent, Counter, Equipment):
        for entity in environment.get_entities_of_type(cls):
            items.append((entity.id, entity.held_item_id))
    for food in environment.get_entities_of_type(Food):
        items.append((food.id, food.name))
    return tuple(sorted(items, key=str))


# Planning time allowed for checking each goal set, in seconds.
FIRST_PROBE_SECONDS = 10.0
LATER_PROBE_SECONDS = 4.0


def goal_sets(kitchen: Kitchen, horizon: int | None = None):
    """Goal sets to try, best first, each with a time limit for checking it.

    A problem with an unreachable goal gives no plan at all and the chefs
    stand still. So the orders that look feasible are tried first, then the
    same queue with the last order dropped, one at a time, and finally each
    order on its own, without the feasibility check in case it was wrong.
    """
    queue = queued_orders(kitchen)
    feasible = feasible_orders(kitchen, queue)[:horizon]
    tried: list[list] = []
    seconds = FIRST_PROBE_SECONDS
    candidates = [feasible[:n] for n in range(len(feasible), 0, -1)]
    candidates += [[entry] for entry in queue]
    for entries in candidates:
        key = [id(e) for e in entries]
        if key in tried:
            continue
        tried.append(key)
        yield entries, seconds
        seconds = LATER_PROBE_SECONDS
    if not candidates:
        yield [], seconds


def probe(problem: PDDLProblem, seconds: float) -> list | None:
    """The plan the planner finds for ``problem`` within ``seconds``.

    An empty list means no plan; None means the planner could not be run,
    which the controller will report itself. The controller solves the
    chosen problem again afterwards; the search is deterministic, so it
    finds the same plan.
    """
    import hashlib
    import pickle
    import tempfile

    from controllers.pddl.solvers import solve_problem_with_logging

    # Playing a level out (strategy.py) and then running it for real meets
    # the same problems again, so plans are kept on disk.
    cache = Path(tempfile.gettempdir()) / "too-many-chefs-plans"
    key = hashlib.sha256(
        (problem.domain + problem.problem + str(seconds)).encode()
    ).hexdigest()
    try:
        return pickle.loads((cache / key).read_bytes())
    except Exception:  # noqa: BLE001
        pass
    try:
        steps = solve_problem_with_logging(problem, time_limit_seconds=seconds)
    except Exception:  # noqa: BLE001
        return None
    try:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / key).write_bytes(pickle.dumps(steps))
    except Exception:  # noqa: BLE001
        pass
    return steps


def evaluate(environment: Environment, steps: list) -> tuple:
    """Best rollout score of a plan over the dispatch variants (rollout.py)."""
    from .from_pddl import DefaultFromPDDL

    try:
        return DefaultFromPDDL(environment).best(steps)[0]
    except Exception:  # noqa: BLE001 - never let scoring stop planning
        return ()


def build_domain() -> str:
    return DOMAIN_PATH.read_text(encoding="utf-8")


def build_problem(
    kitchen: Kitchen, entries: list | None = None, fast: bool = False
) -> str:
    problem = ProblemBuilder(domain_name="overcooked", problem_name="kitchen")
    describe_food_and_plates(kitchen, problem)
    describe_locations(kitchen, problem)
    describe_chefs(kitchen, problem)
    describe_counters(kitchen, problem)
    describe_stations(kitchen, problem)
    describe_dispensers(kitchen, problem)
    if entries is None:
        entries = feasible_orders(kitchen, queued_orders(kitchen))
    describe_orders(kitchen, problem, entries)
    if fast:
        problem.add_fact("(fast_plating)")
    return problem.to_pddl()


# --- Food and plates -----------------------------------------------------------


def describe_food_and_plates(kitchen: Kitchen, problem: ProblemBuilder) -> None:
    """Declare food and plate states and the rules that relate them.

    - ``placeable``: raw food and any plate may rest on a counter.
    - ``plating p f q``: adding food ``f`` to plate ``p`` gives plate ``q``.
    - ``combine_counter f g o``: held ``f`` on counter food ``g`` makes ``o``,
      which must itself be raw to stay on the counter.
    """
    empty = problem.add_object(EMPTY_PLATE, "pstate")
    dirty = problem.add_object(DIRTY_PLATE, "pstate")
    problem.add_fact(f"(empty_plate {empty})")
    problem.add_fact(f"(dirty_plate {dirty})")
    problem.add_fact(f"(placeable {empty})")
    problem.add_fact(f"(placeable {dirty})")

    for name in kitchen.food_names:
        food = problem.add_object(food_symbol(name), "food")
        plate = problem.add_object(plate_symbol(name), "pstate")
        problem.add_fact(f"(placeable {plate})")
        problem.add_fact(f"(plate_of {plate} {food})")
        problem.add_fact(f"(plating {empty} {food} {plate})")
        if kitchen.is_raw(name):
            problem.add_fact(f"(placeable {food})")

    for incoming in kitchen.food_names:
        for existing in kitchen.food_names:
            output = kitchen.combine(incoming, existing)
            if output is None:
                continue
            problem.add_fact(
                f"(plating {plate_symbol(existing)} {food_symbol(incoming)}"
                f" {plate_symbol(output)})"
            )
            if kitchen.is_raw(output):
                problem.add_fact(
                    f"(combine_counter {food_symbol(incoming)}"
                    f" {food_symbol(existing)} {food_symbol(output)})"
                )


# --- Places --------------------------------------------------------------------


def describe_locations(kitchen: Kitchen, problem: ProblemBuilder) -> None:
    """Declare every tile a controlled chef can act on, and what it is."""
    reachable = set().union(*kitchen.reach.values()) if kitchen.reach else set()
    for location in sorted(kitchen.targets & reachable):
        loc = problem.add_object(loc_symbol(location), "loc")
        if location in kitchen.storages:
            food = food_symbol(kitchen.storages[location].food_name)
            problem.add_fact(f"(storage {loc} {food})")
        if location in kitchen.deliveries:
            problem.add_fact(f"(delivery {loc})")
        if location in kitchen.bins:
            problem.add_fact(f"(bin {loc})")
        if location in kitchen.sinks:
            problem.add_fact(f"(sink {loc})")


def describe_chefs(kitchen: Kitchen, problem: ProblemBuilder) -> None:
    """Declare each controlled chef, where it can reach and what it holds.

    A chef holding a pan, pot or bowl is given neither ``hand_empty`` nor
    ``holds``, so the plan leaves it alone; the controller replans once it
    has put the station down.
    """
    for chef in kitchen.chefs:
        c = problem.add_object(chef_symbol(chef.id), "chef")
        for location in sorted(kitchen.reach[chef.id] & kitchen.targets):
            problem.add_fact(f"(reach {c} {loc_symbol(location)})")
        if chef.held_item_id is None:
            problem.add_fact(f"(hand_empty {c})")
            continue
        held = kitchen.item_symbol(chef.held_item_id)
        if held is not None:
            problem.add_fact(f"(holds {c} {held})")


def describe_counters(kitchen: Kitchen, problem: ProblemBuilder) -> None:
    """Record free counters and the food and plates standing on counters.

    A counter carrying a station (cutboard, stove, mixer) is never free.
    """
    for location, counter in sorted(kitchen.counters.items()):
        if loc_symbol(location) not in problem.objects:
            continue
        if counter.empty(kitchen.environment):
            problem.add_fact(f"(free {loc_symbol(location)})")
            continue
        item = kitchen.item_symbol(counter.held_item_id)
        if item is not None:
            problem.add_fact(f"(on {item} {loc_symbol(location)})")
    for location, item_id in sorted(kitchen.loose_items.items()):
        item = kitchen.item_symbol(item_id)
        if item is not None and loc_symbol(location) in problem.objects:
            problem.add_fact(f"(on {item} {loc_symbol(location)})")


def describe_stations(kitchen: Kitchen, problem: ProblemBuilder) -> None:
    """Declare stations, their contents and the recipe rules for each.

    The rules follow the simulator's resolution of a Combine:

    - held food into an empty station: ``accepts``;
    - held food into a station holding food: the combined food stays inside
      if the station accepts it (``merge_in``), otherwise the chef takes it
      (``merge_hand``);
    - a held plate facing a station first tries to pour its own food in, and
      only then takes the station's food, so ``st_to_plate`` excludes plates
      whose food would be poured in instead.
    """
    foods = kitchen.food_names
    for station in kitchen.stations:
        if station.holder is not None:
            # Carried by a chef: it can be set down, then used.
            if chef_symbol(station.holder) not in problem.objects:
                continue
            s = problem.add_object(station.symbol, "station")
            problem.add_fact(f"(holds_st {chef_symbol(station.holder)} {s})")
        elif loc_symbol(station.location) not in problem.objects:
            continue
        else:
            s = problem.add_object(station.symbol, "station")
            problem.add_fact(f"(st_at {s} {loc_symbol(station.location)})")
        name = station.name
        burns = kitchen.burns(station)
        if not station.entity.portable:
            problem.add_fact(f"(fixed {s})")
        elif not station.supported:
            # Only a station left off its stove or mixer may be moved; the
            # others stay put, which keeps pointless carrying out of plans.
            problem.add_fact(f"(portable {s})")
            counter = kitchen.counters.get(station.location) if station.location else None
            if counter is not None and counter.held_item_id == station.entity.id and not station.supported:
                problem.add_fact(f"(st_on_counter {s} {loc_symbol(station.location)})")
            for support in kitchen.supports_for(station):
                if loc_symbol(support) not in problem.objects:
                    continue
                problem.add_fact(f"(support_for {s} {loc_symbol(support)})")
                if kitchen.base_free(support):
                    problem.add_fact(f"(base_free {loc_symbol(support)})")
        if station.supported:
            problem.add_fact(f"(usable {s})")
        if not burns:
            problem.add_fact(f"(open_station {s})")
        contents = kitchen.station_food(station)
        if contents is not None and burns:
            # Already cooking: any chef may collect it.
            for chef in kitchen.chefs:
                problem.add_fact(f"(tending {chef_symbol(chef.id)} {s})")
        if contents is None:
            problem.add_fact(f"(st_empty {s})")
        else:
            problem.add_fact(f"(in {food_symbol(contents)} {s})")

        for food in foods:
            if kitchen.legal_accepts(food, name):
                problem.add_fact(f"(accepts {s} {food_symbol(food)})")
            output = kitchen.state.get_cook_output(food, name)
            if output is not None:
                kind = "cook_auto" if station.entity.cooks_by_itself else "cook_by_hand"
                problem.add_fact(
                    f"({kind} {s} {food_symbol(food)} {food_symbol(output)})"
                )

        for incoming in foods:
            for existing in foods:
                output = kitchen.combine(incoming, existing)
                if output is None:
                    continue
                args = f"{food_symbol(incoming)} {food_symbol(existing)} {food_symbol(output)}"
                if kitchen.legal_accepts(output, name):
                    problem.add_fact(f"(merge_in {s} {args})")
                elif not kitchen.sim_accepts(output, name):
                    problem.add_fact(f"(merge_hand {s} {args})")

        for food in foods:
            # An empty plate always takes the station's food.
            problem.add_fact(
                f"(st_to_plate {s} {EMPTY_PLATE} {food_symbol(food)} {plate_symbol(food)})"
            )
            if burns:
                # Food that can burn is collected on an empty plate, so the
                # chef collecting it needs nothing else first.
                continue
            for on_plate in foods:
                output = kitchen.combine(food, on_plate)
                if output is None:
                    continue
                # Would the plate's food be poured into the station instead?
                if kitchen.sim_accepts(output, name):
                    continue
                problem.add_fact(
                    f"(st_to_plate {s} {plate_symbol(on_plate)} {food_symbol(food)}"
                    f" {plate_symbol(output)})"
                )


def describe_dispensers(kitchen: Kitchen, problem: ProblemBuilder) -> None:
    """Count the plates in each dispenser and say where served plates go."""
    limit = kitchen.plate_count + 1
    counts = [problem.add_object(f"n{i}", "count") for i in range(limit + 1)]
    problem.add_fact(f"(zero {counts[0]})")
    for lower, higher in zip(counts, counts[1:]):
        problem.add_fact(f"(succ {lower} {higher})")

    for location, dispenser in sorted(kitchen.dispensers.items()):
        loc = loc_symbol(location)
        if loc not in problem.objects:
            continue
        if dispenser.infinite_supply:
            problem.add_fact(f"(infinite_dispenser {loc})")
            continue
        problem.add_fact(f"(dispenser {loc})")
        problem.add_fact(f"(clean_n {loc} {counts[min(dispenser.plate_count, limit)]})")
        problem.add_fact(
            f"(dirty_n {loc} {counts[min(dispenser.dirty_plate_count, limit)]})"
        )

    target = kitchen.return_dispenser
    if target is None or target.infinite_supply or target.x is None or target.y is None:
        problem.add_fact("(returns_nowhere)")
        return
    loc = loc_symbol((target.x, target.y))
    if loc not in problem.objects:
        # Plates go back to a dispenser no controlled chef can reach.
        problem.add_fact("(returns_nowhere)")
        return
    kind = "returns_dirty" if target.returns_dirty else "returns_clean"
    problem.add_fact(f"({kind} {loc})")


# --- Orders --------------------------------------------------------------------


def queued_orders(kitchen: Kitchen) -> list:
    """The orders to plan for, in the order they will be served.

    Orders are always served one after another: only the first is open and
    serving it opens the next. When the kitchen lets orders be served in any
    order, they are queued earliest deadline first.

    Fixing the order costs little, because each chef runs its own steps in
    parallel with the others; a chef can start the second dish while the
    first is still being finished. It makes search far easier, since
    otherwise two dishes of the same kind are interchangeable at every step
    and the heuristic plateaus (two black forest cakes in four rooms: 900k
    expanded states with all orders open, 231 in turn).
    """
    queue = kitchen.state.order_queue
    now = kitchen.environment.timestep
    entries = [
        entry
        for entry in queue.visible
        if entry.time_limit is None or now <= entry.revealed_at + entry.time_limit
    ]
    in_turn = queue.strict_ordering or queue.order_reveal == "sequential"
    if queue.order_reveal == "sequential":
        # Tickets still to come are known to the kitchen. Serving them is
        # chained after the visible one, exactly as they will be revealed, so
        # chefs can prepare the next dish early; its delivery simply waits
        # until the ticket appears.
        entries = entries + list(queue.pending)
    if not in_turn:

        def deadline(e) -> float:
            return float("inf") if e.time_limit is None else e.revealed_at + e.time_limit

        if CONFIG.order_sort == "reward":
            entries.sort(key=lambda e: (-e.reward, deadline(e)))
        else:
            entries.sort(key=deadline)
    return entries


def describe_orders(kitchen: Kitchen, problem: ProblemBuilder, entries: list) -> None:
    """One order object per planned order, each with a goal to serve it;
    serving one opens the next (see ``queued_orders``)."""
    end = problem.add_object("o_end", "order")
    previous: str | None = None
    for index, entry in enumerate(entries):
        order = problem.add_object(f"o{index}", "order")
        problem.add_fact(f"(wants {order} {plate_symbol(entry.name)})")
        if previous is None:
            problem.add_fact(f"(open {order})")
        if previous is not None:
            problem.add_fact(f"(next_order {previous} {order})")
        problem.add_goal(f"(served {order})")
        previous = order
    if previous is not None:
        problem.add_fact(f"(next_order {previous} {end})")


def tip_problem(kitchen: Kitchen, most: int = 3) -> str | None:
    """A problem for the tips the orders on display still owe, once none of
    them can be made and no more are to come; None otherwise.

    Each visible order pays one tip for every food on its recipe that is
    fetched from a storage or prepared. An order that lacks an ingredient
    (a cherry, say) cannot be served, but most of its recipe can still be
    made. The goal is to leave up to ``most`` such foods on free counters,
    chosen greedily to cover as many owed tips as possible. Only storages
    (which never run out) are used, never food already in the kitchen, so
    nothing another order could need is spent.
    """
    state = kitchen.state
    queue = state.order_queue
    visible = list(queue.visible)
    if not visible or queue.pending or feasible_orders(kitchen, queued_orders(kitchen)):
        return None
    team = next((c.team for c in kitchen.chefs if c.team), None)
    paid = queue.tipped.get(team or "", {})
    owed = set()
    for food in kitchen.food_names:
        demand = sum(
            entry.name == food or food in state._order_components(entry.name)
            for entry in visible
        )
        if demand > paid.get(food, 0):
            owed.add(food)
    reachable = set().union(*kitchen.reach.values()) if kitchen.reach else set()
    infinite = {
        storage.food_name for loc, storage in kitchen.storages.items() if loc in reachable
    }
    equipment = {
        station.name
        for station in kitchen.stations
        if station.location in reachable or station.holder is not None
    }
    memo: dict[str, frozenset | None] = {}

    def cover(food: str, depth: int = 0) -> frozenset | None:
        """The owed foods made on the way to ``food`` by its best route, or
        None if it cannot be made from storages."""
        if food in memo:
            return memo[food]
        if depth > 12:
            return None
        memo[food] = None  # guards against recipe loops
        options = []
        if food in infinite:
            options.append(frozenset({food}) & owed)
        for recipe in state.cook_recipes:
            if recipe.output == food and recipe.equipment in equipment:
                inner = cover(recipe.ingredient, depth + 1)
                if inner is not None:
                    options.append(inner | (frozenset({food}) & owed))
        for recipe in state.combine_recipes:
            if recipe.output != food:
                continue
            first, *rest = sorted(recipe.ingredients)
            a = cover(first, depth + 1)
            b = cover(rest[0] if rest else first, depth + 1)
            if a is not None and b is not None:
                options.append(a | b | (frozenset({food}) & owed))
        best = max(options, key=len) if options else None
        memo[food] = best
        return best

    covers = {}
    for food in kitchen.food_names:
        memo.clear()
        found = cover(food)
        if found:
            covers[food] = found
    free = [
        loc
        for loc, counter in sorted(kitchen.counters.items())
        if loc in reachable and counter.empty(kitchen.environment)
    ]
    targets: list[str] = []
    covered: set = set()
    while covers and len(targets) < min(most, len(free)):
        food = max(sorted(covers), key=lambda f: len(covers[f] - covered))
        if not covers[food] - covered:
            break
        targets.append(food)
        covered |= covers.pop(food)
    if not targets:
        return None
    problem = ProblemBuilder(domain_name="overcooked", problem_name="kitchen")
    describe_food_and_plates(kitchen, problem)
    describe_locations(kitchen, problem)
    describe_chefs(kitchen, problem)
    describe_counters(kitchen, problem)
    describe_stations(kitchen, problem)
    describe_dispensers(kitchen, problem)
    describe_orders(kitchen, problem, [])
    text = problem.to_pddl()
    head = text[: text.index("(:goal")]
    # Some foods can only be taken out of their station onto a plate, so
    # each target may stand on its counter loose or on a plate. A target
    # is kept if the planner can still reach it together with the ones
    # kept before.
    goals: list[str] = []
    for food, loc in zip(targets, free):
        if loc_symbol(loc) not in problem.objects:
            continue
        for item in (food_symbol(food), plate_symbol(food)):
            trial = goals + [f"(on {item} {loc_symbol(loc)})"]
            candidate = head + "(:goal (and " + " ".join(trial) + "))\n)\n"
            if probe(PDDLProblem(domain=build_domain(), problem=candidate), LATER_PROBE_SECONDS):
                goals = trial
                break
    if not goals:
        return None
    return head + "(:goal (and " + " ".join(goals) + "))\n)\n"


def feasible_orders(kitchen: Kitchen, entries: list) -> list:
    """Keep the orders the kitchen can still make, in queue order.

    A goal the planner cannot reach makes it find no plan at all, so orders
    that cannot be made are left out. Ingredients from a storage never run
    out; food already in the kitchen (on counters, in stations, on plates or
    in hands) can be used once. Orders are checked in turn and each feasible
    one reserves the finite food it needs, so with two loose cherries only
    two cherry dishes are planned.
    """
    reachable = set().union(*kitchen.reach.values()) if kitchen.reach else set()
    infinite = {
        storage.food_name
        for loc, storage in kitchen.storages.items()
        if loc in reachable
    }
    equipment = {
        station.name
        for station in kitchen.stations
        if station.location in reachable or station.holder is not None
    }
    stock: dict[str, int] = {}
    for food in kitchen.environment.get_entities_of_type(Food):
        stock[food.name] = stock.get(food.name, 0) + 1

    def options(food: str, stock: tuple, depth: int) -> set[tuple]:
        """Every stock that could be left after making one ``food``."""
        have = dict(stock)
        results: set[tuple] = set()
        if have.get(food, 0) > 0:
            have[food] -= 1
            results.add(tuple(sorted((k, v) for k, v in have.items() if v)))
        if food in infinite:
            results.add(stock)
        if depth > 12 or len(results) and food in infinite:
            return results
        for recipe in kitchen.state.cook_recipes:
            if recipe.output == food and recipe.equipment in equipment:
                results |= options(recipe.ingredient, stock, depth + 1)
        for recipe in kitchen.state.combine_recipes:
            if recipe.output != food:
                continue
            first, *rest = sorted(recipe.ingredients)
            second = rest[0] if rest else first
            for left in options(first, stock, depth + 1):
                results |= options(second, left, depth + 1)
            if len(results) > 64:
                break
        return results

    plates = plate_supply(kitchen, reachable)
    start = tuple(sorted(stock.items()))
    memo: dict = {}

    def best(i: int, stock: tuple, plates: int | None) -> tuple[int, ...]:
        """Indices of the most orders from ``i`` on that can still be made,
        preferring earlier ones."""
        if i == len(entries) or plates == 0:
            return ()
        key = (i, stock, plates)
        if key in memo:
            return memo[key]
        chosen = best(i + 1, stock, plates)
        for left in sorted(options(entries[i].name, stock, 0)):
            rest = best(i + 1, left, None if plates is None else plates - 1)
            if len(rest) + 1 > len(chosen):
                chosen = (i, *rest)
                if len(chosen) == len(entries) - i:
                    break
        memo[key] = chosen
        return chosen

    return [entries[i] for i in best(0, start, plates)]


def plate_supply(kitchen: Kitchen, reachable: set) -> int | None:
    """How many dishes the plates allow, or None for no limit.

    Served plates go back to the first dispenser; if a chef can reach it,
    plates are never used up. Otherwise each dish uses up one plate a chef
    can get at: on a reachable counter, in a chef's hands or in a reachable
    dispenser (dirty ones only where a sink is reachable).
    """
    target = kitchen.return_dispenser
    if target is not None and (target.x, target.y) in reachable:
        return None
    can_wash = any(loc in reachable for loc in kitchen.sinks)
    count = 0
    for loc, dispenser in kitchen.dispensers.items():
        if loc not in reachable:
            continue
        if dispenser.infinite_supply:
            return None
        count += dispenser.plate_count + (dispenser.dirty_plate_count if can_wash else 0)
    held = {chef.held_item_id for chef in kitchen.chefs}
    for plate in kitchen.environment.get_entities_of_type(Plate):
        if plate.dirty and not can_wash:
            continue
        if plate.id in held:
            count += 1
        elif plate.x is not None and (plate.x, plate.y) in reachable:
            count += 1
    return count
