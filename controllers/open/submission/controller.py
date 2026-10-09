"""Part 4 controller: one chef in a 2v2 match, planning with our PDDL model.

Each tick the controller must answer within 200 ms, and an exception loses
the set, so:

- Planning runs on a background thread. It picks one order to chase, plans
  it for this chef alone with the PDDL model from Parts 1-3 (copied into
  ``planner/``, since the contest mounts only this folder), and turns the plan
  into simulator actions. Until a plan is ready the chef simply waits.
- Movement uses its own breadth-first search that walks around the other
  chefs, instead of the shared navigation policy, which can raise when chefs
  block each other.
- Everything is wrapped so that any error means "do nothing this tick".

Choosing what to cook is where the match is won:

- Orders are ranked by reward per expected step, and an order is skipped if
  there is not enough time left to make it.
- Goal recognition: a teammate holding an ingredient of a dish is probably
  making it, so an order for that dish is left to them unless there are more
  such orders than teammates on it. Opponents close to finishing a dish make
  chasing the same single order pointless, so it is skipped too.
- The plan is redone whenever the chosen order disappears (served by someone
  else or expired), the plan runs out, or the chef has been stuck too long.
  The planner sees every item in the kitchen, so it happily uses food left in
  the shared stations, whoever made it.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

from simulator.context import Context
from simulator.controller import Controller
from simulator.entities import ORIENTATION_DELTAS, Agent, Delivery, Food, OvercookedState, Plate
from simulator.environment import Environment
from simulator.mutations import Mutation, MutationWithLocation, TurnAgent
from simulator.mutations.move_agent import MoveAgent
from simulator.mutations.mutation import can_apply_mutation

# Steps each dish takes one chef in the tournament kitchen, measured with the
# planner (planning itself costs a few ticks on top).
EXPECTED_STEPS = {
    "coconut_juice": 30,
    "fruit_salad": 60,
    "fruit_parfait": 80,
    "hamburger": 26,
    "cheeseburger": 42,
    "fish_and_rice_bowl": 42,
    "sushi": 48,
    "chicken_and_chips": 68,
    "black_forest_cake": 68,
}
DEFAULT_STEPS = 60
# Ticks without progress before the plan is redone.
STUCK_TICKS = 18
# Ticks with no way through before the plan is redone.
BLOCKED_TICKS = 5
# Ticks a chef must stand still to count as an obstacle for planning.
PARKED_TICKS = 3
# Which model to plan with first: the quick one (True) or the tip-keeping one.
MODEL_ORDER = (True, False) if __import__("os").environ.get("L4_FAST", "1") == "1" else (False, True)
# Orders planned together (see _make_plan).
ORDERS_PER_PLAN = int(__import__("os").environ.get("L4_ORDERS", "1"))
# Seconds to wait for the planner within a tick (the host allows 200 ms).
# L4_SYNC (local evaluation only) waits for every plan, so runs repeat.
WAIT_SECONDS = None if __import__("os").environ.get("L4_SYNC") else 0.12
# Seconds the planner may take for one order.
PLAN_SECONDS = 3.0


def _short(name: str) -> str:
    return name.rsplit("/", 1)[-1]


class OpenController(Controller):
    """Control one chef in a 2v2 match."""

    def __init__(self) -> None:
        # Each game creates a new controller instance; this state lasts one game.
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._plan: list[MutationWithLocation] = []
        self._target: str | None = None
        self._failed: dict[str, int] = {}
        self._stuck = 0
        self._last: tuple | None = None
        self._idle_until = 0
        self._seen: dict[str, tuple] = {}
        self._still: dict[str, int] = {}
        self._watched_at = -1
        self._last_held: dict[str, int] = {}
        self._blocked_ticks = 0

    # --- Controller interface -----------------------------------------------

    def get_actions(self, environment: Environment, context: Context) -> list[Mutation]:
        try:
            actions = self._act(environment)
        except Exception:  # noqa: BLE001 - never lose a set to an error
            self._plan = []
            actions = []
        if actions:
            return actions
        # Idle: answer with a turn on the spot, so every tick gets an action.
        try:
            chef = self.controllable_agents(environment)[0]
            return [TurnAgent(agent_id=chef.id, orientation=chef.orientation)]
        except Exception:  # noqa: BLE001
            return []

    def is_busy(self) -> bool:
        return False

    def has_finished(self, environment: Environment) -> bool:
        return False

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # --- Each tick ---------------------------------------------------------------

    def _act(self, environment: Environment) -> list[Mutation]:
        chefs = self.controllable_agents(environment)
        if not chefs:
            return []
        chef = chefs[0]
        state = environment.get_first_entity_of_type(OvercookedState)
        if state is None or chef.x is None:
            return []
        if environment.timestep != self._watched_at:
            self._watched_at = environment.timestep
            self._watch(environment)

        if self._future is not None and not self._future.done():
            # The host waits for our answer (up to 200 ms) before the kitchen
            # moves on, so waiting here costs no game time; idling would.
            try:
                self._future.result(timeout=WAIT_SECONDS)
            except Exception:  # noqa: BLE001 - timeouts included
                pass
        if self._future is not None and self._future.done():
            try:
                plan, target = self._future.result()
            except Exception:  # noqa: BLE001
                plan, target = [], None
            self._future = None
            self._plan, self._target = plan, target
            self._stuck = 0
            if not plan:
                # Nothing worth doing right now: look again shortly.
                self._idle_until = environment.timestep + 3
            if not plan and target:
                self._failed[target] = environment.timestep

        visible = {entry.name for entry in state.order_queue.visible}
        if self._future is None and environment.timestep >= self._idle_until and (
            not self._plan
            or (self._target is not None and self._target not in visible)
            or self._stuck > STUCK_TICKS
        ):
            self._plan = []
            self._future = self._executor.submit(
                self._make_plan, environment, chef.id, self._parked(environment, chef.id)
            )
            try:
                self._future.result(timeout=WAIT_SECONDS)
            except Exception:  # noqa: BLE001
                pass
            if self._future.done() and not getattr(self, "_retrying", False):
                self._retrying = True
                try:
                    return self._act(environment)
                finally:
                    self._retrying = False
        if not self._plan:
            return []
        return self._follow(environment, chef)

    def _watch(self, environment: Environment) -> None:
        """Count how long each chef has stood on the same tile."""
        for agent in environment.get_entities_of_type(Agent):
            if agent.held_item_id is not None:
                self._last_held[agent.id] = environment.timestep
            where = (agent.x, agent.y)
            if self._seen.get(agent.id) == where:
                self._still[agent.id] = self._still.get(agent.id, 0) + 1
            else:
                self._still[agent.id] = 0
            self._seen[agent.id] = where

    def _parked(self, environment: Environment, chef_id: str) -> set:
        """Tiles of other chefs that have not moved for a while: plan around them."""
        return {
            self._seen[agent_id]
            for agent_id, ticks in self._still.items()
            if agent_id != chef_id and ticks >= PARKED_TICKS
        }

    def _follow(self, environment: Environment, chef: Agent) -> list[Mutation]:
        step = self._plan[0]
        mutation = step.mutation
        target = step.location
        position = (chef.x, chef.y)
        progress = (position, len(self._plan))
        self._stuck = self._stuck + 1 if progress == self._last else 0
        self._last = progress
        if target is None:
            self._plan.pop(0)
            return [mutation]
        if abs(position[0] - target[0]) + abs(position[1] - target[1]) == 1:
            facing = _facing(position, target)
            actions: list[Mutation] = []
            if chef.orientation != facing:
                actions.append(TurnAgent(agent_id=chef.id, orientation=facing))
            if can_apply_mutation(environment, *actions, mutation):
                self._plan.pop(0)
                return [*actions, mutation]
            return actions  # face it and wait (food still cooking, say)
        move = _next_move(environment, chef, target)
        if move is None:
            self._blocked_ticks += 1
            if self._blocked_ticks > BLOCKED_TICKS:
                self._stuck = STUCK_TICKS + 1  # replan around whoever is in the way
            return []
        self._blocked_ticks = 0
        return [move]

    # --- Planning (background thread) -------------------------------------------

    def _make_plan(self, environment: Environment, chef_id: str, blocked=()):
        from controllers.pddl.interfaces import PDDLProblem
        from controllers.pddl.solvers import solve_problem_with_logging

        from .planner.dispatch import Options, dispatch
        from .planner.from_pddl import DefaultFromPDDL
        from .planner.kitchen import Kitchen
        from .planner.problem_generator import build_domain, build_problem

        kitchen = Kitchen(environment, [chef_id], blocked=frozenset(blocked))
        choices = self._choose_orders(environment, chef_id)
        # Two orders in one plan let the chef fill the time food spends in
        # the oven or fryer with the next dish; one order is the fallback.
        groups = [choices[:2]] if ORDERS_PER_PLAN > 1 and len(choices) > 1 else []
        groups += [[entry] for entry in choices]
        for group in groups:
            entry = group[0]
            started = time.monotonic()
            for fast in MODEL_ORDER:
                problem = PDDLProblem(
                    domain=build_domain(),
                    problem=build_problem(kitchen, group, fast=fast),
                )
                try:
                    steps = solve_problem_with_logging(
                        problem, time_limit_seconds=PLAN_SECONDS
                    )
                except Exception:  # noqa: BLE001
                    steps = []
                if steps:
                    chosen = dispatch(
                        kitchen,
                        steps,
                        Options(solo=True, split_cooks=False, split_plates=False),
                    )
                    plan = DefaultFromPDDL(environment).translate_all(
                        kitchen, chosen, False
                    )
                    plan = [m for m in plan if getattr(m.mutation, "agent_id", chef_id) == chef_id]
                    if plan:
                        return plan, entry.name
                if time.monotonic() - started > PLAN_SECONDS:
                    break
        return [], None

    def _choose_orders(self, environment: Environment, chef_id: str) -> list:
        """Orders worth chasing, best first.

        Every chef in the kitchen (ours, the teammate and the opponents) is
        assumed to go for an order too. Each pair of chef and order gets an
        estimated time to finish, lower when the chef already holds part of
        the dish (goal recognition), and orders are handed out greedily,
        quickest pair first. Our chef chases the order it is handed, so it
        neither races a chef who is ahead on that order nor doubles up with
        its teammate; the other orders follow as fallbacks.
        """
        state = environment.get_first_entity_of_type(OvercookedState)
        now = environment.timestep
        everyone = [a for a in environment.get_entities_of_type(Agent) if a.x is not None]
        me_agent = next(a for a in everyone if a.id == chef_id)
        # Goal recognition: a chef that has not carried anything for a while
        # is not cooking, so it will not take an order from anyone.
        agents = [
            a
            for a in everyone
            if a.id == chef_id
            or (me_agent.team and a.team == me_agent.team)
            or now < 40
            or now - self._last_held.get(a.id, -999) < 60
        ]
        visible = list(state.order_queue.visible)

        def time_left(entry) -> float:
            if entry.time_limit is None:
                return float("inf")
            return entry.revealed_at + entry.time_limit - now

        def eta(agent: Agent, entry) -> float:
            steps = EXPECTED_STEPS.get(_short(entry.name), DEFAULT_STEPS)
            held = _held_foods(environment, agent)
            if held & {entry.name}:
                return 5.0
            if held & _components(state, entry.name):
                steps *= 0.6
            return steps

        open_orders = [
            (i, entry)
            for i, entry in enumerate(visible)
            if time_left(entry) >= EXPECTED_STEPS.get(_short(entry.name), DEFAULT_STEPS) * 0.8
            and now - self._failed.get(entry.name, -999) >= 20
        ]
        hatch = next(
            ((d.x, d.y) for d in environment.get_entities_of_type(Delivery)), (0, 0)
        )
        # Ties are broken the same way by every chef running this code, so
        # teammates reach the same split instead of all chasing one order.
        pairs = sorted(
            (
                eta(agent, entry)
                + 0.3 * (abs(agent.x - hatch[0]) + abs(agent.y - hatch[1]))
                - entry.reward * 0.2,
                agent.id,
                i,
            )
            for agent in agents
            for i, entry in open_orders
        )
        me = next(a for a in agents if a.id == chef_id)
        mates = {a.id for a in agents if a.id != chef_id and me.team and a.team == me.team}
        taken_chefs: set[str] = set()
        taken_by: dict[int, str] = {}
        mine: list = []
        for _, agent_id, i in pairs:
            if agent_id in taken_chefs or i in taken_by:
                continue
            taken_chefs.add(agent_id)
            taken_by[i] = agent_id
            if agent_id == chef_id:
                mine.append(visible[i])
        # Left without an order of our own: race an opponent for one rather
        # than stand idle, but never a teammate.
        rest = sorted(
            (
                entry
                for i, entry in open_orders
                if taken_by.get(i) != chef_id and taken_by.get(i) not in mates
            ),
            key=lambda e: e.reward / EXPECTED_STEPS.get(_short(e.name), DEFAULT_STEPS),
            reverse=True,
        )
        def worth(entry) -> bool:
            return entry.reward / EXPECTED_STEPS.get(_short(entry.name), DEFAULT_STEPS) >= 0.7

        # A poor order of our own comes after racing for a good one.
        ranked = [e for e in mine if worth(e)] + [e for e in rest if worth(e)]
        ranked += [e for e in mine if not worth(e)] + [e for e in rest if not worth(e)]
        ordered, seen = [], set()
        for entry in ranked:
            if entry.name not in seen:
                seen.add(entry.name)
                ordered.append(entry)
        return ordered[:2]


# --- Helpers ---------------------------------------------------------------------


def _facing(position: tuple[int, int], target: tuple[int, int]) -> str:
    dx, dy = target[0] - position[0], target[1] - position[1]
    for orientation, delta in ORIENTATION_DELTAS.items():
        if delta == (dx, dy):
            return orientation
    return "n"


def _next_move(environment: Environment, chef: Agent, target) -> MoveAgent | None:
    """First step of a shortest path to a free tile beside ``target``,
    treating the other chefs as walls."""
    from simulator.mutations.move_agent import is_walkable

    others = {
        (a.x, a.y)
        for a in environment.get_entities_of_type(Agent)
        if a.id != chef.id and a.x is not None
    }
    start = (chef.x, chef.y)

    def free(tile) -> bool:
        return tile not in others and is_walkable(
            environment, tile[0], tile[1], ignore_entity_id={chef.id}, ignore_agents=True
        )

    goals = {
        (target[0] + dx, target[1] + dy)
        for dx, dy in ORIENTATION_DELTAS.values()
    }
    goals = {g for g in goals if free(g) or g == start}
    if not goals:
        return None
    previous = {start: None}
    queue = deque([start])
    while queue:
        tile = queue.popleft()
        if tile in goals and tile != start:
            while previous[tile] != start:
                tile = previous[tile]
            return MoveAgent(agent_id=chef.id, dx=tile[0] - start[0], dy=tile[1] - start[1])
        for dx, dy in ORIENTATION_DELTAS.values():
            nxt = (tile[0] + dx, tile[1] + dy)
            if nxt not in previous and free(nxt):
                previous[nxt] = tile
                queue.append(nxt)
    return None


def _components(state: OvercookedState, dish: str) -> set[str]:
    try:
        return state._order_components(dish) | {dish}
    except Exception:  # noqa: BLE001
        return {dish}


def _held_foods(environment: Environment, agent: Agent) -> set[str]:
    if agent.held_item_id is None:
        return set()
    item = environment.get_entity(agent.held_item_id)
    if isinstance(item, Food):
        return {item.name}
    held = getattr(item, "held_item_id", None)
    food = environment.get_entity_as(held, Food) if held else None
    return {food.name} if food is not None else set()


def _working_on(environment, state, agent: Agent, visible) -> set[str]:
    """Goal recognition: the visible dishes this chef's held food leads to,
    if it leads to exactly one of them."""
    held = _held_foods(environment, agent)
    if not held:
        return set()
    dishes = {
        entry.name for entry in visible if held & _components(state, entry.name)
    }
    return dishes if len(dishes) == 1 else set()


def _holding_dish(environment, agent: Agent, visible) -> set[str]:
    """Dishes a chef is carrying ready to serve."""
    if agent.held_item_id is None:
        return set()
    item = environment.get_entity(agent.held_item_id)
    if not isinstance(item, Plate):
        return set()
    return _held_foods(environment, agent) & {entry.name for entry in visible}
