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

# Steps each dish takes one chef in the tournament kitchen: the quickest plan in
# the library (tools/l4_library.py) plus 15% for other chefs in the way.
EXPECTED_STEPS = {
    "coconut_juice": 23,
    "fruit_salad": 62,
    "fruit_parfait": 76,
    "hamburger": 26,
    "cheeseburger": 37,
    "fish_and_rice_bowl": 31,
    "sushi": 42,
    "chicken_and_chips": 56,
    "black_forest_cake": 62,
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
# A way round another chef is taken at once if it is at most this much
# longer than the direct way; otherwise the chef waits a few ticks first.
DETOUR_SLACK = int(__import__("os").environ.get("L4_SLACK", "2"))
MAX_WAIT = int(__import__("os").environ.get("L4_MAX_WAIT", "8"))
BLOCKED_EXTRA = int(__import__("os").environ.get("L4_BLOCKED_EXTRA", "8"))
# Swap the plan's stations for nearer equivalents.
RELOCATE = __import__("os").environ.get("L4_RELOCATE", "1") == "1"
NEW_WALK = __import__("os").environ.get("L4_WALK", "1") == "1"
USE_LIBRARY = __import__("os").environ.get("L4_LIBRARY", "1") == "1"
# Local evaluation only (with L4_SYNC): ticks each planner call would cost
# in a real match, where the host does not wait for the planner.
CALL_TICKS = int(__import__("os").environ.get("L4_CALL_TICKS", "0"))
# Ticks between two attempts to mend a plan rather than plan afresh.
REPAIR_GAP = 20
REPAIR = __import__("os").environ.get("L4_REPAIR", "1") == "1"
EARLY = __import__("os").environ.get("L4_EARLY", "1") == "1"
# Leave alone orders an opponent is visibly making (holds part of).
RACE_CHECK = __import__("os").environ.get("L4_RACE_CHECK", "0") == "1"
# Call the planner when food already made lies about (it may use it).
USE_MADE = __import__("os").environ.get("L4_MADE", "1") == "1"
# Leave food teammates put in stations or on counters to them, and read
# what they are cooking from it.
RESPECT_MATES = __import__("os").environ.get("L4_RESPECT_MATES", "1") == "1"
ABANDONED_TICKS = 40
# Count the teammate in the order auction (right when it runs this code);
# otherwise only orders it visibly makes are left to it.
MATE_AUCTION = __import__("os").environ.get("L4_MATE_AUCTION", "0") == "1"
# Weight of the ticks our food sits unattended in a shared station when
# choosing between plans (others may take it).
EXPOSED = float(__import__("os").environ.get("L4_EXPOSED", "0"))
# Plan around chefs standing in the way; step aside when face to face.
UNJAM = __import__("os").environ.get("L4_UNJAM", "1") == "1"
# On the way to serve a dish, plan the next one for the kitchen as it will
# be once it is served, so no ticks go by waiting for the planner after.
AHEAD = __import__("os").environ.get("L4_AHEAD", "1") == "1"
# With nothing to do, step off tiles others need.
PARK = __import__("os").environ.get("L4_PARK", "1") == "1"
# Prefer storages, plate dispensers and bins on our own side.
AVOID = __import__("os").environ.get("L4_AVOID", "1") == "1"
# Plan with only the foods that matter for the dish (much quicker), and
# what that costs in a match (ticks, for local evaluation).
PRUNE = __import__("os").environ.get("L4_PRUNE", "1") == "1"
PRUNED_TICKS = float(__import__("os").environ.get("L4_PRUNED_TICKS", "1"))
# After getting stuck, plan the same dish again first.
STICKY = __import__("os").environ.get("L4_STICKY", "0") == "1"
# Local evaluation only: print plans started and dishes served.
# Give up a dish early on once the teammate is visibly making it too (holds
# or has put down food only that dish needs), so the two do not race for
# the same ticket and one of them throw its work away.
YIELD = __import__("os").environ.get("L4_YIELD", "1") == "1"
# Never give a dish up to the teammate while carrying part of it.
HOLD_ON = __import__("os").environ.get("L4_HOLD_ON", "1") == "1"
YIELD_FRACTION = float(__import__("os").environ.get("L4_YIELD_FRACTION", "0.4"))
# Wait beside food that is cooking by itself for as long as it takes, not a
# count of ticks fixed when the wait began.
PATIENT = __import__("os").environ.get("L4_PATIENT", "1") == "1"
# Skip the rough time check for an order part of which is already made: the
# plan's own timing decides.
PROGRESS = __import__("os").environ.get("L4_PROGRESS", "1") == "1"
# Multiplier applied to EXPECTED_STEPS for the rough-time pre-filter in
# _choose_orders.  Orders with time_left < EXPECTED_STEPS * ROUGH_FRAC are
# skipped before the planner is called.  Default 0.8 matches the original
# hard-coded value; lower values admit more near-deadline orders (tested:
# held-ingredient variants hurt mixed mode, reverted).
ROUGH_FRAC = float(__import__("os").environ.get("L4_ROUGH_FRAC", "0.8"))
LOG = __import__("os").environ.get("L4_LOG") == "1"
# Timesteps in a tournament game; nothing served after this counts.
GAME_TICKS = int(__import__("os").environ.get("L4_GAME_TICKS", "300"))
# Drop orders that cannot be served in time even as a last resort.
STRICT = __import__("os").environ.get("L4_STRICT", "1") == "1"
# Matches run slower than the plan's estimate (other chefs in the way).
SLACK = 1.15


def _short(name: str) -> str:
    return name.rsplit("/", 1)[-1]


CPU_TICKS = __import__("os").environ.get("L4_CPU_TICKS") == "1"
# Local evaluation: how much slower the contest host plans than this machine.
CPU_SCALE = float(__import__("os").environ.get("L4_CPU_SCALE", "1"))


def _cpu_now() -> float:
    """CPU seconds used by this process and the planners it has run."""
    import resource

    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


class OpenController(Controller):
    """Control one chef in a 2v2 match."""

    def __init__(self) -> None:
        # Each game creates a new controller instance; this state lasts one game.
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        # The plan for after the dish we are about to serve (see AHEAD).
        self._ahead: Future | None = None
        self._ahead_tick = 0
        self._ahead_cost = 0.0
        self._from_ahead = False
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
        self._waiting = 0
        self._calls = 0
        self._hold_until = 0
        self._held_back = 0
        # The PDDL steps behind the plan, and for each action left in the
        # plan the step it carries out (-1 if not the first action of it).
        self._steps: list = []
        self._step_of: list[int] = []
        self._repaired_at = -999
        self._ticks_by_kind: dict[str, int] = {}
        self._floor: set | None = None
        self._foreign: frozenset = frozenset()
        self._avoid: frozenset = frozenset()
        self._busy_tiles: frozenset | None = None
        # Who put the food in each station or on each counter (by entity
        # id), kept while it stays there; food cooking keeps its owner.
        self._owner: dict[str, str] = {}
        self._contents: dict[str, str] = {}
        self._prev_held: dict[str, set] = {}
        self._owned_since: dict[str, int] = {}
        # Waste-category subcounters (only used when LOG is on).
        self._last_none_tick: int = -999
        self._last_yield_tick: int = -999
        self._block_is_food: bool = False

    # --- Controller interface -----------------------------------------------

    def get_actions(self, environment: Environment, context: Context) -> list[Mutation]:
        # One budget for all waiting on the planner within this tick: waiting
        # for the last plan and then for the next one could otherwise take
        # twice WAIT_SECONDS and miss the host's 200 ms.
        self._deadline = time.monotonic() + (WAIT_SECONDS or 0.0)
        try:
            actions = self._act(environment)
        except Exception:  # noqa: BLE001 - never lose a set to an error
            self._plan = []
            actions = []
        if LOG:
            kinds = {type(a).__name__ for a in actions}
            if "MoveAgent" in kinds:
                kind = "walk"
            elif kinds - {"TurnAgent"}:
                kind = "act"
            elif self._future is not None or environment.timestep < self._hold_until:
                kind = "planning"
            elif not self._plan:
                kind = "idle"
            elif kinds:
                kind = "wait"
            else:
                kind = "blocked"
            self._ticks_by_kind[kind] = self._ticks_by_kind.get(kind, 0) + 1
            # Subcategories for waste analysis (l4_waste.py).
            now_t = environment.timestep
            if kind in ("blocked", "wait") and self._block_is_food:
                self._ticks_by_kind["food"] = self._ticks_by_kind.get("food", 0) + 1
            elif kind == "idle":
                if now_t - self._last_none_tick < 5:
                    self._ticks_by_kind["impossible"] = self._ticks_by_kind.get("impossible", 0) + 1
                elif now_t - self._last_yield_tick < 12:
                    self._ticks_by_kind["yield_idle"] = self._ticks_by_kind.get("yield_idle", 0) + 1
            elif kind == "planning" and now_t - self._last_yield_tick < 20:
                self._ticks_by_kind["yield_plan"] = self._ticks_by_kind.get("yield_plan", 0) + 1
        if actions:
            return actions
        # Idle: answer with a turn on the spot, so every tick gets an action.
        try:
            chef = self.controllable_agents(environment)[0]
            return [TurnAgent(agent_id=chef.id, orientation=chef.orientation)]
        except Exception:  # noqa: BLE001
            return []

    def _wait_left(self) -> float | None:
        """Seconds still to wait for the planner this tick (None: wait for it,
        as local evaluation does with L4_SYNC)."""
        if WAIT_SECONDS is None:
            return None
        return max(0.0, getattr(self, "_deadline", 0.0) - time.monotonic())

    def is_busy(self) -> bool:
        return False

    def has_finished(self, environment: Environment) -> bool:
        return False

    def shutdown(self) -> None:
        if LOG:
            name = next(iter(self.controlled_agent_ids), "-") if getattr(self, "controlled_agent_ids", None) else "-"
            print("LOG 0", name, "ticks", " ".join(f"{k}={v}" for k, v in sorted(self._ticks_by_kind.items())), flush=True)
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
        if self._floor is None or (chef.x, chef.y) not in self._floor:
            self._map(environment, chef)

        if self._future is not None and not self._future.done():
            # The host waits for our answer (up to 200 ms) before the kitchen
            # moves on, so waiting here costs no game time; idling would.
            try:
                self._future.result(timeout=self._wait_left())
            except Exception:  # noqa: BLE001 - timeouts included
                pass
        if self._future is not None and self._future.done():
            try:
                plan, target, steps, step_of = self._future.result()
            except Exception:  # noqa: BLE001
                plan, target, steps, step_of = [], None, [], []
            self._future = None
            if LOG:
                print("LOG", environment.timestep, chef.id, "plan", target and _short(target), len(plan),
                      "cpu", round(_cpu_now() - getattr(self, "_cpu_start", _cpu_now()), 3), flush=True)
            self._plan, self._target = plan, target
            self._steps, self._step_of = steps, step_of
            self._held_back = 0
            if CPU_TICKS and WAIT_SECONDS is None:
                # Local evaluation: the ticks a match would lose while this
                # plan was made (the host waits 120 ms of each 130 ms tick).
                import math

                if self._from_ahead:
                    # Made while we walked to the hatch: only what is left over.
                    lost = math.ceil((self._ahead_cost * CPU_SCALE - 0.12) / 0.13) - (environment.timestep - self._ahead_tick)
                else:
                    cost = _cpu_now() - getattr(self, "_cpu_start", _cpu_now())
                    lost = math.ceil((cost * CPU_SCALE - 0.12) / 0.13)
                self._hold_until = environment.timestep + max(0, lost)
            elif CALL_TICKS and WAIT_SECONDS is None:
                self._hold_until = environment.timestep + round(CALL_TICKS * self._calls)
            self._calls = 0
            self._stuck = 0
            self._from_ahead = False
            if not plan:
                # Nothing worth doing right now: look again shortly.
                self._idle_until = environment.timestep + 3
                if LOG:
                    self._last_none_tick = environment.timestep
            if not plan and target:
                self._failed[target] = environment.timestep

        visible = {entry.name for entry in state.order_queue.visible}
        if YIELD and self._future is None and self._plan and self._mate_makes_target(environment, chef, state):
            self._why = "mate"
            self._stuck = STUCK_TICKS + 1
            if LOG:
                self._last_yield_tick = environment.timestep
        if (
            AHEAD
            and self._future is None
            and self._ahead is None
            and len(self._plan) == 1
            and type(self._plan[0].mutation).__name__ == "Deliver"
            and self._plan[0].location is not None
        ):
            self._plan_ahead(environment, chef)
        if self._future is None and environment.timestep >= self._idle_until and (
            not self._plan
            or (self._target is not None and self._target not in visible)
            or self._stuck > STUCK_TICKS
        ):
            if LOG:
                why = (
                    "done" if not self._plan
                    else "gone" if self._target is not None and self._target not in visible
                    else getattr(self, "_why", "") or "slow"
                )
                self._why = ""
                print("LOG", environment.timestep, chef.id, "why", why, len(self._plan), flush=True)
            repair = None
            if (
                REPAIR
                and self._plan
                and self._step_of
                and self._step_of[0] >= 0
                and self._target in visible
                and environment.timestep - self._repaired_at > REPAIR_GAP
            ):
                # Stuck partway: first try the rest of the plan with other
                # stations, which needs no planner call.
                repair = (self._target, self._steps[self._step_of[0]:])
                self._repaired_at = environment.timestep
            # Stuck partway through a dish that is still on order: keep at
            # it (the planner can use what is already made) unless that
            # fails.
            keep = self._target if self._plan and self._target in visible and STICKY else None
            ahead, self._ahead = self._ahead, None
            if ahead is not None and not self._plan:
                # The dish is served and the next plan is (being) made.
                self._future, self._from_ahead = ahead, True
            else:
                self._plan, self._step_of = [], []
                self._cpu_start = _cpu_now()
                self._future = self._executor.submit(
                    self._make_plan, environment, chef.id, self._parked(environment, chef.id), repair, keep,
                    self._teammates_things(environment, chef) if RESPECT_MATES else None,
                )
            try:
                self._future.result(timeout=self._wait_left())
            except Exception:  # noqa: BLE001
                pass
            if self._future.done() and not getattr(self, "_retrying", False):
                self._retrying = True
                try:
                    return self._act(environment)
                finally:
                    self._retrying = False
        if not self._plan and self._future is None and PARK:
            return self._park(environment, chef)
        if not self._plan or environment.timestep < self._hold_until:
            return []
        return self._follow(environment, chef)

    def _plan_ahead(self, environment: Environment, chef: Agent) -> None:
        """Start planning the next dish for the kitchen as it will be once
        the dish in hand is served: the chef stood by the hatch, delivering."""
        step = self._plan[0]
        position = (chef.x, chef.y)
        others = {
            (a.x, a.y) for a in environment.get_entities_of_type(Agent)
            if a.id != chef.id and a.x is not None
        }
        tiles = [t for t in _goal_tiles(step.location) if t in (self._floor or set()) and t not in others]
        if not tiles:
            return
        tile = min(tiles, key=lambda t: abs(t[0] - position[0]) + abs(t[1] - position[1]))
        try:
            after = environment.model_copy(deep=True)
            me = after.get_entity(chef.id)
            me.x, me.y = tile
            me.orientation = _facing(tile, step.location)
            after = after.step([step.mutation])
            if after.get_entity(chef.id).held_item_id is not None:
                return  # the dish would not be taken: plan as usual after
        except Exception:  # noqa: BLE001
            return
        self._ahead_tick = environment.timestep
        args = (
            after, chef.id, self._parked(environment, chef.id), None, None,
            self._teammates_things(environment, chef) if RESPECT_MATES else None,
        )
        if WAIT_SECONDS is None:
            # Local evaluation: make it now and charge its time later.
            start = _cpu_now()
            future: Future = Future()
            try:
                future.set_result(self._make_plan(*args))
            except Exception as error:  # noqa: BLE001
                future.set_exception(error)
            self._ahead_cost = _cpu_now() - start
            self._ahead = future
        else:
            self._ahead = self._executor.submit(self._make_plan, *args)

    def _mate_makes_target(self, environment: Environment, chef: Agent, state) -> bool:
        """Whether a teammate visibly makes the dish we are only starting,
        with no second ticket for it."""
        if self._target is None or not self._steps or not self._step_of:
            return False
        step = self._step_of[0]
        if step < 0 or step >= len(self._steps) * YIELD_FRACTION:
            return False
        visible = list(state.order_queue.visible)
        tickets = sum(1 for entry in visible if entry.name == self._target)
        if tickets == 0:
            return False
        if HOLD_ON and _held_foods(environment, chef) & _components(state, self._target):
            return False  # already carrying part of it: giving up wastes that
        put_down = self._teammates_things(environment, chef)[3] if RESPECT_MATES else {}
        makers = sum(
            1
            for a in environment.get_entities_of_type(Agent)
            if a.id != chef.id and chef.team and a.team == chef.team
            and self._target in _working_on(environment, state, a, visible, put_down.get(a.id, set()))
        )
        return makers >= tickets

    def _park(self, environment: Environment, chef: Agent) -> list[Mutation]:
        """With nothing to do, step off any tile others need to reach a
        storage, plate dispenser, hatch, bin or station."""
        from simulator.entities import Bin, Equipment, PlateDispenser, Storage

        if self._busy_tiles is None:
            self._busy_tiles = frozenset(
                n
                for cls in (Storage, PlateDispenser, Delivery, Bin, Equipment)
                for e in environment.get_entities_of_type(cls)
                if e.x is not None and not isinstance(e, Plate)
                for n in _goal_tiles((e.x, e.y))
            )
        position = (chef.x, chef.y)
        if position not in self._busy_tiles:
            return []
        others = {
            (a.x, a.y)
            for a in environment.get_entities_of_type(Agent)
            if a.id != chef.id and a.x is not None
        }
        floor = self._floor or set()
        # Breadth-first to the nearest free tile on our side that nobody
        # needs to stand on.
        previous = {position: None}
        queue = deque([position])
        while queue:
            tile = queue.popleft()
            if tile not in self._busy_tiles and tile not in self._foreign and tile != position:
                while previous[tile] != position:
                    tile = previous[tile]
                return [_move(chef, position, tile)]
            for n in _goal_tiles(tile):
                if n in floor and n not in others and n not in previous:
                    previous[n] = tile
                    queue.append(n)
        return []

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
        self._track_owners(environment)

    def _track_owners(self, environment: Environment) -> None:
        """Note who put food into each station or onto each counter: the
        chef who held it the tick before it appeared there."""
        from simulator.entities import Counter, Equipment

        agents = list(environment.get_entities_of_type(Agent))
        held_now: dict[str, set] = {}
        for a in agents:
            things = set()
            if a.held_item_id:
                things.add(a.held_item_id)
                inner = environment.get_entity(a.held_item_id)
                if inner is not None and getattr(inner, "held_item_id", None):
                    things.add(inner.held_item_id)
            held_now[a.id] = things
        for cls in (Equipment, Counter):
            for container in environment.get_entities_of_type(cls):
                if isinstance(container, Plate) or container.x is None or container.y is None:
                    # A pot or bowl a chef is carrying has no tile; other
                    # controllers do carry them, food and all. Leave it be
                    # (it gets an owner again where it is put down).
                    continue
                item = container.held_item_id
                entity = environment.get_entity(item) if item else None
                if not isinstance(entity, (Food, Plate)):
                    self._owner.pop(container.id, None)
                    self._contents.pop(container.id, None)
                    continue
                if container.id not in self._contents:
                    inside = {item, getattr(entity, "held_item_id", None)}
                    who = next((a for a, held in self._prev_held.items() if held & inside), None)
                    if who is None:
                        # Food changes identity as it is combined: take the
                        # chef beside the container facing it.
                        who = next(
                            (
                                a.id
                                for a in agents
                                if a.x is not None
                                and _facing((a.x, a.y), (container.x, container.y)) == a.orientation
                                and abs(a.x - container.x) + abs(a.y - container.y) == 1
                            ),
                            None,
                        )
                    if who is not None:
                        self._owner[container.id] = who
                        self._owned_since[container.id] = environment.timestep
                self._contents[container.id] = item
        self._prev_held = held_now

    def _teammates_things(self, environment: Environment, chef: Agent):
        """What teammates have put down: their stations (as PDDL symbols),
        their counters, and the ids and names of the food in them."""
        from .planner.kitchen import loc_symbol, short_name

        mates = {
            a.id
            for a in environment.get_entities_of_type(Agent)
            if a.id != chef.id and chef.team and a.team == chef.team
        }
        stations, counters, items, foods = set(), set(), set(), {}
        for container_id, owner in self._owner.items():
            if owner not in mates:
                continue
            if environment.timestep - self._owned_since.get(container_id, 0) > ABANDONED_TICKS:
                continue  # left there long ago: free for anyone
            container = environment.get_entity(container_id)
            if container is None or container.x is None:
                continue
            item = environment.get_entity(container.held_item_id) if container.held_item_id else None
            inner = environment.get_entity(item.held_item_id) if isinstance(item, Plate) and item.held_item_id else None
            for thing in (item, inner):
                if thing is not None:
                    items.add(thing.id)
                    if isinstance(thing, Food):
                        foods.setdefault(owner, set()).add(thing.name)
            if type(container).__name__ == "Counter":
                counters.add(loc_symbol((container.x, container.y)))
            else:
                stations.add(f"s_{container.x}_{container.y}_{short_name(container.name)}")
        return frozenset(stations), frozenset(counters), frozenset(items), foods

    def _map(self, environment: Environment, chef: Agent) -> None:
        """The floor, and which part of it is the other team's side."""
        from simulator.entities import Bounds
        from simulator.mutations.move_agent import is_walkable

        from .routing import foreign_tiles

        b = environment.get_first_entity_of_type(Bounds)
        x0, y0 = b.x or 0, b.y or 0
        self._floor = {
            (x, y)
            for x in range(x0, x0 + b.width)
            for y in range(y0, y0 + b.height)
            if is_walkable(environment, x, y, ignore_agents=True)
        }
        agents = [a for a in environment.get_entities_of_type(Agent) if a.x is not None]
        if chef.team is not None and self._foreign == frozenset():
            ours = [(a.x, a.y) for a in agents if a.team == chef.team]
            theirs = [(a.x, a.y) for a in agents if a.team != chef.team]
            self._foreign = foreign_tiles(self._floor, ours, theirs)
            if AVOID and self._foreign:
                from simulator.entities import Bin, PlateDispenser, Storage

                self._avoid = frozenset(
                    (e.x, e.y)
                    for cls in (Storage, PlateDispenser, Bin)
                    for e in environment.get_entities_of_type(cls)
                    if all(
                        n in self._foreign
                        for n in _goal_tiles((e.x, e.y))
                        if n in self._floor
                    )
                )

    def _parked(self, environment: Environment, chef_id: str) -> set:
        """Tiles of other chefs that have not moved for a while: plan around them."""
        return {
            self._seen[agent_id]
            for agent_id, ticks in self._still.items()
            if agent_id != chef_id and ticks >= PARKED_TICKS
        }

    def _follow(self, environment: Environment, chef: Agent) -> list[Mutation]:
        if LOG:
            self._block_is_food = False
        step = self._plan[0]
        mutation = step.mutation
        target = step.location
        position = (chef.x, chef.y)
        progress = (position, len(self._plan))
        self._stuck = self._stuck + 1 if progress == self._last else 0
        self._last = progress
        if target is None:
            self._advance()
            return [mutation]
        if EARLY and self._taken(environment, target):
            self._why = "taken"
            self._stuck = STUCK_TICKS + 1  # the station we meant to use is taken
            return []
        if abs(position[0] - target[0]) + abs(position[1] - target[1]) == 1:
            facing = _facing(position, target)
            actions: list[Mutation] = []
            if chef.orientation != facing:
                actions.append(TurnAgent(agent_id=chef.id, orientation=facing))
            if can_apply_mutation(environment, *actions, mutation):
                if LOG and type(mutation).__name__ == "Deliver":
                    print("LOG", environment.timestep, chef.id, "deliver", flush=True)
                self._advance()
                self._held_back = 0
                return [*actions, mutation]
            # Wait while the food there is still cooking; anything else
            # (another chef's food in the station we meant to use, say)
            # will not sort itself out, so plan again.
            self._held_back += 1
            if PATIENT and _cooking(environment, target):
                self._held_back = 0  # done in a few ticks, whoever's it is
                if LOG:
                    self._block_is_food = True
            if self._held_back > _patience(environment, target):
                if LOG:
                    from simulator.entities import Equipment

                    step = self._steps[self._step_of[0]] if self._step_of and self._step_of[0] >= 0 and self._steps else None
                    there = [
                        (e.name.rsplit("/", 1)[-1], getattr(environment.get_entity(e.held_item_id), "name", None))
                        for e in environment.get_entities_of_type(Equipment)
                        if (e.x, e.y) == tuple(target)
                    ]
                    held = environment.get_entity(chef.held_item_id) if chef.held_item_id else None
                    print("LOG", environment.timestep, chef.id, "pat", type(mutation).__name__,
                          step and step.name, tuple(target), "there", there,
                          "held", getattr(held, "name", None), flush=True)
                self._held_back = 0
                self._why = "patience"
                self._stuck = STUCK_TICKS + 1
            return actions
        return self._walk(environment, chef, target)

    def _side_step(self, position, ahead, others):
        """A free tile beside ``position`` other than the one ``ahead``,
        preferring tiles nobody needs and our own side."""
        floor = self._floor or set()
        options = [
            n for n in _goal_tiles(position)
            if n in floor and n not in others and n != ahead
        ]
        if not options:
            return None
        busy = self._busy_tiles or frozenset()
        return min(options, key=lambda n: (n in busy, n in self._foreign, n))

    def _advance(self) -> None:
        self._plan.pop(0)
        if self._step_of:
            self._step_of.pop(0)

    def _taken(self, environment: Environment, target) -> bool:
        """Whether the next step puts food into a station that now holds
        someone else's food, so it cannot be done there."""
        if not self._step_of or self._step_of[0] < 0 or not self._steps:
            return False
        step = self._steps[self._step_of[0]]
        if step.name not in ("put_in_station", "plate_into_station"):
            return False
        from .planner.kitchen import parse_loc

        location = parse_loc(step.parameters[2])
        if location != tuple(target):
            return False
        from simulator.entities import Equipment

        name = step.parameters[1].split("_", 3)[-1]
        for item in environment.get_entities_of_type(Equipment):
            if (item.x, item.y) == location and item.name.endswith("/" + name):
                # Only food counts: an oven holds the stove stacked on it.
                held = item.held_item_id
                return held is not None and environment.get_entity_as(held, Food) is not None
        return False

    def _walk(self, environment: Environment, chef: Agent, target) -> list[Mutation]:
        """Step towards ``target``. When another chef blocks the direct way
        and the way round is much longer, edge closer and wait a few ticks
        for them to move, as they usually do."""
        from .routing import route

        if not NEW_WALK:
            move = _next_move(environment, chef, target)
            if move is None:
                self._blocked_ticks += 1
                if self._blocked_ticks > BLOCKED_TICKS:
                    self._why = "?"
                    self._stuck = STUCK_TICKS + 1
                return []
            self._blocked_ticks = 0
            return [move]
        position = (chef.x, chef.y)
        others = frozenset(
            (a.x, a.y)
            for a in environment.get_entities_of_type(Agent)
            if a.id != chef.id and a.x is not None
        )
        floor = self._floor or set()
        free_step, free_cost = route(floor, self._foreign, position, target, others)
        direct_step, direct_cost = route(floor, self._foreign, position, target)
        if free_step is not None and (
            direct_cost is None or free_cost <= direct_cost + DETOUR_SLACK
        ):
            self._waiting = 0
            self._blocked_ticks = 0
            return [_move(chef, position, free_step)]
        self._waiting += 1
        if free_step is not None and self._waiting > min(MAX_WAIT, free_cost - direct_cost):
            self._blocked_ticks = 0
            return [_move(chef, position, free_step)]  # waited long enough: go round
        if direct_step is not None and direct_step not in others:
            return [_move(chef, position, direct_step)]
        if free_step is None:
            self._blocked_ticks += 1
            blockers = [
                a
                for a in environment.get_entities_of_type(Agent)
                if a.id != chef.id and a.x is not None
                and ((a.x, a.y) == direct_step or (a.x, a.y) in _goal_tiles(target))
            ]
            if (
                UNJAM
                and blockers
                and self._blocked_ticks >= 3
                and all(self._still.get(a.id, 0) >= PARKED_TICKS for a in blockers)
            ):
                # Whoever is in the way has stood still for a while: plan
                # around them (another storage or station of the kind).
                self._why = "parked"
                self._stuck = STUCK_TICKS + 1
                return []
            head_on = next((a for a in blockers if (a.x, a.y) == direct_step), None)
            if (
                UNJAM
                and head_on is not None
                and self._blocked_ticks >= 2
                and (head_on.team != chef.team or chef.id > head_on.id)
            ):
                # Face to face in a corridor: step aside to let them by.
                side = self._side_step(position, direct_step, others)
                if side is not None:
                    return [_move(chef, position, side)]
            if self._blocked_ticks > BLOCKED_TICKS + BLOCKED_EXTRA:
                self._why = "blocked"
                self._stuck = STUCK_TICKS + 1  # replan around whoever is in the way
        if LOG and self._waiting % 4 == 1:
            who = [
                (a.id, "mate" if a.team == chef.team else "opp", self._still.get(a.id, 0),
                 "held" if a.held_item_id else "empty")
                for a in environment.get_entities_of_type(Agent)
                if (a.x, a.y) == direct_step
                or (a.x, a.y) in [n for n in _goal_tiles(target) if n != position]
            ]
            print("LOG", environment.timestep, chef.id, "blk", position, tuple(target),
                  "free", free_cost, "direct", direct_cost, "by", who, flush=True)
        return []

    # --- Planning (background thread) -------------------------------------------

    def _make_plan(
        self, environment: Environment, chef_id: str, blocked=(), repair=None, keep=None, mates=None
    ):
        """A plan for this chef for the best order it can make now.

        Plans from the library (see library.py) that check out against the
        kitchen as it is come free; the planner is called when none does,
        or when food already made in the kitchen could make a better plan.
        """
        from controllers.pddl.interfaces import PDDLProblem
        from controllers.pddl.solvers import solve_problem_with_logging

        from . import library
        from .planner.dispatch import Options, dispatch
        from .planner.from_pddl import _Translator as _translator
        from .planner.kitchen import Kitchen, chef_symbol
        from .planner.problem_generator import build_domain, build_problem
        from .routing import _retargeted, relocate_stations
        from .schedule import exposure, ticks

        kitchen = Kitchen(environment, [chef_id], blocked=frozenset(blocked))
        me = next(c for c in kitchen.chefs if c.id == chef_id)
        c = chef_symbol(chef_id)
        choices = self._choose_orders(environment, chef_id)
        if keep is not None:
            current = [
                e for e in environment.get_first_entity_of_type(OvercookedState).order_queue.visible
                if e.name == keep
            ][:1]
            choices = current + [e for e in choices if e.name != keep]

        def relocated(steps):
            if not RELOCATE:
                return steps
            try:
                return relocate_stations(kitchen, me, steps, environment, self._avoid)
            except Exception:  # noqa: BLE001 - keep the planner's choice
                return steps

        options = Options(solo=True, split_cooks=False, split_plates=False)

        def finish(steps, entry):
            chosen = dispatch(kitchen, steps, options)
            if self._avoid:
                # The dispatcher picks the nearest storage, the top one on a
                # tie: keep to our own side instead.
                chosen = _retargeted(kitchen, me, chosen, self._avoid)
            translator = _translator(kitchen)
            plan, step_of = [], []
            for index, step in enumerate(chosen):
                actions = [
                    m for m in translator.translate(step)
                    if getattr(m.mutation, "agent_id", chef_id) == chef_id
                ]
                plan += actions
                if actions:
                    step_of += [index] + [-1] * (len(actions) - 1)
            return (plan, entry.name, chosen, step_of) if plan else None

        if repair is not None:
            target, remaining = repair
            entry = next(
                (e for e in environment.get_first_entity_of_type(OvercookedState).order_queue.visible
                 if e.name == target),
                None,
            )
            if entry is not None and remaining:
                fixes = []
                for fast in MODEL_ORDER:
                    init, goal = library.facts(_problem_text(kitchen, entry, fast, build_problem))
                    for steps in (remaining, relocated(remaining)):
                        if library.valid(steps, init, goal):
                            fixes.append(steps)
                for steps in sorted(fixes, key=lambda st: ticks(kitchen, me, st)):
                    done = finish(steps, entry)
                    if done:
                        return done

        now = environment.timestep
        reserve = []
        for entry in choices:
            dish = _short(entry.name)
            started = time.monotonic()
            candidates = []
            problems = {}
            for fast in MODEL_ORDER:
                text = _problem_text(kitchen, entry, fast, build_problem)
                problems[fast] = text
                if not USE_LIBRARY:
                    continue
                init, goal = library.facts(text)
                for steps in library.plans_for(dish, c):
                    moved = relocated(steps)
                    if library.valid(moved, init, goal):
                        candidates.append(moved)
                    elif library.valid(steps, init, goal):
                        candidates.append(steps)
            if not candidates or _made_food_helps(
                environment, kitchen, entry, chef_id, mates[2] if mates else frozenset()
            ):
                for fast in MODEL_ORDER:
                    steps = []
                    if PRUNE:
                        self._calls += PRUNED_TICKS / max(1, CALL_TICKS)
                        try:
                            steps = solve_problem_with_logging(
                                PDDLProblem(
                                    domain=build_domain(),
                                    problem=_hide(_pruned(problems[fast], environment, entry.name), mates),
                                ),
                                time_limit_seconds=PLAN_SECONDS,
                            )
                        except Exception:  # noqa: BLE001 - try the whole problem
                            steps = []
                    if not steps:
                        self._calls += 1
                        try:
                            steps = solve_problem_with_logging(
                                PDDLProblem(domain=build_domain(), problem=_hide(problems[fast], mates)),
                                time_limit_seconds=PLAN_SECONDS,
                            )
                        except Exception:  # noqa: BLE001
                            steps = []
                    if steps:
                        # Kept even if it uses food already out: the check
                        # before any reuse rules out plans that no longer fit.
                        library.remember(dish, steps, c)
                        candidates.append(relocated(steps))
                        break
                    if time.monotonic() - started > PLAN_SECONDS:
                        break
            if not candidates:
                continue
            timed = sorted(
                (ticks(kitchen, me, steps) + EXPOSED * exposure(kitchen, me, steps), i)
                for i, steps in enumerate(candidates)
            )
            quickest = timed[0][0]
            left = GAME_TICKS - now
            if entry.time_limit is not None:
                left = min(left, entry.revealed_at + entry.time_limit - now)
            if LOG:
                if left >= quickest * SLACK + 3:
                    chk = "ok"
                elif quickest + 2 <= left or not STRICT:
                    chk = "reserve"
                else:
                    chk = "too_late"
                print("LOG", now, chef_id, "plan_check", dish,
                      f"planned={quickest:.0f}", f"deadline={left:.0f}", chk, flush=True)
            if left < quickest * SLACK + 3:
                # Would not make it in time (the order expires, or the game
                # ends): try the next order, keeping this one in reserve.
                if quickest + 2 <= left or not STRICT:
                    reserve.append((quickest, entry, candidates, timed))
                continue
            for _, i in timed:
                done = finish(candidates[i], entry)
                if done:
                    return done
        for _, entry, candidates, timed in sorted(reserve, key=lambda r: r[0]):
            for _, i in timed:
                done = finish(candidates[i], entry)
                if done:
                    return done
        if LOG:
            state = environment.get_first_entity_of_type(OvercookedState)
            now = environment.timestep
            print("LOG", now, chef_id, "none", [_short(e.name) for e in choices], "reserve",
                  [(_short(e.name), round(q)) for q, e, *_ in reserve], "visible",
                  [(_short(e.name), e.revealed_at + (e.time_limit or 0) - now) for e in state.order_queue.visible],
                  flush=True)
        return [], None, [], []

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
        def mate(a) -> bool:
            return a.id != chef_id and bool(me_agent.team) and a.team == me_agent.team

        agents = [
            a
            for a in everyone
            if a.id == chef_id
            or (mate(a) and MATE_AUCTION)
            or (not mate(a) and (now < 40 or now - self._last_held.get(a.id, -999) < 60))
        ]
        visible = list(state.order_queue.visible)
        # The teammate is someone else's controller: an order is left to it
        # only when it visibly makes that dish (holds food only that dish
        # needs), not on a guess that it will.
        left_to_mates: list[str] = []
        if not MATE_AUCTION:
            put_down = self._teammates_things(environment, me_agent)[3] if RESPECT_MATES else {}
            for a in everyone:
                if mate(a):
                    dishes = _working_on(environment, state, a, visible, put_down.get(a.id, set()))
                    left_to_mates += sorted(dishes)

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

        skipped = list(left_to_mates)

        mine = _held_foods(environment, me_agent)

        def leave(entry) -> bool:
            if HOLD_ON and mine & _components(state, entry.name):
                return False  # we carry part of it too: finish it
            if entry.name in skipped:
                skipped.remove(entry.name)  # one order per dish a teammate makes
                return True
            return False

        made = (
            {f.name for f in environment.get_entities_of_type(Food) if not state.is_raw_food(f.name)}
            if PROGRESS else set()
        )

        def rough_time(entry) -> float:
            if made & _components(state, entry.name):
                return 8.0  # partly made already: let the planner time it
            return EXPECTED_STEPS.get(_short(entry.name), DEFAULT_STEPS) * ROUGH_FRAC

        open_orders = [
            (i, entry)
            for i, entry in enumerate(visible)
            if not leave(entry)
            and time_left(entry) >= rough_time(entry)
            and now - self._failed.get(entry.name, -999) >= 20
        ]
        if LOG:
            open_idx = {i for i, _ in open_orders}
            for i, entry in enumerate(visible):
                tl = time_left(entry)
                rt = rough_time(entry)
                e = eta(me_agent, entry)
                if i in open_idx:
                    reason = "ok"
                elif now - self._failed.get(entry.name, -999) < 20:
                    reason = "cooldown"
                elif entry.name in left_to_mates:
                    reason = "mate"
                else:
                    reason = "too_late"
                print("LOG", now, chef_id, "pick", _short(entry.name),
                      f"eta={e:.0f}", f"deadline={tl:.0f}", reason, flush=True)
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
        by_id = {a.id: a for a in agents}

        def ahead(i, entry) -> bool:
            """Whether the chef handed this order is visibly making it: it
            holds the dish, or made food that no other visible order needs.
            (A raw bun could be for any burger, so it says little.)"""
            taker = by_id.get(taken_by.get(i))
            if not RACE_CHECK or taker is None:
                return False
            held = _held_foods(environment, taker)
            if entry.name in held:
                return True
            others = set()
            for other in visible:
                if other.name != entry.name:
                    others |= _components(state, other.name)
            return any(
                food in _components(state, entry.name)
                and food not in others
                and not state.is_raw_food(food)
                for food in held
            )

        rest = sorted(
            (
                entry
                for i, entry in open_orders
                if taken_by.get(i) != chef_id
                and taken_by.get(i) not in mates
                and not ahead(i, entry)
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
        return ordered[:3]


# --- Helpers ---------------------------------------------------------------------


def _goal_tiles(target):
    x, y = target
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def _move(chef: Agent, position, tile) -> MoveAgent:
    return MoveAgent(agent_id=chef.id, dx=tile[0] - position[0], dy=tile[1] - position[1])


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


def _cooking(environment: Environment, target) -> bool:
    """Whether food is cooking by itself at ``target`` and not done yet."""
    from simulator.entities import Equipment
    from simulator.mutations.cooking import cooking_progress

    for item in environment.get_entities_of_type(Equipment):
        if (item.x, item.y) != tuple(target) or isinstance(item, Plate) or not item.cooks_by_itself:
            continue
        progress = cooking_progress(environment, item)
        if progress is not None and progress < 1:
            return True
    return False


def _patience(environment: Environment, target) -> int:
    """Ticks to wait beside ``target`` for an action to become possible:
    until food cooking there by itself is done, otherwise two."""
    from simulator.entities import Equipment
    from simulator.mutations.cooking import cooking_progress

    wait = 2
    for item in environment.get_entities_of_type(Equipment):
        if (item.x, item.y) != tuple(target) or isinstance(item, Plate) or not item.cooks_by_itself:
            continue
        progress = cooking_progress(environment, item)
        if progress is not None and progress < 1:
            wait = max(wait, round(item.cook_time * (1 - progress)) + 2)
    return wait


def _problem_text(kitchen, entry, fast: bool, build_problem) -> str:
    """The PDDL problem for ``entry``. The two models differ only in the
    ``(fast_plating)`` fact, so the kitchen is described once per order."""
    cache = kitchen.__dict__.setdefault("_problem_texts", {})
    key = id(entry)
    if key not in cache:
        cache[key] = build_problem(kitchen, [entry], fast=True)
    text = cache[key]
    return text if fast else text.replace("    (fast_plating)\n", "", 1)


def _pruned(problem: str, environment: Environment, dish: str) -> str:
    """``problem`` without the foods (and plates of them) that neither go
    into ``dish`` nor lie in the kitchen already, nor any fact about them.

    The full problem lists all 45 foods of the menu and every way to plate
    them; Fast Downward spends most of its time grounding those. Taking
    facts away only takes plans away, so any plan for the smaller problem
    works for the whole one, and it is found three to four times faster.
    """
    import re

    from .planner.kitchen import food_symbol, plate_symbol

    state = environment.get_first_entity_of_type(OvercookedState)
    foods = _components(state, dish) | {f.name for f in environment.get_entities_of_type(Food)}
    keep = {food_symbol(f) for f in foods} | {plate_symbol(f) for f in foods}
    keep |= {"p_empty", "p_dirty"}
    lines = []
    for line in problem.split("\n"):
        if any(
            (token.startswith("f_") or token.startswith("p_")) and token not in keep
            for token in re.findall(r"[^\s()]+", line)
        ):
            continue
        lines.append(line)
    return "\n".join(lines)


def _hide(problem: str, mates) -> str:
    """``problem`` without the stations and counters teammates have put
    food in: their food is theirs to use, and the planner would happily
    take it (the opponents' food stays in: taking that is fair game)."""
    if not mates:
        return problem
    stations, counters = mates[0], mates[1]
    if not stations and not counters:
        return problem
    import re

    lines = []
    for line in problem.split("\n"):
        tokens = re.findall(r"[^\s()]+", line)
        if any(token in stations for token in tokens):
            continue
        if tokens and tokens[0] == "on" and tokens[-1] in counters:
            continue
        lines.append(line)
    return "\n".join(lines)


def _made_food_helps(environment, kitchen, entry, chef_id, ignore=frozenset()) -> bool:
    """Whether food already made (not raw) that goes into ``entry`` lies in
    the kitchen, or this chef holds anything: then a fresh plan may use it,
    which plans from the library never do."""
    state = environment.get_first_entity_of_type(OvercookedState)
    wanted = _components(state, entry.name)
    holders = {
        a.held_item_id: a.id for a in environment.get_entities_of_type(Agent) if a.held_item_id
    }
    me = next(a for a in environment.get_entities_of_type(Agent) if a.id == chef_id)
    if me.held_item_id is not None:
        return True
    if not USE_MADE:
        return False
    for food in environment.get_entities_of_type(Food):
        if food.id in ignore:
            continue
        if food.name in wanted and not kitchen.is_raw(food.name):
            holder = holders.get(food.id)
            if holder is None or holder == chef_id:
                return True
    return False


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


def _working_on(environment, state, agent: Agent, visible, put_down=frozenset()) -> set[str]:
    """Goal recognition: the visible dishes the food this chef holds (or
    has put in a station) leads to, if it leads to exactly one of them."""
    held = _held_foods(environment, agent) | set(put_down)
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
