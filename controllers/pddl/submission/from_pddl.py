"""Translate solver plans into simulator mutations.

``DefaultPDDLController`` creates a ``DefaultFromPDDL`` with the environment
the plan was made for and calls ``from_pddl`` in a worker process. Each plan
step names its chef, so the controller can run every chef's steps in
parallel; the ``expected_*`` fields on each mutation make a chef wait until
the kitchen is in the state the plan assumed (for example, until another
chef has put the plate down, or until the oven has finished).

Each step becomes the mutation the simulator would apply for the matching
key press, together with the tile the chef must face. Steps for stations
that cook by themselves produce no mutation: the next step that takes the
food out waits for it to be ready.
"""

from __future__ import annotations

import hashlib
import pickle
import tempfile
from pathlib import Path

from controllers.pddl.interfaces import FromPDDL, PDDLPlanStep
from simulator.entities.plate import PLATE_NAME
from simulator.environment import Environment
from simulator.mutations import (
    Combine,
    Cook,
    Deliver,
    Discard,
    MutationWithLocation,
    PickUp,
    Place,
    TakeFromStorage,
    Wash,
)
from simulator.mutations.agent_mutation import AgentMutation
from simulator.mutations.cooking import current_timer
from simulator.entities import Food

from .config import CONFIG
from .dispatch import Options, dispatch
from .kitchen import Kitchen, chef_symbol, parse_loc
from .rollout import simulate


class DefaultFromPDDL(FromPDDL):
    def __init__(self, environment: Environment) -> None:
        self._environment = environment

    def from_pddl(self, steps: list[PDDLPlanStep]) -> list[MutationWithLocation]:
        return self.best(steps)[1]

    def best(
        self, steps: list[PDDLPlanStep], quick: bool = False
    ) -> tuple[tuple, list[MutationWithLocation]]:
        """The best way found to carry out ``steps``, and its rollout score.

        ``quick`` tries only the default dispatch.
        """
        kitchen = Kitchen(self._environment)
        left_out = _orders_left_out(kitchen, steps)
        cache_key = None if quick else _cache_key(self._environment, steps)
        cached = _cache_get(cache_key)
        if cached is not None:
            key, chosen = cached
            return key, self.translate_all(kitchen, chosen, left_out)
        best: tuple | None = None
        options_list = [Options()] if quick else candidate_options(kitchen, steps)
        best_steps: list[PDDLPlanStep] = steps
        seen: list[list] = []
        for options in options_list:
            if options is None:
                chosen = steps  # the planner's own assignment
            else:
                chosen = dispatch(kitchen, steps, options)
            if chosen in seen:
                continue
            seen.append(chosen)
            plan = self.translate_all(kitchen, chosen, left_out)
            outcome = simulate(self._environment, plan)
            if best is None or outcome.key() > best[0]:
                best = (outcome.key(), plan)
                best_steps = chosen
        if best is None:
            return ((), [])
        _cache_put(cache_key, (best[0], best_steps))
        return best

    def translate_all(
        self, kitchen: Kitchen, steps: list[PDDLPlanStep], left_out: bool
    ) -> list[MutationWithLocation]:
        translator = _Translator(kitchen)
        plan: list[MutationWithLocation] = []
        for step in steps:
            plan.extend(translator.translate(step))
        if plan and left_out:
            plan.append(_wait_to_replan(plan[-1]))
        return plan


# The problem generator already tries plans out to choose between models;
# the plan it picks is tried again here in another worker process. A small
# disk cache, keyed by the kitchen and the plan, saves doing it twice.
_CACHE_DIR = Path(tempfile.gettempdir()) / "too-many-chefs-rollouts"


def _cache_key(environment: Environment, steps: list[PDDLPlanStep]) -> str | None:
    try:
        text = canonical(environment) + repr(steps) + repr(CONFIG)
        return hashlib.sha256(text.encode()).hexdigest()
    except Exception:  # noqa: BLE001
        return None


def canonical(environment: Environment) -> str:
    """The kitchen's state without entity ids, which are random for food
    created during a run (so a played-out run and the real one match)."""
    from simulator.entities import Entity, OvercookedState

    def describe(entity_id: str | None, depth: int = 0) -> str:
        if entity_id is None or depth > 3:
            return "-"
        entity = environment.get_entity(entity_id)
        if entity is None:
            return "?"
        held = getattr(entity, "held_item_id", None)
        return f"{getattr(entity, 'name', '')}/{getattr(entity, 'dirty', '')}[{describe(held, depth + 1)}]"

    rows = []
    for entity in environment.get_entities_of_type(Entity):
        if isinstance(entity, OvercookedState):
            queue = entity.order_queue
            rows.append(
                f"Q{[(e.name, e.revealed_at, e.time_limit, e.reward) for e in queue.visible]}"
                f"{[(e.name, e.time_limit) for e in queue.pending]}"
                f"{queue.delivered_count},{queue.expired_count},{entity.score}"
            )
            continue
        x, y = getattr(entity, "x", None), getattr(entity, "y", None)
        if x is None:
            continue
        cooking = getattr(entity, "cooking", None)
        rows.append(
            f"{type(entity).__name__}@{x},{y}:{getattr(entity, 'name', '')}"
            f":{getattr(entity, 'orientation', '')}:{describe(getattr(entity, 'held_item_id', None))}"
            f":{cooking.elapsed if cooking else ''}"
            f":{getattr(entity, 'plate_count', '')},{getattr(entity, 'dirty_plate_count', '')}"
            f":{getattr(entity, 'id', '') if type(entity).__name__ == 'Agent' else ''}"
        )
    return f"t{environment.timestep}|" + "|".join(sorted(rows))


def _cache_get(key: str | None):
    if key is None:
        return None
    try:
        with open(_CACHE_DIR / key, "rb") as handle:
            return pickle.load(handle)
    except Exception:  # noqa: BLE001
        return None


def _cache_put(key: str | None, value) -> None:
    if key is None:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_DIR / key, "wb") as handle:
            pickle.dump(value, handle)
    except Exception:  # noqa: BLE001
        pass


def candidate_options(kitchen: Kitchen, steps: list[PDDLPlanStep]):
    """Ways of handing out the plan's jobs, to be compared by rollout.

    ``None`` stands for the planner's own choice of chefs.
    """
    if CONFIG.candidates == "small":
        yield Options()
        yield Options(move_stations="off", split_cooks=False, split_plates=False)
        yield None
        return
    for move in ("any", "off"):
        for split in (True, False):
            yield Options(move_stations=move, split_cooks=split, split_plates=split)
    if any(step.name == "take_food_onto_dish" for step in steps):
        # Win back the tips the quick model gives up, if time allows.
        for split in (True, False):
            yield Options(split_cooks=split, split_plates=split, keep_tips=True)
    yield Options(move_stations="any", split_cooks=True, split_plates=False)
    # Hand out jobs by list scheduling: whichever job can finish soonest
    # goes next, so an urgent dish is not left waiting behind a later one.
    if CONFIG.candidates != "noreorder":
        for split in (True, False):
            yield Options(reorder=True, split_cooks=split, split_plates=split)
    yield Options(park_dishes=True)
    yield Options(move_stations="off", park_dishes=True)
    yield None
    chefs = [chef_symbol(a.id) for a in kitchen.chefs]
    if len(chefs) >= 3:
        # A crowded kitchen may go faster, and jam less, with a chef fewer.
        for chef in chefs:
            yield Options(exclude=frozenset({chef}))


def _orders_left_out(kitchen: Kitchen, steps: list[PDDLPlanStep]) -> bool:
    """Whether the plan serves fewer orders than are waiting.

    That happens when the planner ran out of time on the full set of orders
    and the problem generator fell back to fewer, or when the stock check
    judged an order impossible. Either way, once the plan is done the
    controller replans and tries the rest one by one rather than stopping.
    """
    from .problem_generator import queued_orders

    served = sum(1 for step in steps if step.name.startswith("deliver"))
    return served < len(queued_orders(kitchen))


def _wait_to_replan(last: MutationWithLocation) -> MutationWithLocation:
    """A final step that never becomes legal, so the chef waits where it is.

    Once everything else is done, nobody can act and time stops; the problem
    generator notices and replans for the orders this plan left out. Without
    it the controller would finish with those orders unserved.
    """
    return MutationWithLocation(
        mutation=PickUp(
            agent_id=last.mutation.agent_id,  # type: ignore[attr-defined]
            expected_input_name="(replan for the remaining orders)",
        ),
        location=last.location,
    )


class _Translator:
    def __init__(self, kitchen: Kitchen) -> None:
        self.k = kitchen
        # Stations already used by an earlier step; a cook step on one of
        # these starts from a fresh timer.
        self._touched_stations: set[str] = set()

    # --- Helpers -------------------------------------------------------------

    def food(self, symbol: str) -> str:
        return self.k.food_name(symbol)

    def plate(self, symbol: str) -> str | None:
        return self.k.plate_food(symbol)

    def station_name(self, symbol: str) -> str:
        return self.k.station_by_symbol[symbol].name

    def at(self, mutation: AgentMutation, loc: str) -> list[MutationWithLocation]:
        return [MutationWithLocation(mutation=mutation, location=parse_loc(loc))]

    # --- Steps ---------------------------------------------------------------

    def translate(self, step: PDDLPlanStep) -> list[MutationWithLocation]:
        name, p = step.name, step.parameters
        if name == "go":
            # Walking (modelled only in the Part 4 variant) is left to the
            # controller's path finding.
            return []
        if name == "cook_by_itself":
            self._touched_stations.add(p[0])
            return []
        chef = self.k.chef_by_symbol(p[0]).id
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            raise ValueError(f"No translation for this action yet: {step}")
        return handler(chef, *p[1:])

    def _take_food(self, chef: str, loc: str, food: str):
        name = self.food(food)
        return self.at(
            TakeFromStorage(
                agent_id=chef, expected_input_name=name, expected_output_name=name
            ),
            loc,
        )

    def _take_food_onto_plate(self, chef: str, loc: str, food: str, p: str, q: str):
        return self.at(
            TakeFromStorage(
                agent_id=chef,
                expected_input_name=self.food(food),
                expected_output_name=self.plate(q),
            ),
            loc,
        )

    _take_food_onto_dish = _take_food_onto_plate

    def _take_plate_infinite(self, chef: str, loc: str, p: str):
        return self.at(TakeFromStorage(agent_id=chef), loc)

    def _take_plate_clean(self, chef: str, loc: str, n: str, m: str, p: str):
        return self.at(TakeFromStorage(agent_id=chef), loc)

    def _take_plate_dirty(self, chef: str, loc: str, z: str, n: str, m: str, p: str):
        return self.at(TakeFromStorage(agent_id=chef), loc)

    def _pick_up(self, chef: str, loc: str, thing: str):
        if thing.startswith("p_"):
            mutation = PickUp(
                agent_id=chef,
                expected_held_equipment_name=PLATE_NAME,
                expected_held_equipment_contents_name=self.plate(thing),
            )
        else:
            mutation = PickUp(agent_id=chef, expected_input_name=self.food(thing))
        return self.at(mutation, loc)

    def _put_down(self, chef: str, loc: str, thing: str):
        if thing.startswith("p_"):
            mutation = Place(
                agent_id=chef,
                expected_held_equipment_name=PLATE_NAME,
                expected_held_equipment_contents_name=self.plate(thing),
            )
        else:
            mutation = Place(agent_id=chef, expected_held_food=self.food(thing))
        return self.at(mutation, loc)

    def _combine_on_counter(self, chef: str, loc: str, f: str, g: str, o: str):
        return self.at(
            Combine(
                agent_id=chef,
                expected_held_food=self.food(f),
                expected_target_food_name=self.food(g),
                expected_output_name=self.food(o),
            ),
            loc,
        )

    def _food_onto_counter_plate(self, chef: str, loc: str, f: str, p: str, q: str):
        return self.at(
            Combine(
                agent_id=chef,
                expected_held_food=self.food(f),
                expected_target_equipment_name=PLATE_NAME,
                expected_target_equipment_contents_name=self.plate(p),
                expected_output_name=self.plate(q),
            ),
            loc,
        )

    def _counter_food_onto_plate(self, chef: str, loc: str, f: str, p: str, q: str):
        return self.at(
            Combine(
                agent_id=chef,
                expected_held_equipment_name=PLATE_NAME,
                expected_held_equipment_contents_name=self.plate(p),
                expected_target_food_name=self.food(f),
                expected_output_name=self.plate(q),
            ),
            loc,
        )

    def _counter_plate_onto_plate(
        self, chef: str, loc: str, r: str, f: str, p: str, q: str, e: str
    ):
        return self.at(
            Combine(
                agent_id=chef,
                expected_held_equipment_name=PLATE_NAME,
                expected_held_equipment_contents_name=self.plate(p),
                expected_target_equipment_name=PLATE_NAME,
                expected_target_equipment_contents_name=self.food(f),
                expected_output_name=self.plate(q),
            ),
            loc,
        )

    def _put_in_station(self, chef: str, s: str, loc: str, f: str):
        self._touched_stations.add(s)
        return self.at(
            Combine(
                agent_id=chef,
                expected_held_food=self.food(f),
                expected_target_equipment_name=self.station_name(s),
                expected_output_name=self.food(f),
            ),
            loc,
        )

    def _add_to_station(self, chef: str, s: str, loc: str, f: str, e: str, o: str):
        self._touched_stations.add(s)
        return self.at(
            Combine(
                agent_id=chef,
                expected_held_food=self.food(f),
                expected_target_equipment_name=self.station_name(s),
                expected_target_equipment_contents_name=self.food(e),
                expected_output_name=self.food(o),
            ),
            loc,
        )

    def _combine_from_station(
        self, chef: str, s: str, loc: str, f: str, e: str, o: str
    ):
        self._touched_stations.add(s)
        return self.at(
            Combine(
                agent_id=chef,
                expected_held_food=self.food(f),
                expected_target_equipment_name=self.station_name(s),
                expected_target_equipment_contents_name=self.food(e),
                expected_output_name=self.food(o),
            ),
            loc,
        )

    def _take_out(self, chef: str, s: str, loc: str, f: str):
        self._touched_stations.add(s)
        name = self.food(f)
        return self.at(
            PickUp(agent_id=chef, expected_input_name=name, expected_output_name=name),
            loc,
        )

    def _cook(self, chef: str, s: str, loc: str, f: str, o: str):
        station = self.k.station_by_symbol[s]
        presses = station.entity.cook_time
        if s not in self._touched_stations:
            # The food may already be part-way through.
            held = (
                self.k.environment.get_entity_as(station.entity.held_item_id, Food)
                if station.entity.held_item_id
                else None
            )
            if held is not None:
                timer = current_timer(station.entity, held)
                if timer is not None:
                    presses = max(1, presses - timer.elapsed)
        self._touched_stations.add(s)
        mutation = Cook(
            agent_id=chef,
            expected_held_food=self.food(f),
            expected_input_name=station.name,
            expected_output_name=self.food(o),
        )
        return self.at(mutation, loc) * presses

    def _station_onto_plate(
        self, chef: str, s: str, loc: str, f: str, p: str, q: str
    ):
        self._touched_stations.add(s)
        return self.at(
            Combine(
                agent_id=chef,
                expected_held_equipment_name=PLATE_NAME,
                expected_held_equipment_contents_name=self.plate(p),
                expected_target_equipment_name=self.station_name(s),
                expected_target_equipment_contents_name=self.food(f),
                expected_output_name=self.plate(q),
            ),
            loc,
        )

    def _combine_from_station_tended(self, *args: str):
        return self._combine_from_station(*args)

    def _station_onto_plate_tended(self, *args: str):
        return self._station_onto_plate(*args)

    def _plate_into_station(
        self, chef: str, s: str, loc: str, p: str, f: str, e: str
    ):
        self._touched_stations.add(s)
        return self.at(
            Combine(
                agent_id=chef,
                expected_held_equipment_name=PLATE_NAME,
                expected_held_equipment_contents_name=self.plate(p),
                expected_target_equipment_name=self.station_name(s),
                expected_output_name=self.food(f),
            ),
            loc,
        )

    def _plate_add_to_station(
        self,
        chef: str,
        s: str,
        loc: str,
        p: str,
        f: str,
        g: str,
        o: str,
        e: str,
    ):
        self._touched_stations.add(s)
        return self.at(
            Combine(
                agent_id=chef,
                expected_held_equipment_name=PLATE_NAME,
                expected_held_equipment_contents_name=self.plate(p),
                expected_target_equipment_name=self.station_name(s),
                expected_target_equipment_contents_name=self.food(g),
                expected_output_name=self.food(o),
            ),
            loc,
        )

    def _lift_station_from_counter(self, chef: str, s: str, loc: str):
        return self.at(
            PickUp(agent_id=chef, expected_held_equipment_name=self.station_name(s)),
            loc,
        )

    _lift_station_from_support = _lift_station_from_counter

    def _set_station_on_support(self, chef: str, s: str, loc: str):
        return self.at(
            Place(agent_id=chef, expected_held_equipment_name=self.station_name(s)),
            loc,
        )

    _set_station_on_counter = _set_station_on_support

    def _discard_food(self, chef: str, loc: str, f: str):
        return self.at(Discard(agent_id=chef), loc)

    def _discard_from_plate(self, chef: str, loc: str, p: str, f: str, e: str):
        return self.at(Discard(agent_id=chef), loc)

    def _wash(self, chef: str, loc: str, p: str, e: str):
        return self.at(Wash(agent_id=chef), loc)

    def _deliver(self, chef: str, loc: str, p: str, *_rest: str):
        return self.at(
            Deliver(
                agent_id=chef,
                expected_held_equipment_name=PLATE_NAME,
                expected_held_food=self.plate(p),
            ),
            loc,
        )

    _deliver_return_clean = _deliver
    _deliver_return_dirty = _deliver
