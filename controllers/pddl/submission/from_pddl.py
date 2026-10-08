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

from .dispatch import dispatch
from .kitchen import Kitchen, parse_loc


class DefaultFromPDDL(FromPDDL):
    def __init__(self, environment: Environment) -> None:
        self._environment = environment

    def from_pddl(self, steps: list[PDDLPlanStep]) -> list[MutationWithLocation]:
        kitchen = Kitchen(self._environment)
        steps = dispatch(kitchen, steps)
        translator = _Translator(kitchen)
        plan: list[MutationWithLocation] = []
        for step in steps:
            plan.extend(translator.translate(step))
        if plan and _orders_left_out(kitchen, steps):
            plan.append(_wait_to_replan(plan[-1]))
        return plan


def _orders_left_out(kitchen: Kitchen, steps: list[PDDLPlanStep]) -> bool:
    """Whether the plan serves fewer orders than the kitchen could.

    That happens when the planner ran out of time on the full set of orders
    and the problem generator fell back to fewer.
    """
    from .problem_generator import feasible_orders, queued_orders

    served = sum(1 for step in steps if step.name.startswith("deliver"))
    return served < len(feasible_orders(kitchen, queued_orders(kitchen)))


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
