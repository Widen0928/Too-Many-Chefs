"""A planner's-eye view of a kitchen, shared by the problem generator and the
plan translator.

``Kitchen`` reads an ``Environment`` once and records everything the PDDL
model needs: which tiles each chef can reach, the stations and what they
hold, the food on counters, plate supplies and the recipe rules. Both
``problem_generator.py`` (to write the problem) and ``from_pddl.py`` (to
resolve plan symbols back to kitchen objects) build one from the same
environment, so the symbol names always agree.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass, field
from functools import cached_property

from simulator.entities import (
    Agent,
    Bin,
    Bounds,
    Counter,
    Delivery,
    Equipment,
    Food,
    OvercookedState,
    Plate,
    PlateDispenser,
    Sink,
    Storage,
)
from simulator.environment import Environment
from simulator.mutations.move_agent import is_walkable
from simulator.types import Location

EMPTY_PLATE = "p_empty"
DIRTY_PLATE = "p_dirty"


# --- Symbols -----------------------------------------------------------------


def pddl_name(value: str) -> str:
    """Lower-case ``value`` and replace each run of other characters with ``_``."""
    return re.sub(r"[^0-9a-z_]+", "_", value.lower()).strip("_")


def short_name(catalog_name: str) -> str:
    """``catalog/food/coconut_juice`` -> ``coconut_juice``."""
    return pddl_name(catalog_name.rsplit("/", 1)[-1])


def food_symbol(food_name: str) -> str:
    return "f_" + short_name(food_name)


def plate_symbol(food_name: str | None, *, dirty: bool = False) -> str:
    if dirty:
        return DIRTY_PLATE
    if food_name is None:
        return EMPTY_PLATE
    return "p_" + short_name(food_name)


def loc_symbol(location: Location) -> str:
    return f"loc_{location[0]}_{location[1]}"


def parse_loc(symbol: str) -> Location:
    match = re.fullmatch(r"loc_(\d+)_(\d+)", symbol)
    if match is None:
        raise ValueError(f"Unknown location in plan: {symbol}")
    return int(match.group(1)), int(match.group(2))


def chef_symbol(agent_id: str) -> str:
    return "c_" + pddl_name(agent_id)


# --- Kitchen -----------------------------------------------------------------


@dataclass
class Station:
    """A piece of equipment that holds and transforms food."""

    symbol: str
    entity: Equipment
    location: Location | None
    supported: bool = True
    # The chef holding this station, if a chef is carrying it.
    holder: str | None = None

    @property
    def name(self) -> str:
        return self.entity.name


@dataclass
class Kitchen:
    environment: Environment
    controlled_agent_ids: Collection[str] | None = None
    # Floor tiles to treat as blocked, such as chefs standing in the way
    # (Part 4). Empty for Parts 1-3.
    blocked: frozenset = frozenset()
    # Filled in by __post_init__.
    chefs: list[Agent] = field(default_factory=list)
    reach: dict[str, set[Location]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        agents = sorted(
            self.environment.get_entities_of_type(Agent), key=lambda a: a.id
        )
        if self.controlled_agent_ids is not None:
            agents = [a for a in agents if a.id in self.controlled_agent_ids]
        self.chefs = [a for a in agents if a.x is not None and a.y is not None]
        for chef in self.chefs:
            region = self.region(chef.x, chef.y)  # type: ignore[arg-type]
            self.reach[chef.id] = self.reach_from(region)

    # --- Level-wide information ------------------------------------------------

    @cached_property
    def state(self) -> OvercookedState:
        state = self.environment.get_first_entity_of_type(OvercookedState)
        if state is None:
            raise ValueError("The environment has no OvercookedState")
        return state

    @cached_property
    def food_names(self) -> list[str]:
        """Every food the level defines, plus any odd food lying about."""
        names = dict.fromkeys(self.state.food_definitions)
        for food in self.environment.get_entities_of_type(Food):
            names.setdefault(food.name)
        return list(names)

    @cached_property
    def food_by_symbol(self) -> dict[str, str]:
        return {food_symbol(name): name for name in self.food_names}

    def is_raw(self, food_name: str) -> bool:
        return self.state.is_raw_food(food_name)

    def combine(self, first: str, second: str) -> str | None:
        """The legal combine recipe output, ignoring illegal-recipe garbage."""
        return self._combine_table.get(frozenset((first, second)))

    @cached_property
    def _combine_table(self) -> dict[frozenset, str]:
        # Looked up for every pair of foods when describing the kitchen, so
        # the recipes are indexed once (the first recipe wins, as in
        # OvercookedState.get_combine_output).
        table: dict[frozenset, str] = {}
        for recipe in self.state.combine_recipes:
            table.setdefault(frozenset(recipe.ingredients), recipe.output)
        return table

    def legal_accepts(self, food_name: str, equipment_name: str) -> bool:
        """Whether a station takes food that legally leads to its recipes.

        This is ``OvercookedState.equipment_accepts`` without the clause that
        lets any cooking station take anything when illegal recipes are on.
        """
        return self._legal_accepts(food_name, equipment_name, frozenset())

    def _legal_accepts(
        self, food_name: str, equipment_name: str, seen: frozenset[str]
    ) -> bool:
        if self.state.can_cook_with(food_name, equipment_name):
            return True
        if food_name in seen:
            return False
        seen = seen | {food_name}
        return any(
            self._legal_accepts(recipe.output, equipment_name, seen)
            for recipe in self.state.combine_recipes
            if food_name in recipe.ingredients
        )

    def sim_accepts(self, food_name: str, equipment_name: str) -> bool:
        """Whether the simulator would let the station take the food."""
        return self.state.equipment_accepts(food_name, equipment_name)

    # --- Floor plan ------------------------------------------------------------

    @cached_property
    def bounds(self) -> Bounds:
        bounds = self.environment.get_first_entity_of_type(Bounds)
        if bounds is None:
            raise ValueError("Environment is missing Bounds")
        return bounds

    @cached_property
    def walkable(self) -> set[Location]:
        b = self.bounds
        x0, y0 = b.x or 0, b.y or 0
        return {
            (x, y)
            for x in range(x0, x0 + b.width)
            for y in range(y0, y0 + b.height)
            if is_walkable(self.environment, x, y, ignore_agents=True)
            and (x, y) not in self.blocked
        }

    def reach_from(self, region: set[Location]) -> set[Location]:
        """Tiles a chef walking ``region`` can act on: those beside it that
        are not floor, plus floor tiles with food or a plate lying on them."""
        return {
            neighbour
            for tile in region
            for neighbour in _neighbours(tile)
            if neighbour not in self.walkable or neighbour in self.loose_items
        }

    def region(self, x: int, y: int) -> set[Location]:
        """Floor tiles connected to (x, y), ignoring chefs."""
        start = (x, y)
        seen = {start}
        queue = deque([start])
        while queue:
            tile = queue.popleft()
            for neighbour in _neighbours(tile):
                if neighbour in self.walkable and neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        return seen

    def walk_from(self, *starts: Location) -> dict[Location, int]:
        """Steps from the nearest of ``starts`` to every floor tile reachable."""
        dist: dict[Location, int] = {}
        queue: deque[Location] = deque()
        for tile in starts:
            dist[tile] = 0
            queue.append(tile)
        while queue:
            tile = queue.popleft()
            for neighbour in _neighbours(tile):
                if neighbour in self.walkable and neighbour not in dist:
                    dist[neighbour] = dist[tile] + 1
                    queue.append(neighbour)
        return dist

    @cached_property
    def distances(self) -> dict[Location, dict[Location, int]]:
        """Steps from the floor tiles beside each target to every floor tile."""
        return {
            target: self.walk_from(
                *(t for t in _neighbours(target) if t in self.walkable)
            )
            for target in self.targets
        }

    def environment_agents(self) -> list[Agent]:
        """Every chef in the kitchen, controlled or not, in id order."""
        return sorted(self.environment.get_entities_of_type(Agent), key=lambda a: a.id)

    def chef_location(self, chef: Agent) -> Location:
        return (chef.x, chef.y)  # type: ignore[return-value]

    # --- Things in the kitchen -------------------------------------------------

    def _at(self, cls: type) -> dict[Location, list]:
        found: dict[Location, list] = {}
        for entity in self.environment.get_entities_of_type(cls):
            if entity.x is None or entity.y is None:
                continue
            found.setdefault((entity.x, entity.y), []).append(entity)
        return found

    @cached_property
    def counters(self) -> dict[Location, Counter]:
        return {loc: items[0] for loc, items in self._at(Counter).items()}

    @cached_property
    def storages(self) -> dict[Location, Storage]:
        return {loc: items[0] for loc, items in self._at(Storage).items()}

    @cached_property
    def dispensers(self) -> dict[Location, PlateDispenser]:
        return {loc: items[0] for loc, items in self._at(PlateDispenser).items()}

    @cached_property
    def deliveries(self) -> set[Location]:
        return set(self._at(Delivery))

    @cached_property
    def bins(self) -> set[Location]:
        return set(self._at(Bin))

    @cached_property
    def sinks(self) -> set[Location]:
        return set(self._at(Sink))

    @cached_property
    def equipment_by_location(self) -> dict[Location, list[Equipment]]:
        found = self._at(Equipment)
        return {
            loc: [e for e in items if not isinstance(e, Plate)]
            for loc, items in found.items()
        }

    @cached_property
    def stations(self) -> list[Station]:
        """Stations that can hold food and do something with it.

        A stove or mixer on its own takes no food, so those are left out. A
        pan or bowl is listed wherever it is; ``supported`` says whether it
        stands on the station it needs and so can be used.
        """
        stations: list[Station] = []
        for loc, equipments in sorted(self.equipment_by_location.items()):
            for equipment in equipments:
                if not equipment.can_process_food:
                    continue
                if equipment.held_item_id is None and not any(
                    self.legal_accepts(food, equipment.name) for food in self.food_names
                ):
                    # Takes nothing; but food a level puts straight onto a
                    # stove or mixer can still be taken off it.
                    continue
                symbol = f"s_{loc[0]}_{loc[1]}_{short_name(equipment.name)}"
                stations.append(
                    Station(
                        symbol=symbol,
                        entity=equipment,
                        location=loc,
                        supported=self._is_supported(equipment, equipments),
                    )
                )
        # A pan, pot or bowl a chef is carrying.
        for agent in self.environment.get_entities_of_type(Agent):
            if agent.held_item_id is None:
                continue
            equipment = self.environment.get_entity_as(agent.held_item_id, Equipment)
            if equipment is None or isinstance(equipment, Plate):
                continue
            stations.append(
                Station(
                    symbol=f"s_held_{pddl_name(agent.id)}_{short_name(equipment.name)}",
                    entity=equipment,
                    location=None,
                    supported=False,
                    holder=agent.id,
                )
            )
        return stations

    def burns(self, station: Station) -> bool:
        """Whether a station cooks by itself and would cook its output further,
        like a pan that turns grilled meat into burnt meat."""
        if not station.entity.cooks_by_itself:
            return False
        recipes = [r for r in self.state.cook_recipes if r.equipment == station.name]
        return bool({r.output for r in recipes} & {r.ingredient for r in recipes})

    def supports_for(self, station: Station) -> list[Location]:
        """Tiles with the stove or mixer a portable station must stand on."""
        required = station.entity.requires
        if not required:
            return []
        return sorted(
            loc
            for loc, equipments in self.equipment_by_location.items()
            if any(e.name == required for e in equipments)
        )

    def base_free(self, location: Location) -> bool:
        """Whether nothing portable stands on the stove or mixer at ``location``."""
        return not any(e.portable for e in self.equipment_by_location.get(location, []))

    @cached_property
    def support_locations(self) -> set[Location]:
        return {loc for st in self.stations for loc in self.supports_for(st)}

    @cached_property
    def station_by_symbol(self) -> dict[str, Station]:
        return {station.symbol: station for station in self.stations}

    @staticmethod
    def _is_supported(equipment: Equipment, cell: list[Equipment]) -> bool:
        if not equipment.requires:
            return True
        return any(other.name == equipment.requires for other in cell)

    @cached_property
    def loose_items(self) -> dict[Location, str]:
        """Food or plates lying on a tile with no counter or station to hold them.

        A level puts food straight on the floor when its cell has nothing to
        hold it. It can be picked up, but nothing can be put back there.
        """
        held: set[str] = set()
        for cls in (Counter, Equipment, Agent):
            for entity in self.environment.get_entities_of_type(cls):
                if entity.held_item_id is not None:
                    held.add(entity.held_item_id)
        loose: dict[Location, str] = {}
        for cls in (Food, Plate):
            for item in self.environment.get_entities_of_type(cls):
                if item.x is None or item.y is None or item.id in held:
                    continue
                if (item.x, item.y) in self.counters:
                    continue
                loose.setdefault((item.x, item.y), item.id)
        return loose

    @cached_property
    def targets(self) -> set[Location]:
        """Every tile a chef might act on."""
        return (
            set(self.counters)
            | set(self.storages)
            | set(self.dispensers)
            | self.deliveries
            | self.bins
            | self.sinks
            | {station.location for station in self.stations if station.location}
            | self.support_locations
            | set(self.loose_items)
        )

    def item_symbol(self, item_id: str | None) -> str | None:
        """The symbol for a food or plate, or None for anything else."""
        if item_id is None:
            return None
        item = self.environment.get_entity(item_id)
        if isinstance(item, Food):
            return food_symbol(item.name)
        if isinstance(item, Plate):
            if item.dirty:
                return DIRTY_PLATE
            if item.held_item_id is None:
                return EMPTY_PLATE
            food = self.environment.get_entity_as(item.held_item_id, Food)
            return plate_symbol(food.name) if food is not None else None
        return None

    def station_food(self, station: Station) -> str | None:
        if station.entity.held_item_id is None:
            return None
        food = self.environment.get_entity_as(station.entity.held_item_id, Food)
        return food.name if food is not None else None

    @cached_property
    def plate_count(self) -> int:
        """All plates in the kitchen, for sizing the dispenser counters."""
        total = len(self.environment.get_entities_of_type(Plate))
        for dispenser in self.dispensers.values():
            if not dispenser.infinite_supply:
                total += dispenser.plate_count + dispenser.dirty_plate_count
        return total

    @cached_property
    def return_dispenser(self) -> PlateDispenser | None:
        """The dispenser a delivered plate goes back to: the first in the level."""
        dispensers = sorted(
            self.environment.get_entities_of_type(PlateDispenser),
            key=lambda d: (
                d.y if d.y is not None else -1,
                d.x if d.x is not None else -1,
                d.id,
            ),
        )
        return dispensers[0] if dispensers else None

    def chef_by_symbol(self, symbol: str) -> Agent:
        for chef in self.environment.get_entities_of_type(Agent):
            if chef_symbol(chef.id) == symbol:
                return chef
        raise ValueError(f"Unknown chef in plan: {symbol}")

    def food_name(self, symbol: str) -> str:
        try:
            return self.food_by_symbol[symbol]
        except KeyError:
            raise ValueError(f"Unknown food in plan: {symbol}") from None

    def plate_food(self, symbol: str) -> str | None:
        """The food on a plate symbol, or None for an empty or dirty plate."""
        if symbol in (EMPTY_PLATE, DIRTY_PLATE):
            return None
        return self.food_name("f_" + symbol.removeprefix("p_"))


def _neighbours(location: Location) -> list[Location]:
    x, y = location
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
