"""Give each part of a plan to the chef who can finish it soonest.

The planner (lama-first) ignores action costs and does not model walking, so
it decides *what* has to happen correctly but picks chefs, storages and bins
more or less arbitrarily: in a kitchen with two identical halves it may send
all four chefs to the same pot. This module keeps the plan's steps and their
order, and re-decides *who* does them.

1. The plan is cut into jobs. A job is what one chef does between having
   empty hands and having empty hands again (take a fish, put it on the
   cutboard), so any chef who can reach all of a job's tiles could do it
   instead. A chop on its own is a job too.
2. Jobs are taken in plan order and each goes to the chef with the earliest
   estimated finish time. The estimate walks the chef along the job's tiles
   using breadth-first distances on the floor plan and waits wherever an
   earlier job must have finished first: one that used the same counter,
   station or dispenser, plus the cooking time of stations that cook by
   themselves.
3. Steps whose tile does not matter (which storage of the same food, which
   bin or sink) are moved to the one closest to the chef.

Each chef's steps stay in plan order, so the controller's step-by-step
legality checks still keep chefs in step with one another.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from controllers.pddl.interfaces import PDDLPlanStep
from simulator.types import Location

from .kitchen import Kitchen, chef_symbol, loc_symbol, parse_loc

# Steps that start a job (empty hands -> holding) and that end one.
BEGIN = {
    "take_food",
    "take_plate_infinite",
    "take_plate_clean",
    "take_plate_dirty",
    "pick_up",
    "take_out",
    "lift_station_from_counter",
    "lift_station_from_support",
}
END = {
    "put_down",
    "combine_on_counter",
    "food_onto_counter_plate",
    "put_in_station",
    "add_to_station",
    "discard_food",
    "deliver",
    "deliver_return_clean",
    "deliver_return_dirty",
    "set_station_on_support",
    "set_station_on_counter",
}
# Steps that act on a station: (station parameter, location parameter).
STATION_STEPS = {
    "station_onto_plate_tended",
    "put_in_station", "add_to_station", "combine_from_station", "take_out", "cook",
    "station_onto_plate", "plate_into_station", "plate_add_to_station",
}
FILL_STEPS = {"put_in_station", "plate_into_station"}
EMPTY_STEPS = {
    "take_out", "station_onto_plate", "station_onto_plate_tended", "combine_from_station",
}
# Steps whose target can be swapped for any tile of the same kind.
STORAGE_STEPS = {"take_food", "take_food_onto_plate"}
BIN_STEPS = {"discard_food", "discard_from_plate"}
SINK_STEPS = {"wash"}
DELIVERY_STEPS = {"deliver", "deliver_return_clean", "deliver_return_dirty"}


DEBUG = False


@dataclass
class Job:
    steps: list[tuple[int, PDDLPlanStep]] = field(default_factory=list)
    chef: str | None = None  # original chef symbol
    pinned: bool = False  # must stay with its chef (it was holding at the start)


def dispatch(kitchen: Kitchen, steps: list[PDDLPlanStep]) -> list[PDDLPlanStep]:
    try:
        return _Dispatcher(kitchen, steps).run()
    except Exception:  # noqa: BLE001 - the original plan is always valid
        return steps


class _Dispatcher:
    def __init__(self, kitchen: Kitchen, steps: list[PDDLPlanStep]) -> None:
        self.k = kitchen
        self.steps = steps
        plan_chefs = {s.parameters[0] for s in steps if _has_chef(s)}
        teams = {
            chef.team for chef in kitchen.environment_agents() if chef_symbol(chef.id) in plan_chefs
        }
        self.chefs = {
            chef_symbol(chef.id): chef
            for chef in kitchen.environment_agents()
            if chef.x is not None and (chef_symbol(chef.id) in plan_chefs or chef.team in teams)
        }
        self.regions = {
            c: kitchen.region(chef.x, chef.y) for c, chef in self.chefs.items()
        }
        self.reach = {
            c: {
                n
                for tile in region
                for n in _neighbours(tile)
                if n not in kitchen.walkable
            }
            for c, region in self.regions.items()
        }
        self._find_episodes()
        # Dead-end tiles: one chef waiting in one while another needs it
        # makes the controller's traffic rules give up with a deadlock, so
        # each dead end is given to a single chef.
        self.pocket_owner: dict[Location, str] = {}
        for c, chef in self.chefs.items():
            tile = (chef.x, chef.y)
            if self._dead_end(tile):
                self.pocket_owner[tile] = c
        # Episode -> station symbol it was moved to.
        self.moved: dict[int, str] = {}
        # Spare station -> plan index of the last step of the last episode moved there.
        self.spare_until: dict[str, float] = {}
        # Steps already given to a chef; an episode can only move before any
        # of its steps has been.
        self.committed: set[int] = set()

    # --- Station episodes -----------------------------------------------------

    def _find_episodes(self) -> None:
        """Group the steps that use a station from filling it to emptying it.

        The planner often sends every dish through one station while an
        identical one stands idle. A whole episode (put the rice in the pot,
        cook it, take it out) can move to an idle twin, which spreads chefs
        out and lets dishes cook side by side.
        """
        used = {
            sym
            for step in self.steps
            for sym in step.parameters
            if sym in self.k.station_by_symbol
        }
        self.spares: dict[str, list[str]] = {}
        for station in self.k.stations:
            self.spares[station.symbol] = [
                other.symbol
                for other in self.k.stations
                if other.name == station.name
                and other.symbol != station.symbol
                and other.symbol not in used
                and other.supported
                and other.entity.held_item_id is None
                and other.entity.portable == station.entity.portable
            ]
        self.episode_of: dict[int, int] = {}
        self.episode_start: dict[int, float] = {}
        open_episode: dict[str, int] = {}
        for index, step in enumerate(self.steps):
            station = _station_of(step)
            if station is None:
                continue
            if step.name in FILL_STEPS:
                open_episode[station] = index
                self.episode_start[index] = index
            episode = open_episode.get(station)
            if episode is None:
                continue
            self.episode_of[id(step)] = episode
            if step.name in EMPTY_STEPS:
                open_episode.pop(station, None)
        # Episodes still open at the end of the plan stay where they are.
        for station, episode in open_episode.items():
            for key in [k for k, v in self.episode_of.items() if v == episode]:
                del self.episode_of[key]

    # --- Cutting the plan into jobs ---------------------------------------------

    def jobs(self) -> list[Job | tuple[int, PDDLPlanStep]]:
        """Jobs in plan order; chefless steps (cooking by itself) stay as they are."""
        out: list[Job | tuple[int, PDDLPlanStep]] = []
        open_jobs: dict[str, Job] = {}
        for c, chef in self.chefs.items():
            if chef.held_item_id is not None:
                job = Job(chef=c, pinned=True)
                open_jobs[c] = job
                out.append(job)
        for index, step in enumerate(self.steps):
            if not _has_chef(step):
                out.append((index, step))
                continue
            c = step.parameters[0]
            job = open_jobs.get(c)
            if job is None:
                job = Job(chef=c)
                out.append(job)
                if step.name in BEGIN:
                    open_jobs[c] = job
            job.steps.append((index, step))
            if step.name in END:
                open_jobs.pop(c, None)
        out = [j for j in out if not isinstance(j, Job) or j.steps]
        return self._pair_burnable(out)

    def _pair_burnable(self, items: list) -> list:
        """Keep food from burning in stations that keep cooking.

        In a pan that turns grilled meat into burnt meat, food must come out
        soon after it is done. A job that only fetches food from a storage
        and puts it into such a station is moved to just before the job that
        takes the food out again, and both go to the same chef, so the food
        goes in only when someone is about to collect it.
        """
        risky = self._burnable_stations()
        if not risky:
            return items
        jobs = [j for j in items if isinstance(j, Job)]
        deferred: set[int] = set()
        for i, job in enumerate(jobs):
            names = [step.name for _, step in job.steps]
            if (
                job.pinned
                or names[0] not in ("take_food", "pick_up")
                or names[-1] != "put_in_station"
            ):
                continue
            station = job.steps[-1][1].parameters[1]
            if station not in risky:
                continue
            put_index = job.steps[-1][0]
            # The step that takes this food out again, with nothing else
            # going into the station first.
            taker = None
            for index, step in enumerate(self.steps):
                if index <= put_index or _station_of(step) != station:
                    continue
                if step.name in EMPTY_STEPS:
                    taker = step
                if step.name not in ("cook", "cook_by_itself"):
                    break
            if taker is None:
                continue
            later = next(
                (j for j in jobs if any(st is taker for _, st in j.steps)), None
            )
            if later is None or later.pinned or later is job:
                continue
            first = later.steps[0][0]
            if first <= put_index:
                continue
            count = len(job.steps)
            moved = [
                (first - 0.5 + k / (2 * count + 2), step)
                for k, (_, step) in enumerate(job.steps)
            ]
            later.steps = moved + later.steps
            deferred.add(id(job))
        return [j for j in items if id(j) not in deferred]

    def _burnable_stations(self) -> set[str]:
        return {st.symbol for st in self.k.stations if self.k.burns(st)}


    # --- Scheduling --------------------------------------------------------------

    def run(self) -> list[PDDLPlanStep]:
        free = {c: 0.0 for c in self.chefs}
        pos: dict[str, Location] = {
            c: (chef.x, chef.y) for c, chef in self.chefs.items()
        }
        at_target: dict[str, Location | None] = {c: None for c in self.chefs}
        # Resource -> (plan index, time) of each use so far. A step waits only
        # for uses that come before it in the plan.
        ready: dict[object, list[tuple[float, float]]] = {}
        # A chef's jobs must not interleave in plan order: if they did, a job
        # could end up waiting on a later step of the same chef.
        last_index = {c: -1 for c in self.chefs}
        result: list[tuple[int, PDDLPlanStep]] = []

        for item in self.jobs():
            if not isinstance(item, Job):
                index, step = item
                # Cooking time was added when the food went in.
                episode = self.episode_of.get(id(step))
                if episode is not None and episode in self.moved:
                    step = _on_station(step, self.moved[episode], self.k)
                result.append((index, step))
                continue
            job = item
            candidates = [job.chef] if job.pinned else list(self.chefs)
            best = None
            first = job.steps[0][0]
            for c in candidates:
                if last_index[c] > first:
                    continue
                trial = self._simulate(job, c, free[c], pos[c], at_target[c], ready)
                if trial is None:
                    continue
                finish = trial[0]
                key = (finish, c != job.chef)
                if best is None or key < best[0]:
                    best = (key, c, trial)
            if best is None:
                # Nobody else fits; leave the job as planned.
                result.extend(job.steps)
                self.committed.update(id(step) for _, step in job.steps)
                if job.chef in last_index:
                    last_index[job.chef] = max(last_index[job.chef], job.steps[-1][0])
                continue
            _, c, (finish, new_steps, last_target, touched, moves, pockets) = best
            self.pocket_owner.update(pockets)
            self.committed.update(id(step) for _, step in job.steps)
            for episode, station in moves.items():
                self.moved[episode] = station
                if station != _station_of(self.steps[episode]):
                    self.spare_until[station] = self._end_of_episode(episode)
            if DEBUG:
                print(f"{c:12s} {free[c]:6.1f}->{finish:6.1f} "
                      + "; ".join(f"{i}:{st.name} {' '.join(st.parameters[1:])}" for i, st in new_steps))
            free[c] = finish
            last_index[c] = job.steps[-1][0]
            at_target[c] = last_target
            for key, uses in touched.items():
                ready.setdefault(key, []).extend(uses)
            result.extend(new_steps)

        result.sort(key=lambda pair: pair[0])
        return [step for _, step in result]

    def _simulate(self, job: Job, c: str, time: float, start: Location,
                  at: Location | None, ready: dict[object, list[tuple[float, float]]]):
        """Walk chef ``c`` through ``job``; None if it cannot reach a tile."""
        reach = self.reach[c]
        new_steps: list[tuple[int, PDDLPlanStep]] = []
        touched: dict[object, list[tuple[float, float]]] = {}
        moves: dict[int, str] = {}
        pockets: dict[Location, str] = {}
        for index, step in job.steps:
            step = self._move_station(step, index, c, at, start, moves)
            step = self._retarget(step, c, at, start)
            target = _target(step)
            if target is not None:
                if target not in reach:
                    return None
                pocket = self._pocket(c, target)
                if pocket is not None:
                    for tile in pocket:
                        owner = self.pocket_owner.get(tile) or pockets.get(tile)
                        if owner is not None and owner != c:
                            return None
                        pockets[tile] = c
                time += self._walk(c, at, start, target)
                key = self._resource(step, target)
                if key is not None:
                    for uses in (ready.get(key, ()), touched.get(key, ())):
                        for used_at, done in uses:
                            if used_at < index:
                                time = max(time, done)
                at = target
            time += self._duration(step)
            if target is not None:
                key = self._resource(step, target)
                if key is not None:
                    touched.setdefault(key, []).append((index, time + self._cook_delay(step)))
            new_steps.append((index, _with_chef(step, c)))
        return time, new_steps, at, touched, moves, pockets

    def _pocket(self, c: str, target: Location) -> frozenset | None:
        """The dead-end floor tiles chef ``c`` must stand on to use ``target``.

        None unless every tile from which ``c`` can reach the target is a dead
        end (a floor tile with a single floor neighbour).
        """
        region = self.regions[c]
        tiles = [t for t in _neighbours(target) if t in region]
        if not tiles:
            return None
        if not all(self._dead_end(tile) for tile in tiles):
            return None
        return frozenset(tiles)

    def _dead_end(self, tile: Location) -> bool:
        return sum(1 for n in _neighbours(tile) if n in self.k.walkable) <= 1

    def _move_station(self, step: PDDLPlanStep, index: float, c: str,
                      at: Location | None, start: Location,
                      moves: dict[int, str]) -> PDDLPlanStep:
        episode = self.episode_of.get(id(step))
        if episode is None:
            return step
        target = self.moved.get(episode) or moves.get(episode)
        if target is None and step.name in FILL_STEPS and not any(
            id(other) in self.committed
            for other in self.steps
            if self.episode_of.get(id(other)) == episode
        ):
            original = _station_of(step)
            options = [original] + [
                spare
                for spare in self.spares.get(original, [])
                if self.spare_until.get(spare, -1) < index
                and self.k.station_by_symbol[spare].location in self.reach[c]
            ]
            target = min(
                options,
                key=lambda sym: (
                    self._walk(c, at, start, self.k.station_by_symbol[sym].location),
                    sym != original,
                ),
            )
            moves[episode] = target
        if target is None or target == _station_of(step):
            return step
        return _on_station(step, target, self.k)

    def _end_of_episode(self, episode: int) -> float:
        return max(
            index
            for index, step in enumerate(self.steps)
            if self.episode_of.get(id(step)) == episode
        )

    def _walk(self, c: str, at: Location | None, start: Location, target: Location) -> float:
        region = self.regions[c]
        dist = self.k.distances.get(target, {})
        if at is None:
            tiles = [start]
        else:
            tiles = [n for n in _neighbours(at) if n in region]
        best = min((dist[t] for t in tiles if t in dist), default=None)
        return 30.0 if best is None else float(best)

    def _retarget(self, step: PDDLPlanStep, c: str, at: Location | None,
                  start: Location) -> PDDLPlanStep:
        """Swap a storage, bin, sink or delivery tile for the nearest equivalent."""
        if step.name in STORAGE_STEPS:
            food = step.parameters[2]
            options = [
                loc for loc, storage in self.k.storages.items()
                if _food_sym(storage.food_name) == food
            ]
        elif step.name in BIN_STEPS:
            options = list(self.k.bins)
        elif step.name in SINK_STEPS:
            options = list(self.k.sinks)
        elif step.name in DELIVERY_STEPS:
            options = list(self.k.deliveries)
        else:
            return step
        options = [o for o in options if o in self.reach[c]]
        if not options:
            return step
        best = min(options, key=lambda o: (self._walk(c, at, start, o), o))
        params = list(step.parameters)
        params[1] = loc_symbol(best)
        return PDDLPlanStep(step.name, tuple(params))

    def _resource(self, step: PDDLPlanStep, target: Location) -> object | None:
        """What an earlier job must have finished with before this step."""
        if step.name in DELIVERY_STEPS:
            return "delivery"
        if step.name in STORAGE_STEPS or step.name in BIN_STEPS or step.name in SINK_STEPS:
            return None
        return target

    def _duration(self, step: PDDLPlanStep) -> float:
        if step.name == "cook":
            station = self.k.station_by_symbol.get(step.parameters[1])
            return float(station.entity.cook_time) if station else 1.0
        return 1.0

    def _cook_delay(self, step: PDDLPlanStep) -> float:
        """Time a station that cooks by itself needs after food goes in."""
        if step.name in ("put_in_station", "add_to_station", "plate_into_station",
                         "plate_add_to_station"):
            station = self.k.station_by_symbol.get(step.parameters[1])
            if station is not None and station.entity.cooks_by_itself:
                return float(station.entity.cook_time)
        return 0.0

def _station_of(step: PDDLPlanStep) -> str | None:
    if step.name == "cook_by_itself":
        return step.parameters[0]
    if step.name in STATION_STEPS:
        return step.parameters[1]
    return None


def _on_station(step: PDDLPlanStep, station: str, kitchen: Kitchen) -> PDDLPlanStep:
    params = list(step.parameters)
    if step.name == "cook_by_itself":
        params[0] = station
    else:
        params[1] = station
        params[2] = loc_symbol(kitchen.station_by_symbol[station].location)
    return PDDLPlanStep(step.name, tuple(params))


def _has_chef(step: PDDLPlanStep) -> bool:
    return bool(step.parameters) and step.parameters[0].startswith("c_")


def _with_chef(step: PDDLPlanStep, chef: str) -> PDDLPlanStep:
    return PDDLPlanStep(step.name, (chef, *step.parameters[1:]))


def _target(step: PDDLPlanStep) -> Location | None:
    for p in step.parameters:
        if p.startswith("loc_"):
            return parse_loc(p)
    return None


def _food_sym(name: str) -> str:
    from .kitchen import food_symbol

    return food_symbol(name)


def _neighbours(location: Location) -> list[Location]:
    x, y = location
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
