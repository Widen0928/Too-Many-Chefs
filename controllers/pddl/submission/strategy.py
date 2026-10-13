"""Pick a planning strategy for the level by playing the whole level out.

Different kitchens favour different strategies: planning three orders at a
time gives lama-first small, focused problems, while planning every order at
once lets chefs prepare far ahead. Since execution is deterministic (planning
costs no timesteps and the navigation policy is fixed), the problem
generator can play the level out from the start under each strategy, with
the very same planner, dispatcher, navigation policy and simulator, and keep
the strategy that scores best.

The choice is made on the first planning call and remembered on disk, keyed
by the episode's stable state id and layout, because every planning call
runs in a worker process with its own copy of the problem generator.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from simulator.environment import Environment

_DIR = Path(tempfile.gettempdir()) / "too-many-chefs-strategy"
# Stop trying further strategies once this many seconds have gone on it
# (config.selection_seconds), and never start one likely to end after
# HARD_CAP_SECONDS.
SELECTION_SECONDS = 30.0
HARD_CAP_SECONDS = 40.0


def fingerprint(environment: Environment) -> str:
    """A stable key shared by this episode's workers, never by another run."""
    from simulator.entities import (
        Bin,
        Counter,
        Delivery,
        OvercookedState,
        PlateDispenser,
        Sink,
        Storage,
    )

    state = environment.get_first_entity_of_type(OvercookedState)
    parts: list[str] = [state.name or "" if state else "", state.id if state else ""]
    for cls in (Counter, Storage, Delivery, Bin, Sink, PlateDispenser):
        for entity in environment.get_entities_of_type(cls):
            extra = getattr(entity, "food_name", "")
            parts.append(f"{cls.__name__}:{entity.x},{entity.y}:{extra}")
    if state is not None:
        parts.extend(sorted(state.food_definitions))
    return hashlib.sha256(("|".join(sorted(parts)) + _code_version()).encode()).hexdigest()


def _code_version() -> str:
    """Include the simulation rules used to compare strategies."""
    from .config import CONFIG
    from .from_pddl import _cache_namespace

    return hashlib.sha256((repr(CONFIG) + _cache_namespace()).encode()).hexdigest()


def remembered(key: str):
    try:
        return json.loads((_DIR / key).read_text())["horizon"]
    except Exception:  # noqa: BLE001
        return "missing"


def remember(key: str, horizon) -> None:
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        (_DIR / key).write_text(json.dumps({"horizon": horizon}))
    except Exception:  # noqa: BLE001
        pass


def _selection_seconds() -> float:
    from .config import CONFIG

    return CONFIG.selection_seconds


def play_out(environment: Environment, horizon, max_ticks: int = 700) -> tuple:
    """Play the level from ``environment`` to the end under ``horizon``.

    Mirrors DefaultPDDLController: plan, carry the plan out with the default
    navigation policy, and replan whenever the problem generator asks.
    Returns a score key: points, then fewer timesteps.
    """
    from controllers.navigation import DefaultMutationInterpolationPolicy
    from controllers.pddl.controller import group_mutations_by_agent
    from controllers.pddl.solvers import solve_problem_with_logging
    from simulator.context import Context
    from simulator.entities import Agent, Equipment, OvercookedState, Plate

    from .from_pddl import DefaultFromPDDL
    from .problem_generator import DefaultProblemGenerator
    from .rollout import navigation_deadlock_errors

    deadlock_errors = navigation_deadlock_errors()
    generator = DefaultProblemGenerator()
    generator.forced_horizon = horizon
    policy = DefaultMutationInterpolationPolicy()
    context = Context()
    env = environment
    start = env.get_first_entity_of_type(OvercookedState).score
    plans = None
    replan_due = False
    idle_calls = 0
    agent_ids = [
        a.id
        for a in sorted(
            env.get_entities_of_type(Agent),
            key=lambda a: (a.y if a.y is not None else -1, a.x if a.x is not None else -1, a.id),
        )
    ]
    while env.timestep - environment.timestep < max_ticks:
        state = env.get_first_entity_of_type(OvercookedState)
        if not state.visible_orders and not state.order_queue.pending:
            break
        if plans is not None:
            # should_replan may consume the event that triggered it. Keep
            # the request until a carried station is put down, exactly as
            # PDDLController does in the real run.
            if generator.should_replan(env):
                replan_due = True
            carrying = any(
                isinstance(env.get_entity(a.held_item_id), Equipment)
                and not isinstance(env.get_entity(a.held_item_id), Plate)
                for a in env.get_entities_of_type(Agent)
                if a.held_item_id
            )
            if replan_due and not carrying:
                plans = None
                policy = DefaultMutationInterpolationPolicy()
                replan_due = False
        if plans is None:
            steps = solve_problem_with_logging(generator.to_pddl(env))
            mutations = DefaultFromPDDL(env).from_pddl(steps)
            plans = group_mutations_by_agent(mutations=mutations, agent_ids=agent_ids)
        try:
            actions, plans = policy.get_mutations(plans, env, context)
        except deadlock_errors:
            break
        if not actions:
            if all(not queue for queue in plans.values()):
                break  # the controller would stop here
            if not env.has_pending_effects():
                idle_calls += 1
                if idle_calls > 12:
                    break  # stuck for good
                continue
        idle_calls = 0
        env = env.step(actions)
    end = env.get_first_entity_of_type(OvercookedState)
    return (end.score - start, -(env.timestep - environment.timestep))


def choose_horizon(environment: Environment, options: list) -> object:
    """The horizon whose play-out scores best, remembered for the level."""
    key = fingerprint(environment)
    known = remembered(key)
    if known != "missing":
        return known
    import sys
    import time

    from .config import CONFIG
    from .score_bound import remaining_score_upper_bound

    score_bound = remaining_score_upper_bound(environment) if CONFIG.stop_at_score_bound else None
    options = strategy_options(environment, options)
    print(f"[tmc] strategy options {options}", file=sys.stderr, flush=True)
    best = None
    started = time.monotonic()
    longest = 0.0
    for horizon in options:
        spent = time.monotonic() - started
        # The server stops a Level 3 run after 90 s (Level 2: 60 s), and the
        # run itself takes a while: stop once past the budget, or before a
        # play-out that would likely end past the hard cap.
        if best is not None and (
            spent > _selection_seconds() or spent + longest > HARD_CAP_SECONDS
        ):
            break
        began = time.monotonic()
        try:
            score = play_out(environment, horizon)
        except Exception:  # noqa: BLE001 - a failed play-out just loses
            continue
        finally:
            longest = max(longest, time.monotonic() - began)
        print(f"[tmc] strategy {horizon!r} score {score} in {time.monotonic() - began:.1f}s",
              file=sys.stderr, flush=True)
        if best is None or score > best[0]:
            best = (score, horizon)
        if score_bound is not None and best[0][0] >= score_bound:
            print(f"[tmc] reached remaining-score bound {score_bound}",
                  file=sys.stderr, flush=True)
            break
    chosen = best[1] if best is not None else options[0]
    print(f"[tmc] strategy chosen {chosen!r} after {time.monotonic() - started:.1f}s",
          file=sys.stderr, flush=True)
    remember(key, chosen)
    return chosen


# Played out first on rush levels (see choose_horizon).
RUSH_FIRST = ("3r", "2r", "1r", "3safer", "r", 3, 2)


def strategy_options(environment: Environment, options: list) -> list:
    """Keep baseline coverage before comparing optional whole-run swaps."""
    from simulator.entities import OvercookedState

    from .config import CONFIG

    options = list(options)
    if _rush(environment):
        options = [o for o in RUSH_FIRST if o in options or o == "3safer"] + [
            o for o in options if o not in RUSH_FIRST
        ]
    queue = environment.get_first_entity_of_type(OvercookedState).order_queue
    if (
        CONFIG.select_order_batches
        and queue.order_reveal == "all"
        and not queue.strict_ordering
        and len({entry.name for entry in queue.visible}) < len(queue.visible)
        and len({entry.name for entry in queue.visible}) >= 2
        and any(entry.time_limit is not None for entry in queue.visible)
        and "3r" in options
    ):
        from .kitchen import Kitchen

        kitchen = Kitchen(environment)
        # Repeated dishes on a cooking chain can avoid costly alternation
        # between short and long recipes. Compare the full run, since this
        # intentionally changes which deadline the controller serves first.
        has_chain = any(
            {recipe.output for recipe in kitchen.state.cook_recipes if recipe.equipment == station.name}
            & {recipe.ingredient for recipe in kitchen.state.cook_recipes if recipe.equipment == station.name}
            for station in kitchen.stations
        )
        if has_chain and "3rb" not in options:
            options.insert(options.index("3r") + 1, "3rb")
        # A short first ticket can block a feasible four-dish plan. Also
        # compare planning the complete queue with a local deadline swap;
        # a full rollout rejects it when the urgent ticket is still better.
        if has_chain and len(queue.visible) >= 4 and "swap" not in options:
            options.insert(options.index("3rb") + 1, "swap")
        if has_chain and len(queue.visible) == 4 and "4rq5" not in options:
            options.insert(options.index("swap") + 1, "4rq5")
    if (
        CONFIG.select_deadline_alternatives
        and not CONFIG.deadline_alternatives
        and queue.order_reveal == "all"
        and not queue.strict_ordering
        and len(queue.visible) >= 2
        and any(entry.time_limit is not None for entry in queue.visible)
    ):
        for candidate in ("3swap", "3rswap", "2rswap"):
            if candidate not in options:
                options.append(candidate)
    return options


def _rush(environment: Environment) -> bool:
    from simulator.entities import OvercookedState

    queue = environment.get_first_entity_of_type(OvercookedState).order_queue
    entries = [*queue.visible, *queue.pending]
    if len(entries) < 2:
        return False
    limits = [e.time_limit for e in entries if e.time_limit is not None]
    return queue.order_reveal == "sequential" or bool(limits and min(limits) <= 30)
