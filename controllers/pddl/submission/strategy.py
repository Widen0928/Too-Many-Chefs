"""Pick a planning strategy for the level by playing the whole level out.

Different kitchens favour different strategies: planning three orders at a
time gives lama-first small, focused problems, while planning every order at
once lets chefs prepare far ahead. Since execution is deterministic (planning
costs no timesteps and the navigation policy is fixed), the problem
generator can play the level out from the start under each strategy, with
the very same planner, dispatcher, navigation policy and simulator, and keep
the strategy that scores best.

The choice is made on the first planning call and remembered on disk, keyed
by the kitchen's fixed layout, because every planning call runs in a worker
process with its own copy of the problem generator.
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
    """A key for the level: its name and everything that never moves."""
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
    parts: list[str] = [state.name or "" if state else ""]
    for cls in (Counter, Storage, Delivery, Bin, Sink, PlateDispenser):
        for entity in environment.get_entities_of_type(cls):
            extra = getattr(entity, "food_name", "")
            parts.append(f"{cls.__name__}:{entity.x},{entity.y}:{extra}")
    if state is not None:
        parts.extend(sorted(state.food_definitions))
    return hashlib.sha256(("|".join(sorted(parts)) + _code_version()).encode()).hexdigest()


def _code_version() -> str:
    """The submission's own source and settings, so a stale choice made by
    an older version of the code is never reused."""
    from .config import CONFIG

    digest = hashlib.sha256(repr(CONFIG).encode())
    for path in sorted(Path(__file__).parent.glob("*.p*")):
        if path.suffix in (".py", ".pddl"):
            digest.update(path.read_bytes())
    return digest.hexdigest()


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
    from controllers.navigation.traffic import AgentDeadlockError
    from controllers.pddl.controller import group_mutations_by_agent
    from controllers.pddl.solvers import solve_problem_with_logging
    from simulator.context import Context
    from simulator.entities import Agent, Equipment, OvercookedState, Plate

    from .from_pddl import DefaultFromPDDL
    from .problem_generator import DefaultProblemGenerator

    generator = DefaultProblemGenerator()
    generator.forced_horizon = horizon
    policy = DefaultMutationInterpolationPolicy()
    context = Context()
    env = environment
    start = env.get_first_entity_of_type(OvercookedState).score
    plans = None
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
        if plans is not None and generator.should_replan(env):
            carrying = any(
                isinstance(env.get_entity(a.held_item_id), Equipment)
                and not isinstance(env.get_entity(a.held_item_id), Plate)
                for a in env.get_entities_of_type(Agent)
                if a.held_item_id
            )
            if not carrying:
                plans = None
                policy = DefaultMutationInterpolationPolicy()
        if plans is None:
            steps = solve_problem_with_logging(generator.to_pddl(env))
            mutations = DefaultFromPDDL(env).from_pddl(steps)
            plans = group_mutations_by_agent(mutations=mutations, agent_ids=agent_ids)
        try:
            actions, plans = policy.get_mutations(plans, env, context)
        except AgentDeadlockError:
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
    import time

    import sys

    if _rush(environment):
        # Orders revealed one at a time, or one due very soon (a rush
        # ticket): list scheduling earned full points on such a kitchen, so
        # its strategies are played out first.
        options = [o for o in RUSH_FIRST if o in options or o == "3safer"] + [
            o for o in options if o not in RUSH_FIRST
        ]
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
    chosen = best[1] if best is not None else options[0]
    print(f"[tmc] strategy chosen {chosen!r} after {time.monotonic() - started:.1f}s",
          file=sys.stderr, flush=True)
    remember(key, chosen)
    return chosen


# Played out first on rush levels (see choose_horizon).
RUSH_FIRST = ("3r", "2r", "1r", "3safer", "r", 3, 2)


def _rush(environment: Environment) -> bool:
    from simulator.entities import OvercookedState

    queue = environment.get_first_entity_of_type(OvercookedState).order_queue
    entries = [*queue.visible, *queue.pending]
    if len(entries) < 2:
        return False
    limits = [e.time_limit for e in entries if e.time_limit is not None]
    return queue.order_reveal == "sequential" or bool(limits and min(limits) <= 30)
