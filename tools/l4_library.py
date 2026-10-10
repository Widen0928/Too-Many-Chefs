"""Build the Part 4 plan library (controllers/open/submission/library.json).

For every dish on the tournament menu, from each team's side of the kitchen,
with both PDDL models (quick plating and tip-keeping), this collects:

- the lama-first plan the controller would get during a match, and
- a plan from an anytime cost-aware search (LAMA, ``SECONDS`` seconds) on
  the model with walking costs (controllers/open/submission/movement.py),
  which lama-first cannot afford during a match,

then reorders each (schedule.improve) and keeps every distinct valid plan.
In a match the controller checks each plan against the kitchen as it is,
moves its stations to free ones nearby and takes the quickest.

Usage: uv run python tools/l4_library.py [SECONDS] [--jobs N]
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LEVEL = ROOT / "levels" / "3_too_many_chefs" / "0_full_menu.yaml"
OUT = ROOT / "controllers" / "open" / "submission" / "library.json"
CHEFS = ["Ada", "Cyd"]


def _job(args):
    dish, chef_id, fast, seconds = args
    from controllers.open.submission import library, movement, routing, schedule
    from controllers.pddl.interfaces import PDDLProblem
    from controllers.pddl.solvers import solve_problem_with_logging
    from controllers.open.submission.planner.kitchen import Kitchen, chef_symbol
    from controllers.open.submission.planner.problem_generator import (
        build_domain,
        build_problem,
    )
    from simulator.configuration.load import load, load_configuration
    from simulator.entities import OvercookedState

    env = load(load_configuration(str(LEVEL)))
    state = env.get_first_entity_of_type(OvercookedState)
    kitchen = Kitchen(env, [chef_id])
    chef = kitchen.chefs[0]
    entry = state.order_queue.visible[0].model_copy(update={"name": f"catalog/food/{dish}"})
    problem = build_problem(kitchen, [entry], fast=fast)
    init, goal = library.facts(problem)
    found = []
    steps = solve_problem_with_logging(
        PDDLProblem(domain=build_domain(), problem=problem), time_limit_seconds=10
    )
    if steps:
        found.append(steps)
    if seconds > 0:
        steps = movement.solve_with_costs(
            movement.domain_with_walking(build_domain()),
            movement.problem_with_walking(problem, kitchen, chef_id),
            seconds,
        )
        steps = [s for s in steps if s.name != "go"]
        if steps and library.valid(steps, init, goal):
            found.append(steps)
    plans = []
    for steps in found:
        for variant in (steps, routing.relocate_stations(kitchen, chef, steps, env)):
            better = schedule.improve(kitchen, chef, variant, init, goal, seconds=5)
            # Also a version that keeps watch over food in shared stations.
            guarded = schedule.improve(kitchen, chef, variant, init, goal, seconds=5, guard=1.0)
            for plan in (variant, better, guarded):
                if library.valid(plan, init, goal):
                    plans.append(
                        (schedule.ticks(kitchen, chef, plan),
                         [[s.name, *s.parameters] for s in plan])
                    )
    c = chef_symbol(chef_id)
    out = []
    for t, plan in sorted(plans):
        general = [[name, *("?chef" if p == c else p for p in params)] for name, *params in plan]
        if general not in [g for _, g in out]:
            out.append((t, general))
    return dish, chef_id, fast, out


def main() -> None:
    args = sys.argv[1:]
    jobs = 8
    if "--jobs" in args:
        i = args.index("--jobs")
        jobs = int(args[i + 1])
        del args[i : i + 2]
    seconds = float(args[0]) if args else 30.0
    from simulator.configuration.load import load, load_configuration
    from simulator.entities import OvercookedState

    state = load(load_configuration(str(LEVEL))).get_first_entity_of_type(OvercookedState)
    dishes = sorted({o.food.rsplit("/", 1)[-1] for o in state.order_queue.definitions}) if hasattr(
        state.order_queue, "definitions") else None
    if not dishes:
        dishes = [
            "coconut_juice", "fruit_salad", "fruit_parfait", "hamburger", "cheeseburger",
            "fish_and_rice_bowl", "sushi", "chicken_and_chips", "black_forest_cake",
        ]
    work = [(d, c, f, seconds) for d in dishes for c in CHEFS for f in (True, False)]
    library: dict[str, list] = {}
    with ProcessPoolExecutor(jobs) as pool:
        for dish, chef_id, fast, plans in pool.map(_job, work):
            print(dish, chef_id, "fast" if fast else "safe", [round(t) for t, _ in plans], flush=True)
            for t, plan in plans:
                if plan not in [p for _, p in library.setdefault(dish, [])]:
                    library[dish].append((t, plan))
    saved = {
        dish: [plan for _, plan in sorted(plans, key=lambda tp: tp[0])][:16]
        for dish, plans in sorted(library.items())
    }
    OUT.write_text(json.dumps(saved, indent=0))
    print("wrote", OUT, sum(len(p) for p in saved.values()), "plans")


if __name__ == "__main__":
    main()
