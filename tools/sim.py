"""Run the PDDL pipeline in one process, step by step, for debugging.

It mirrors DefaultPDDLController: plan when needed, ask the problem generator
whether to replan, and turn plans into actions with the default navigation
policy. Planning takes no timesteps, as in the real run.

Usage: uv run python tools/sim.py LEVEL [--max-steps N] [--show-from T] [--quiet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.navigation import DefaultMutationInterpolationPolicy  # noqa: E402
from controllers.pddl.controller import group_mutations_by_agent, PDDLController  # noqa: E402
from controllers.pddl.solvers import solve_problem_with_logging  # noqa: E402
from controllers.pddl.submission.from_pddl import DefaultFromPDDL  # noqa: E402
from controllers.pddl.submission.problem_generator import DefaultProblemGenerator  # noqa: E402
from simulator.configuration.load import load, load_configuration  # noqa: E402
from simulator.context import Context  # noqa: E402
from simulator.entities import Agent, Equipment, OvercookedState, Plate  # noqa: E402


def grid(env) -> str:
    from simulator.entities import Bounds, Entity

    b = env.get_first_entity_of_type(Bounds)
    rows = [["." for _ in range(b.width)] for _ in range(b.height)]
    marks = {"counter": "-", "storage": "S", "delivery": "*", "bin": "X", "plate": "p",
             "equipment": "E", "plate_dispenser": "D", "sink": "W"}
    for e in env.get_entities_of_type(Entity):
        d = e.to_dict()
        if d.get("x") is None or d["kind"] in ("bounds", "shadows", "food"):
            continue
        ch = d["costume"] if d["kind"] == "agent" else marks.get(d["kind"], "?")
        if rows[d["y"]][d["x"]] in ".-" or d["kind"] == "agent":
            rows[d["y"]][d["x"]] = str(ch)
    return "\n".join("".join(r) for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("level")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--show-from", type=int, default=10**9)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    env = load(load_configuration(args.level))
    generator = DefaultProblemGenerator()
    policy = DefaultMutationInterpolationPolicy()
    plans = None
    context = Context()
    frozen_guard = 0
    while env.timestep < args.max_steps:
        state = env.get_first_entity_of_type(OvercookedState)
        if not state.visible_orders and not state.order_queue.pending:
            break
        if plans is not None and generator.should_replan(env):
            carrying = any(
                isinstance(env.get_entity(a.held_item_id), Equipment)
                and not isinstance(env.get_entity(a.held_item_id), Plate)
                for a in env.get_entities_of_type(Agent) if a.held_item_id
            )
            if not carrying:
                plans = None
                policy = DefaultMutationInterpolationPolicy()
        if plans is None:
            problem = generator.to_pddl(env)
            steps = solve_problem_with_logging(problem)
            muts = DefaultFromPDDL(env).from_pddl(steps)
            ids = PDDLController.ordered_agent_ids(env)
            plans = group_mutations_by_agent(mutations=muts, agent_ids=ids)
            if not args.quiet:
                print(f"--- plan at t={env.timestep}: {len(muts)} mutations")
                for agent, queue in plans.items():
                    print(" ", agent, [f"{i}:{type(m.mutation).__name__}@{m.location}" for i, m in queue][:12])
        actions, plans = policy.get_mutations(plans, env, context)
        if not actions:
            if all(not q for q in plans.values()):
                print("finished plans with orders left")
                break
            if not env.has_pending_effects():
                frozen_guard += 1
                if frozen_guard > 10:
                    print("frozen")
                    break
                continue
        frozen_guard = 0
        env = env.step(actions)
        if env.timestep >= args.show_from:
            print(f"t={env.timestep}", {a.id: (a.x, a.y, a.held_item_id is not None) for a in env.get_entities_of_type(Agent)})
            print(grid(env))
            print(" next:", {a: [f"{i}:{type(m.mutation).__name__}@{m.location}" for i, m in q][:3] for a, q in plans.items()})
    state = env.get_first_entity_of_type(OvercookedState)
    print("t", env.timestep, "score", state.score, "delivered", state.order_queue.delivered_count,
          "expired", state.order_queue.expired_count)


if __name__ == "__main__":
    main()
