"""Add walking to the PDDL model, with costs, for a single chef.

In Parts 1-3 the planner (lama-first) ignores costs, so the domain leaves
walking out. In a match our controller may use any planner configuration, so
the chef's position is modelled: every action needs the chef to stand at its
target, ``walk`` moves the chef between targets, and its cost is the real
number of steps (breadth-first search on the floor plan). Other actions cost
one step. An anytime cost-aware search (LAMA) then looks for the plan with
the fewest steps, which removes the back-and-forth trips a cost-blind plan
makes.
"""

from __future__ import annotations

import re

from simulator.types import Location

from .planner.kitchen import Kitchen, chef_symbol, loc_symbol

WALK_ACTION = """
  (:action go
    :parameters (?c - chef ?from - loc ?to - loc)
    :precondition (and (chef_at ?c ?from) (path ?from ?to))
    :effect (and (chef_at ?c ?to) (not (chef_at ?c ?from))
                 (increase (total-cost) (steps ?from ?to)))
  )
"""


def domain_with_walking(domain: str) -> str:
    """The Parts 1-3 domain with positions, walking and action costs."""
    text = domain.replace("(:requirements :strips :typing)",
                          "(:requirements :strips :typing :action-costs)")
    text = text.replace("(reach ?c ?l)", "(chef_at ?c ?l)")
    text = text.replace(
        "    (hand_empty ?c - chef)",
        "    (chef_at ?c - chef ?l - loc)\n    (path ?a - loc ?b - loc)\n    (hand_empty ?c - chef)",
        1,
    )
    text = text.replace(
        "  (:action take_food\n",
        "  (:functions (steps ?a - loc ?b - loc) - number (total-cost) - number)\n"
        + WALK_ACTION
        + "\n  (:action take_food\n",
        1,
    )
    # Every other action takes one step.
    text = re.sub(
        r"(:effect \(and )",
        r"\1(increase (total-cost) 1) ",
        text,
    )
    # ...but walking already says what it costs.
    text = text.replace(
        "(increase (total-cost) 1) (chef_at ?c ?to)", "(chef_at ?c ?to)"
    )
    # Cooking by itself takes no chef's time.
    text = re.sub(
        r"(\(:action cook_by_itself.*?:effect \(and )\(increase \(total-cost\) 1\) ",
        r"\1",
        text,
        flags=re.S,
    )
    return text


def problem_with_walking(problem: str, kitchen: Kitchen, chef_id: str) -> str:
    """Add the chef's position and the walking distances to a problem."""
    chef = next(c for c in kitchen.chefs if c.id == chef_id)
    c = chef_symbol(chef_id)
    region = kitchen.region(chef.x, chef.y)
    targets = sorted(kitchen.reach[chef_id] & kitchen.targets)
    start = (chef.x, chef.y)
    names = [loc_symbol(t) for t in targets]
    declared = set(re.findall(r"(loc_\d+_\d+) - loc", problem))
    facts: list[str] = []

    def between(dist: dict, b: Location) -> int | None:
        options = [dist[n] for n in _neighbours(b) if n in region and n in dist]
        return min(options) if options else None

    # The chef starts beside a target if one is next to it, otherwise at a
    # spot of its own.
    here = next((t for t in targets if abs(t[0] - start[0]) + abs(t[1] - start[1]) == 1), None)
    start_dist = kitchen.walk_from(start)
    if here is None:
        facts.append(f"(chef_at {c} start_here)")
    else:
        facts.append(f"(chef_at {c} {loc_symbol(here)})")
    for b in targets:
        if loc_symbol(b) not in declared:
            continue
        if here is None:
            d = between(start_dist, b)
            if d is not None:
                facts.append(f"(= (steps start_here {loc_symbol(b)}) {max(1, d)})")
                facts.append(f"(path start_here {loc_symbol(b)})")
        for a in targets:
            if a == b or loc_symbol(a) not in declared:
                continue
            d = between(kitchen.distances.get(a, {}), b)
            if d is not None:
                facts.append(f"(= (steps {loc_symbol(a)} {loc_symbol(b)}) {max(1, d)})")
                facts.append(f"(path {loc_symbol(a)} {loc_symbol(b)})")
    facts.append("(= (total-cost) 0)")
    text = problem
    if here is None:
        text = text.replace("  (:objects\n", "  (:objects\n    start_here - loc\n", 1)
    text = text.replace(f"(reach {c} ", "(chef_at_unused " + c + " ")
    text = re.sub(r"\s*\(chef_at_unused [^)]*\)", "", text)
    text = text.replace("  (:init\n", "  (:init\n" + "".join(f"    {f}\n" for f in facts), 1)
    text = text.rstrip()
    assert text.endswith(")")
    text = text[:-1] + "  (:metric minimize (total-cost))\n)\n"
    return text


def _neighbours(location: Location) -> list[Location]:
    x, y = location
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def solve_with_costs(domain: str, problem: str, seconds: float) -> list:
    """Run Fast Downward's anytime LAMA for ``seconds`` and return its best plan.

    Anytime search keeps improving its plan until the time is up, writing each
    better plan to a numbered file; the newest one is the cheapest.
    """
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from controllers.pddl.solvers.fast_downward import FastDownwardSolver
    from controllers.pddl.solvers.plan_format import parse_sas_plan

    driver = FastDownwardSolver.driver_path()
    if driver is None:
        return []
    with tempfile.TemporaryDirectory(prefix="tmc-l4-") as folder:
        folder_path = Path(folder)
        (folder_path / "domain.pddl").write_text(domain)
        (folder_path / "problem.pddl").write_text(problem)
        plan = folder_path / "sas_plan"
        command = [sys.executable, str(driver), "--plan-file", str(plan)]
        command += ["--overall-time-limit", f"{max(1, int(seconds))}s"]
        command += ["--alias", "seq-sat-lama-2011"]
        command += [str(folder_path / "domain.pddl"), str(folder_path / "problem.pddl")]
        try:
            subprocess.run(
                command, cwd=folder, capture_output=True, timeout=seconds + 3
            )
        except subprocess.TimeoutExpired:
            pass
        found = sorted(
            folder_path.glob("sas_plan*"),
            key=lambda p: int(p.suffix[1:]) if p.suffix[1:].isdigit() else 0,
        )
        if not found:
            return []
        return parse_sas_plan(found[-1])
