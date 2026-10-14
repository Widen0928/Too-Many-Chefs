"""Plan the orders one at a time and chain the plans.

With several orders in one problem, lama-first decides freely which
grilled patty or which plate goes into which dish. It often builds the
second order's parts first, so the most urgent dish is finished last and
expires. Planning one order at a time, each from the state the previous
plan leaves (the domain is plain STRIPS, so that state is exact), puts the
urgent dish's steps first. The dispatcher's list scheduling then hands the
later orders' independent jobs to the chef that is free, so the dishes are
still prepared side by side.
"""

from __future__ import annotations

import re
from functools import lru_cache

from controllers.pddl.interfaces import PDDLPlanStep, PDDLProblem


def _tokens(text: str) -> list[str]:
    text = re.sub(r";[^\n]*", "", text)
    return re.findall(r"\(|\)|[^\s()]+", text.lower())


def _parse(tokens: list[str]):
    stack: list[list] = [[]]
    for token in tokens:
        if token == "(":
            stack.append([])
        elif token == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(token)
    return stack[0]


def _literals(expr) -> tuple[list[tuple], list[tuple]]:
    if not expr:
        return [], []
    if expr[0] == "and":
        pos, neg = [], []
        for part in expr[1:]:
            p, n = _literals(part)
            pos += p
            neg += n
        return pos, neg
    if expr[0] == "not":
        return [], [tuple(expr[1])]
    return [tuple(expr)], []


@lru_cache(maxsize=4)
def _actions(domain: str) -> dict[str, tuple]:
    """name -> (parameter names, preconditions, adds, deletes)."""
    tree = _parse(_tokens(domain))[0]
    found = {}
    for part in tree:
        if not isinstance(part, list) or not part or part[0] != ":action":
            continue
        fields = {part[i]: part[i + 1] for i in range(2, len(part) - 1, 2)}
        params = [p for p in fields.get(":parameters", []) if p.startswith("?")]
        pre, _ = _literals(fields.get(":precondition", []))
        add, delete = _literals(fields.get(":effect", []))
        found[part[1]] = (params, pre, add, delete)
    return found


def _init(problem: str) -> set[tuple]:
    tree = _parse(_tokens(problem))[0]
    for part in tree:
        if isinstance(part, list) and part and part[0] == ":init":
            return {tuple(f) for f in part[1:] if isinstance(f, list) and f}
    return set()


def progress(domain: str, state: set[tuple], steps: list[PDDLPlanStep]) -> set[tuple] | None:
    """The facts after ``steps``; None if a step is not applicable."""
    schemas = _actions(domain)
    state = set(state)
    for step in steps:
        schema = schemas.get(step.name.lower())
        if schema is None:
            return None
        params, pre, add, delete = schema
        if len(params) != len(step.parameters):
            return None
        bind = dict(zip(params, (p.lower() for p in step.parameters)))

        def ground(literal):
            return tuple(bind.get(t, t) for t in literal)

        if any(ground(p) not in state for p in pre):
            return None
        for d in delete:
            state.discard(ground(d))
        for a in add:
            state.add(ground(a))
    return state


def _with(problem: str, state: set[tuple], goal: str) -> str:
    head = problem[: problem.index("(:init")]
    facts = "".join(f"    ({' '.join(f)})\n" for f in sorted(state))
    return f"{head}(:init\n{facts}  )\n  (:goal\n    (and\n      {goal}\n    )\n  )\n)\n"


def staged_plan(problem: PDDLProblem, orders: int, solve) -> list[PDDLPlanStep] | None:
    """One plan per order, in queue order, chained; None if a stage fails.

    ``solve`` maps a PDDLProblem to its steps (empty or None: no plan).
    """
    state = _init(problem.problem)
    plan: list[PDDLPlanStep] = []
    for k in range(orders):
        if ("served", f"o{k}") in state:
            continue
        stage = PDDLProblem(domain=problem.domain,
                            problem=_with(problem.problem, state, f"(served o{k})"))
        steps = solve(stage)
        if not steps:
            return None
        state = progress(problem.domain, state, steps)
        if state is None:
            return None
        plan.extend(steps)
    return plan
