"""Reuse plans: a library of dish plans, checked against the kitchen as it is.

Calling Fast Downward costs about a third of a second, three ticks of a
match, every time the chef starts a dish. But the PDDL problems name food
by kind (``f_meat``), stations and tiles by place (``s_13_4_pan``,
``loc_6_0``) and the order as ``o0``, so a plan made for a dish once fits
the next order for that dish too, once its chef is renamed and its
stations moved to free ones (see routing.relocate_stations).

Whether a reused plan still works is checked exactly: the domain is plain
STRIPS, so ``valid`` applies the plan's actions to the problem's initial
facts and tests the goal, as a plan validator would. A plan that fails the
check is never used; the chef calls the planner instead and the new plan
joins the library.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from controllers.pddl.interfaces import PDDLPlanStep

DOMAIN = Path(__file__).parent / "planner" / "overcooked.pddl"
SAVED = Path(__file__).parent / "library.json"
CHEF = "?chef"


# --- A small STRIPS reader and validator ------------------------------------------


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
    """Positive and negative literals of a conjunction."""
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


@lru_cache(maxsize=1)
def actions() -> dict[str, tuple]:
    """name -> (parameter names, preconditions, adds, deletes)."""
    domain = _parse(_tokens(DOMAIN.read_text()))[0]
    found = {}
    for part in domain:
        if not isinstance(part, list) or not part or part[0] != ":action":
            continue
        name = part[1]
        fields = {part[i]: part[i + 1] for i in range(2, len(part) - 1, 2)}
        params = [p for p in fields.get(":parameters", []) if p.startswith("?")]
        pre, _ = _literals(fields.get(":precondition", []))
        add, delete = _literals(fields.get(":effect", []))
        found[name] = (params, pre, add, delete)
    return found


def facts(problem: str) -> tuple[frozenset, list[tuple]]:
    """The initial facts and the goal of a problem."""
    tree = _parse(_tokens(problem))[0]
    init: set = set()
    goal: list[tuple] = []
    for part in tree:
        if isinstance(part, list) and part and part[0] == ":init":
            init = {tuple(f) for f in part[1:] if isinstance(f, list) and f and f[0] != "="}
        elif isinstance(part, list) and part and part[0] == ":goal":
            goal, _ = _literals(part[1])
    return frozenset(init), goal


def valid(steps: list[PDDLPlanStep], init: frozenset, goal: list[tuple]) -> bool:
    """Whether ``steps`` can be carried out from ``init`` and reach ``goal``."""
    schemas = actions()
    state = set(init)
    for step in steps:
        schema = schemas.get(step.name.lower())
        if schema is None:
            return False
        params, pre, add, delete = schema
        if len(params) != len(step.parameters):
            return False
        bind = dict(zip(params, (p.lower() for p in step.parameters)))

        def ground(literal):
            return tuple(bind.get(t, t) for t in literal)

        if any(ground(p) not in state for p in pre):
            return False
        for d in delete:
            state.discard(ground(d))
        for a in add:
            state.add(ground(a))
    return all(tuple(g) in state for g in goal)


# --- The library -----------------------------------------------------------------

# dish -> plans, each a tuple of (action, parameters) with the chef as CHEF.
_LIBRARY: dict[str, list[tuple]] = {}
_LOADED = False


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    try:
        saved = json.loads(SAVED.read_text())
    except Exception:  # noqa: BLE001 - no saved library is fine
        return
    for dish, plans in saved.items():
        for plan in plans:
            remember(dish, [PDDLPlanStep(s[0], tuple(s[1:])) for s in plan], CHEF)


def _general(steps: list[PDDLPlanStep], chef: str) -> tuple:
    return tuple(
        (s.name, tuple(CHEF if p == chef else p for p in s.parameters)) for s in steps
    )


def remember(dish: str, steps: list[PDDLPlanStep], chef: str) -> None:
    """Add a plan that served ``dish`` (made for ``chef``) to the library."""
    plan = _general(steps, chef)
    plans = _LIBRARY.setdefault(dish, [])
    if plan not in plans:
        plans.append(plan)
        del plans[:-12]  # keep the newest dozen


def plans_for(dish: str, chef: str) -> list[list[PDDLPlanStep]]:
    """Every plan in the library for ``dish``, for ``chef``."""
    _load()
    return [
        [PDDLPlanStep(name, tuple(chef if p == CHEF else p for p in params))
         for name, params in plan]
        for plan in _LIBRARY.get(dish, [])
    ]


def export() -> dict:
    return {
        dish: [[name, *params] for name, params in plans]
        for dish, plans in _LIBRARY.items()
    }
