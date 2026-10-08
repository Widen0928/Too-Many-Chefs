"""Plan a level's initial state and show the dispatched plan per chef.

Usage: uv run python tools/plan_level.py LEVEL
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.pddl.solvers import solve_problem_with_logging  # noqa: E402
from controllers.pddl.submission import dispatch as d  # noqa: E402
from controllers.pddl.submission.kitchen import Kitchen  # noqa: E402
from controllers.pddl.submission.problem_generator import DefaultProblemGenerator  # noqa: E402
from simulator.configuration.load import load, load_configuration  # noqa: E402

env = load(load_configuration(sys.argv[1]))
problem = DefaultProblemGenerator().to_pddl(env)
steps = solve_problem_with_logging(problem)
kitchen = Kitchen(env)
d.DEBUG = True
new = d._Dispatcher(kitchen, steps).run()  # raises instead of falling back
