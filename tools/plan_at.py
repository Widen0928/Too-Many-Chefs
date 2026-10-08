"""Re-plan from a recorded timestep and print the dispatched jobs.

Usage: uv run python tools/plan_at.py RECORDING TIMESTEP
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.pddl.solvers import solve_problem_with_logging  # noqa: E402
from controllers.pddl.submission import dispatch as d  # noqa: E402
from controllers.pddl.submission.kitchen import Kitchen  # noqa: E402
from controllers.pddl.submission.problem_generator import DefaultProblemGenerator  # noqa: E402
from simulator.environment import Environment  # noqa: E402

docs = list(yaml.safe_load_all(open(sys.argv[1])))
doc = next(x for x in docs if x["environment"]["timestep"] == int(sys.argv[2]))
env = Environment.model_validate(doc["environment"])
problem = DefaultProblemGenerator().to_pddl(env)
steps = solve_problem_with_logging(problem)
d.DEBUG = True
d._Dispatcher(Kitchen(env), steps).run()
