"""Write the domain and problem for a level's initial state, and optionally solve.

Usage: uv run python tools/dump_problem.py LEVEL OUT_DIR [--solve] [--alias A]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from controllers.pddl.submission.problem_generator import DefaultProblemGenerator  # noqa: E402
from simulator.configuration.load import load, load_configuration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("level")
    parser.add_argument("out")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--alias", default="lama-first")
    parser.add_argument("--limit", default="30s")
    args = parser.parse_args()
    env = load(load_configuration(args.level))
    problem = DefaultProblemGenerator().to_pddl(env)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "domain.pddl").write_text(problem.domain)
    (out / "problem.pddl").write_text(problem.problem)
    if not args.solve:
        return
    import importlib.util

    spec = importlib.util.find_spec("up_fast_downward")
    driver = Path(spec.origin).parent / "downward" / "fast-downward.py"
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(driver), "--plan-file", str(out / "sas_plan"),
         "--overall-time-limit", args.limit, "--alias", args.alias,
         str(out / "domain.pddl"), str(out / "problem.pddl")],
        capture_output=True, text=True,
    )
    lines = [l for l in result.stdout.splitlines() if any(
        k in l for k in ("Plan length", "Expanded", "Search time", "Solution", "Translator variables", "Translator operators", "Completely", "Time limit"))]
    print("\n".join(lines[-8:]), f"\nwall {time.time() - start:.1f}s exit {result.returncode}")


if __name__ == "__main__":
    main()
