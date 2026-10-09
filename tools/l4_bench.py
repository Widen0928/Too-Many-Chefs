"""Part 4 benchmark: our chef in each seat, on several order seeds, against
wandering examples and against three copies of our own controller.

Deterministic (L4_SYNC=1). Extra KEY=VALUE arguments are passed as
environment variables, to compare controller settings.

Usage: uv run python tools/l4_bench.py LEVEL_DIR [KEY=VALUE ...] [--jobs N]
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHEFS = ["Ada", "Bo", "Cyd", "Dev"]


def main() -> None:
    args = sys.argv[1:]
    jobs = 6
    if "--jobs" in args:
        i = args.index("--jobs")
        jobs = int(args[i + 1])
        del args[i : i + 2]
    folder, settings = args[0], args[1:]
    env = dict(os.environ, L4_SYNC="1")
    for setting in settings:
        key, value = setting.split("=", 1)
        env[key] = value
    runs = []
    for level in sorted(glob.glob(f"{folder}/seed*.yaml")):
        for me in CHEFS:
            others = ",".join(c for c in CHEFS if c != me)
            runs.append(("walkers", level, me, ["--chefs", me, "--walkers", others]))
            runs.append(("ours", level, me, ["--chefs", f"{me},{others}"]))

    def play(run):
        kind, level, me, extra = run
        out = subprocess.run(
            ["uv", "run", "python", "tools/l4_sim.py", "--ticks", "300", "--level", level,
             "--focus", me, *extra],
            cwd=ROOT, capture_output=True, text=True, env=env,
        ).stdout
        score = next(
            (int(line.split()[2]) for line in out.splitlines() if line.startswith("FOCUS")), 0
        )
        return kind, score

    with ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(play, runs))
    walk = sum(s for k, s in results if k == "walkers")
    ours = sum(s for k, s in results if k == "ours")
    print(f"vs walkers {walk}   vs ours {ours}   TOTAL {walk + ours}   ({len(results)} runs) {settings}")


if __name__ == "__main__":
    main()
