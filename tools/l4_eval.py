"""Part 4 evaluation over many order seeds (deterministic, L4_SYNC=1).

- solo: our chef in each seat with three wandering example chefs.
- h2h: two of our chefs against two of the --base submission, both ways round.
- mixed: one of ours with a --base teammate against two --base chefs.
- self: four of our chefs (total points, which shows how they share the kitchen).

Usage: uv run python tools/l4_eval.py SEEDS_DIR --base DIR [KEY=VALUE ...]
       [--jobs N] [--modes solo,h2h,mixed,self]
"""

from __future__ import annotations

import ast
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REDS, BLUES = ["Ada", "Bo"], ["Cyd", "Dev"]
ALL = REDS + BLUES


def main() -> None:
    args = sys.argv[1:]

    def option(name, default):
        if name in args:
            i = args.index(name)
            value = args[i + 1]
            del args[i : i + 2]
            return value
        return default

    jobs = int(option("--jobs", "8"))
    base = option("--base", "")
    modes = option("--modes", "solo,h2h,mixed,self").split(",")
    folder, settings = args[0], args[1:]
    env = dict(os.environ, L4_SYNC="1")
    for setting in settings:
        key, value = setting.split("=", 1)
        env[key] = value
    runs = []
    for level in sorted(glob.glob(f"{folder}/seed*.yaml")):
        if "solo" in modes:
            for me in ALL:
                walkers = ",".join(c for c in ALL if c != me)
                runs.append(("solo", level, [me], ["--chefs", me, "--walkers", walkers]))
        if "self" in modes:
            runs.append(("self", level, ALL, ["--chefs", ",".join(ALL)]))
        if base and "h2h" in modes:
            for ours, theirs in ((REDS, BLUES), (BLUES, REDS)):
                runs.append(("h2h", level, ours, ["--chefs", ",".join(ours),
                             "--base", base, "--base-chefs", ",".join(theirs)]))
        if base and "mixed" in modes:
            for me in ALL:
                others = ",".join(c for c in ALL if c != me)
                runs.append(("mixed", level, [me], ["--chefs", me, "--base", base,
                             "--base-chefs", others]))

    def play(run):
        kind, level, ours, extra = run
        out = subprocess.run(
            ["uv", "run", "python", "tools/l4_sim.py", "--ticks", "300", "--level", level, *extra],
            cwd=ROOT, capture_output=True, text=True, env=env,
        ).stdout
        line = next((l for l in out.splitlines() if l.startswith("score ")), "")
        if not line:
            return kind, 0, 0, 0
        scores = ast.literal_eval(line.split("by agent ")[1].split(" delivered")[0])
        team = REDS if ours[0] in REDS else BLUES
        mine = sum(scores.get(c, 0) for c in ours)
        ours_team = sum(scores.get(c, 0) for c in team)
        theirs = sum(scores.get(c, 0) for c in ALL if c not in team)
        return kind, mine, ours_team, theirs

    with ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(play, runs))
    parts = []
    for kind in ("solo", "self", "h2h", "mixed"):
        rows = [r for r in results if r[0] == kind]
        if not rows:
            continue
        mine = sum(r[1] for r in rows)
        if kind in ("h2h", "mixed"):
            diff = sum(r[2] - r[3] for r in rows)
            wins = sum(r[2] > r[3] for r in rows)
            parts.append(f"{kind}: ours {mine} margin {diff:+d} wins {wins}/{len(rows)}")
        else:
            parts.append(f"{kind}: {mine}")
    print("   ".join(parts), settings)


if __name__ == "__main__":
    main()
