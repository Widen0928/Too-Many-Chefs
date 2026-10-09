"""Play Part 4 matches and print a compact summary.

Usage: uv run python tools/l4_match.py SEAT SEAT SEAT SEAT [--sets N]
A seat is '.', 'example', or a folder with controller.py (e.g. a saved copy).
"""
from __future__ import annotations

import json
import subprocess
import sys

args = sys.argv[1:]
sets = 1
if "--sets" in args:
    i = args.index("--sets")
    sets = int(args[i + 1])
    del args[i : i + 2]
seats = args
totals = {"a": 0, "b": 0}
wins = {"a": 0, "b": 0, "draw": 0}
for i in range(sets):
    cmd = ["uv", "run", "python", "main.py", "match", "--quiet"]
    for s in seats:
        cmd += ["--seat", s]
    run = subprocess.run(cmd, capture_output=True, text=True)
    out = run.stdout + run.stderr
    out = run.stdout if run.stdout.lstrip().startswith("{") else out
    start = out.find("{")
    d, _ = json.JSONDecoder().raw_decode(out[start:])
    for k in totals:
        totals[k] += d["score"][k]
    wins[d["winner"]] += 1
    games = [{k: v for k, v in g.items() if k not in ("replay",)} for g in d["games"]]
    seat_scores = [g.get("score_by_seat") for g in games]
    errs = [(s["name"], s["missed_ticks"], s["error"] and s["error"].get("message", "")[:80]) for s in d["seats"]]
    print(f"set {i}: score {d['score']} winner {d['winner']} seats {seat_scores} missed/errors {errs}")
print("TOTAL", totals, wins)
