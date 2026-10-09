"""Where a Part 4 chef's ticks go: walking, acting, waiting, idle.

Runs like tools/l4_sim.py (same options) and classifies every tick of each
of our chefs by what it sent. Also counts plans abandoned before delivery.
"""
from __future__ import annotations

import collections
import subprocess
import sys

out = subprocess.run(
    ["uv", "run", "python", "tools/l4_sim.py", "--debug", *sys.argv[1:]],
    capture_output=True, text=True,
).stdout
counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
last_plan: dict[str, int] = {}
for line in out.splitlines():
    if not line[:1].isdigit() or "(" not in line:
        continue
    head, _, rest = line.partition(" [")
    parts = line.split()
    chef = "?"
    acts = rest.split("]")[0]
    kinds = [a.strip(" '") for a in acts.split(",") if a.strip()]
    plan_len = int(rest.split("]")[1].split()[0])
    if "MoveAgent" in kinds:
        kind = "walk"
    elif any(k not in ("TurnAgent",) for k in kinds):
        kind = "act"
    elif plan_len:
        kind = "wait"
    else:
        kind = "idle"
    counts["all"][kind] += 1
print(dict(counts["all"]))
print([l for l in out.splitlines() if l.startswith("score")][-1])
