"""Which dishes our chefs start, serve and give up on (tools/l4_sim.py runs
with L4_LOG=1). Usage: uv run python tools/l4_orders.py SEEDS_DIR --base DIR"""
from __future__ import annotations

import collections
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

args = sys.argv[1:]
folder = args[0]
base = args[args.index("--base") + 1]
env = dict(os.environ, L4_SYNC="1", L4_CPU_TICKS="1", L4_LOG="1")
for setting in args[2:]:
    if "=" in setting and not setting.startswith("--"):
        key, value = setting.split("=", 1)
        env[key] = value


def play(level):
    out = subprocess.run(
        ["uv", "run", "python", "tools/l4_sim.py", "--ticks", "300", "--level", level,
         "--chefs", "Ada,Bo", "--base", base, "--base-chefs", "Cyd,Dev"],
        capture_output=True, text=True, env=env,
    ).stdout
    return [line.split()[1:] for line in out.splitlines() if line.startswith("LOG ")]


started = collections.Counter()
served = collections.Counter()
dropped = collections.Counter()
ticks_served = collections.defaultdict(list)
ticks_dropped = collections.defaultdict(list)
idle = 0
replans = collections.Counter()
tick_kinds = collections.Counter()
reasons = collections.Counter()
with ThreadPoolExecutor(8) as pool:
    for events in pool.map(play, sorted(glob.glob(f"{folder}/seed*.yaml"))):
        current = {}
        last_why = {}
        for t, chef, kind, *rest in events:
            t = int(t)
            if chef not in ("Ada", "Bo"):
                continue
            if kind == "plan":
                dish = rest[0]
                if chef in current and current[chef][0] == dish:
                    replans[dish] += 1  # planned again for the same dish
                    continue
                if chef in current:
                    old, since = current.pop(chef)
                    reasons[last_why.get(chef, "?")] += 1
                    dropped[old] += 1
                    ticks_dropped[old].append(t - since)
                if dish == "None":
                    idle += 1
                    continue
                started[dish] += 1
                current[chef] = (dish, t)
            elif kind == "ticks" and chef in ("Ada", "Bo"):
                for part in rest:
                    k, v = part.split("=")
                    tick_kinds[k] += int(v)
            elif kind == "why":
                last_why[chef] = rest[0]
            elif kind == "deliver" and chef in current:
                dish, since = current.pop(chef)
                served[dish] += 1
                ticks_served[dish].append(t - since)
print(f"{'dish':20s} start serve drop  avg-serve-ticks  ticks-lost-dropping  replans")
for dish in sorted(started, key=lambda d: -started[d]):
    s = ticks_served[dish]
    print(f"{dish:20s} {started[dish]:5d} {served[dish]:5d} {dropped[dish]:4d}  "
          f"{sum(s) / max(1, len(s)):8.1f}  {sum(ticks_dropped[dish]):8d}  {replans[dish]:8d}")
print("empty plans", idle, "drop reasons", dict(reasons))
print("ticks", dict(tick_kinds))
