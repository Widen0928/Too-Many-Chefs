"""Analyse idle/blocked frame waste from L4_LOG output.

Runs h2h games on a levelset with L4_LOG=1, parses the per-chef LOG summary
lines, and breaks total chef-ticks into 6 waste categories:

  确实来不及  impossible  – idle right after a "none" plan result (no order fits)
  等待规划    planning    – waiting for the background planner
  被挡路      blocked_nav – blocked by another chef (navigation)
  重复选单    dup_plan    – planning ticks spent re-planning the same order
  错误让单    wrong_yield – planning ticks triggered by an incorrect YIELD
  等食物      food        – blocked/wait ticks while food is still cooking

Usage:
    uv run python tools/l4_waste.py SEEDS_DIR --base BASE_DIR [--jobs N]
        [--mode h2h|mixed] [KEY=VALUE ...]

KEY=VALUE are passed as env vars to l4_sim.py (e.g. L4_CPU_TICKS=1).
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UV = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
REDS, BLUES = ["Ada", "Bo"], ["Cyd", "Dev"]
ALL = REDS + BLUES


# ---------------------------------------------------------------------------
# Run one game, return its stdout
# ---------------------------------------------------------------------------

def _run(level: str, extra: list[str], env: dict) -> str:
    result = subprocess.run(
        [UV, "run", "python", "tools/l4_sim.py", "--ticks", "300",
         "--level", level, *extra],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    return result.stdout


# ---------------------------------------------------------------------------
# Parse one game's LOG output
# ---------------------------------------------------------------------------

def _parse(stdout: str) -> dict:
    """Return per-chef dicts with ticks_by_kind and event counts."""
    chefs: dict[str, dict] = {}

    def chef(cid):
        if cid not in chefs:
            chefs[cid] = {
                "ticks": {},        # from LOG 0 summary
                "why": defaultdict(int),    # why=X counts
                "plan_seq": [],     # list of (tick, dish) in order
                "pick_reasons": defaultdict(int),   # pick LOG reasons
                "plan_check": defaultdict(int),     # plan_check LOG results
                "yield_tick": [],   # ticks when yield triggered
                "dup_plan_ticks": 0,
                "wrong_yield_ticks": 0,
            }
        return chefs[cid]

    for line in stdout.splitlines():
        if not line.startswith("LOG "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        tick_s, cid, event = parts[1], parts[2], parts[3]

        if event == "0" or tick_s == "0":
            # LOG 0 chef ticks k1=v1 k2=v2 ...
            if parts[2] == "ticks" or (len(parts) > 3 and parts[3] == "ticks"):
                # format: LOG 0 chef ticks k=v ...
                cid2 = parts[2] if parts[2] != "ticks" else parts[1]
                # Actually: LOG 0 chef_id ticks k=v ...
                # parts[0]=LOG parts[1]=0 parts[2]=chef_id parts[3]=ticks parts[4..]=k=v
                if len(parts) >= 4 and parts[3] == "ticks":
                    c = chef(parts[2])
                    for kv in parts[4:]:
                        k, _, v = kv.partition("=")
                        if _:
                            c["ticks"][k] = int(v)
            continue

        try:
            tick = int(tick_s)
        except ValueError:
            continue

        c = chef(cid)

        if event == "why":
            # LOG tick chef why REASON len
            reason = parts[4] if len(parts) > 4 else "?"
            c["why"][reason] += 1

        elif event == "plan":
            # LOG tick chef plan DISH len
            dish = parts[4] if len(parts) > 4 else "?"
            c["plan_seq"].append((tick, dish))

        elif event == "pick":
            # LOG tick chef pick DISH eta=X deadline=Y reason
            reason = parts[7] if len(parts) > 7 else "?"
            c["pick_reasons"][reason] += 1

        elif event == "plan_check":
            # LOG tick chef plan_check DISH planned=X deadline=Y result
            result = parts[7] if len(parts) > 7 else "?"
            c["plan_check"][result] += 1

    # Post-process: detect dup plans and wrong yields from plan_seq + why events
    # We need the full event stream for this; rebuild from plan_seq.
    # For dup plans: consecutive plans for same dish where the previous ended
    # with why=slow/taken/patience/blocked (i.e. not done/gone/mate).
    # We approximate: re-scan stdout for interleaved why and plan events.
    for cid, c in chefs.items():
        prev_dish = None
        prev_why = "done"
        prev_plan_tick = 0
        prev_why_tick = 0

        for line in stdout.splitlines():
            if not line.startswith("LOG "):
                continue
            parts2 = line.split()
            if len(parts2) < 5:
                continue
            try:
                t = int(parts2[1])
            except ValueError:
                continue
            if parts2[2] != cid:
                continue
            ev = parts2[3]
            if ev == "why":
                prev_why = parts2[4] if len(parts2) > 4 else "?"
                prev_why_tick = t
            elif ev == "plan":
                dish = parts2[4] if len(parts2) > 4 else "?"
                if dish != "None":
                    if dish == prev_dish and prev_why not in ("done", "gone"):
                        # Replanned same dish after being stuck → dup plan
                        c["dup_plan_ticks"] += t - prev_why_tick
                    if prev_why == "mate":
                        # YIELD: check if we ended up on the same dish
                        if dish == prev_dish:
                            c["wrong_yield_ticks"] += t - prev_why_tick
                    prev_dish = dish
                    prev_plan_tick = t

    return chefs


# ---------------------------------------------------------------------------
# Aggregate across games
# ---------------------------------------------------------------------------

def _aggregate(all_chefs: list[dict]) -> dict[str, int]:
    total: dict[str, int] = defaultdict(int)
    for c in all_chefs:
        for k, v in c["ticks"].items():
            total[k] += v
        total["_dup_plan_ticks"] += c["dup_plan_ticks"]
        total["_wrong_yield_ticks"] += c["wrong_yield_ticks"]
    return total


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _report(totals: dict[str, int], games: int) -> None:
    walk = totals.get("walk", 0)
    act = totals.get("act", 0)
    planning = totals.get("planning", 0)
    idle = totals.get("idle", 0)
    wait = totals.get("wait", 0)
    blocked = totals.get("blocked", 0)
    food = totals.get("food", 0)
    impossible = totals.get("impossible", 0)
    yield_idle = totals.get("yield_idle", 0)
    yield_plan = totals.get("yield_plan", 0)
    dup_plan = totals["_dup_plan_ticks"]
    wrong_yield = totals["_wrong_yield_ticks"]

    total_ticks = walk + act + planning + idle + wait + blocked
    if total_ticks == 0:
        print("No LOG data found. Run with L4_LOG=1 and L4_SYNC=1.")
        return

    def pct(n):
        return f"{100 * n / total_ticks:5.1f}%"

    print(f"\n{'='*62}")
    print(f"  L4 waste breakdown  ({games} games, {total_ticks} chef-ticks total)")
    print(f"{'='*62}")
    print(f"  {'Category':<22}  {'Ticks':>7}  {'%':>6}  {'Notes'}")
    print(f"  {'-'*58}")

    # Productive
    print(f"  {'[productive] walk':<22}  {walk:>7}  {pct(walk)}")
    print(f"  {'[productive] act':<22}  {act:>7}  {pct(act)}")

    # The 6 waste categories
    print(f"  {'-'*58}")
    print(f"  {'确实来不及 impossible':<22}  {impossible:>7}  {pct(impossible)}  idle after none-plan")
    print(f"  {'等待规划   planning':<22}  {planning:>7}  {pct(planning)}  of which yield_plan={yield_plan}")
    print(f"  {'被挡路     blocked_nav':<22}  {blocked - food:>7}  {pct(blocked - food)}  nav-blocked (excl. food)")
    print(f"  {'重复选单   dup_plan':<22}  {dup_plan:>7}  {pct(dup_plan)}  replanning same dish (est)")
    print(f"  {'错误让单   wrong_yield':<22}  {wrong_yield:>7}  {pct(wrong_yield)}  yield → same dish back (est)")
    print(f"  {'等食物     food':<22}  {food:>7}  {pct(food)}  waiting for cooking")

    # Remainder idle
    other_idle = idle - impossible - yield_idle
    if other_idle > 0:
        print(f"  {'   idle (other)':<22}  {other_idle:>7}  {pct(other_idle)}")
    if yield_idle > 0:
        print(f"  {'   idle (yield)':<22}  {yield_idle:>7}  {pct(yield_idle)}")
    wait_other = wait - food  # wait ticks not attributed to food
    if wait_other > 0:
        print(f"  {'   wait (other)':<22}  {wait_other:>7}  {pct(wait_other)}")

    print(f"  {'='*58}")
    waste = total_ticks - walk - act
    print(f"  {'Total waste':<22}  {waste:>7}  {pct(waste)}")
    print(f"{'='*62}\n")

    # Summary ranking
    cats = [
        ("确实来不及 impossible", impossible),
        ("等待规划   planning",   planning - yield_plan),
        ("被挡路     blocked_nav", blocked - food),
        ("重复选单   dup_plan",   dup_plan),
        ("错误让单   wrong_yield", wrong_yield),
        ("等食物     food",       food),
    ]
    cats.sort(key=lambda x: -x[1])
    print("  Ranking (largest waste first):")
    for rank, (name, n) in enumerate(cats, 1):
        print(f"    {rank}. {name:<26} {n:>7} ticks  {pct(n)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    def option(name, default):
        if name in args:
            i = args.index(name)
            val = args[i + 1]
            del args[i: i + 2]
            return val
        return default

    jobs = int(option("--jobs", "6"))
    base = option("--base", "")
    mode = option("--mode", "h2h")
    if not args:
        print(__doc__)
        sys.exit(1)
    folder = args[0]
    settings = args[1:]
    env = dict(os.environ, L4_SYNC="1", L4_LOG="1")
    for s in settings:
        k, _, v = s.partition("=")
        env[k] = v

    levels = sorted(glob.glob(f"{folder}/seed*.yaml"))
    runs = []
    for level in levels:
        if mode == "h2h" and base:
            for ours, theirs in ((REDS, BLUES), (BLUES, REDS)):
                runs.append((level, ["--chefs", ",".join(ours),
                                     "--base", base, "--base-chefs", ",".join(theirs)]))
        elif mode == "mixed" and base:
            for me in ALL:
                others = ",".join(c for c in ALL if c != me)
                runs.append((level, ["--chefs", me, "--base", base,
                                     "--base-chefs", others]))
        else:
            for me in ALL:
                walkers = ",".join(c for c in ALL if c != me)
                runs.append((level, ["--chefs", me, "--walkers", walkers]))

    all_chefs: list[dict] = []
    games = 0

    def play(run):
        level, extra = run
        out = _run(level, extra, env)
        return _parse(out)

    print(f"Running {len(runs)} games across {len(levels)} seeds ({mode} mode)…", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for i, result in enumerate(pool.map(play, runs), 1):
            for c in result.values():
                if c["ticks"]:
                    all_chefs.append(c)
            games += 1
            if i % 10 == 0:
                print(f"  {i}/{len(runs)} done", flush=True)

    totals = _aggregate(all_chefs)
    _report(totals, games)


if __name__ == "__main__":
    main()
