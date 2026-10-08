"""Benchmark: run levels, compare each score with its estimated maximum.

Usage: uv run python tools/bench.py GLOB [--jobs N] [--exclude SUBSTR ...]
Prints per-level score/max, wall time, and the overall ratio.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from max_tips import best_route  # noqa: E402
from simulator.configuration.load import load, load_configuration  # noqa: E402
from simulator.entities import Food, OvercookedState, Storage  # noqa: E402


def max_score(path: str) -> int:
    env = load(load_configuration(path))
    state = env.get_first_entity_of_type(OvercookedState)
    storages = {s.food_name for s in env.get_entities_of_type(Storage)}
    loose = {f.name for f in env.get_entities_of_type(Food)}
    total = 0
    for entry in [*state.order_queue.visible, *state.order_queue.pending]:
        route = best_route(state, storages, loose, entry.name)
        total += entry.reward + (len(route) if route is not None else 0)
    return total


def run(level: str) -> tuple[dict, float]:
    start = time.time()
    out = subprocess.run(
        ["uv", "run", "cook", "agent", "--level", level, "--headless", "--quiet",
         "--max-timesteps", "1500", "--time-limit", "300"],
        cwd=ROOT, capture_output=True, text=True,
    )
    text = out.stdout
    try:
        result = json.loads(text[text.find("{"):])
    except Exception:
        result = {"score": 0, "reason": "crash"}
    return result, time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--exclude", nargs="*", default=["mid_service", "out_of_stock"])
    args = parser.parse_args()
    levels = sorted(
        p for p in glob.glob(args.pattern) if not any(x in p for x in args.exclude)
    )
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run, levels))
    got = best = 0
    for level, (result, seconds) in zip(levels, results):
        m = max_score(level)
        s = result.get("score", 0) or 0
        got += s
        best += m
        err = (result.get("error") or {}).get("type", "") if result.get("error") else ""
        print(f"{Path(level).stem[:60]:60s} {s:4d}/{m:4d} d={result.get('orders_delivered')}"
              f" t={result.get('timesteps')} {seconds:5.1f}s {result.get('reason')} {err}")
    print(f"TOTAL {got}/{best} = {got / max(best, 1):.4f}")


if __name__ == "__main__":
    main()
