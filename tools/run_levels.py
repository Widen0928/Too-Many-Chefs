"""Run the PDDL controller headless on many levels in parallel and summarise.

Usage: uv run python tools/run_levels.py [glob ...] [--jobs N] [--budget S]
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(level: str, budget: float | None, max_steps: int) -> dict:
    command = [
        "uv", "run", "cook", "agent", "--level", level, "--headless", "--quiet",
        "--max-timesteps", str(max_steps), "--time-limit", "300",
    ]
    if budget is not None:
        command += ["--planning-budget", str(budget)]
    out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    text = out.stdout
    start = text.find("{")
    try:
        return json.loads(text[start:])
    except Exception:
        return {"reason": "crash", "error": {"message": (out.stdout + out.stderr)[-800:]}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="*", default=["levels/[012]_*/*.yaml"])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()
    levels = sorted({p for pattern in args.patterns for p in glob.glob(pattern)})
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda l: run(l, args.budget, args.max_steps), levels))
    total = 0
    for level, result in zip(levels, results):
        err = result.get("error")
        msg = ""
        if err:
            msg = (err.get("message") or "").strip().splitlines()[-1:] if err.get("message") else ""
            msg = msg[0][:120] if msg else ""
        print(
            f"{Path(level).parent.name[:2]}/{Path(level).stem:55s} "
            f"score={result.get('score', 0):4} delivered={result.get('orders_delivered', 0)}"
            f" left={result.get('orders_remaining', '?')} t={result.get('timesteps', '?'):>4}"
            f" e={result.get('elapsed_seconds', 0) or 0:5.1f}s {result.get('reason')} {msg}"
        )
        total += result.get("score", 0) or 0
    print("TOTAL", total)


if __name__ == "__main__":
    sys.exit(main())
