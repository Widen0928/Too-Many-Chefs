"""Run levels with the deadline alternatives on and off, and report, per
level, the final score both ways and how often each alternative was
generated, played out and chosen (from the [tmc] lines on stderr).

Usage: uv run python tools/alt_audit.py OUT.json LEVEL... [--jobs N]
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(level: str, on: bool) -> dict:
    on_settings = json.loads(os.environ.get("AUDIT_ON", '{"deadline_alternatives": true}'))
    settings = on_settings if on else {"deadline_alternatives": False}
    env = dict(os.environ, TMPDIR=tempfile.mkdtemp(prefix="audit-"),
               TMC_EXPERIMENT=json.dumps(settings))
    out = subprocess.run(
        ["uv", "run", "cook", "agent", "--level", level, "--headless", "--quiet",
         "--max-timesteps", "600", "--time-limit", "300"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    text = out.stdout
    try:
        result = json.loads(text[text.find("{"):])
    except Exception:  # noqa: BLE001
        result = {}
    lines = [l for l in out.stderr.splitlines() if l.startswith("[tmc]")]
    # Only the real run counts, not the play-outs before it: the real run
    # comes after the 'strategy chosen' line.
    after = lines[next((i for i, l in enumerate(lines) if "strategy chosen" in l), -1) + 1:]
    stats = {}
    for kind in ("leave-out", "swap"):
        stats[kind] = {
            "generated": sum(f"{kind}:" in l for l in after),
            "chosen": sum(f"{kind}:" in l and "-> chosen" in l for l in after),
            "chosen_in_playouts": sum(f"{kind}:" in l and "-> chosen" in l for l in lines),
        }
    return {"score": result.get("score"), "delivered": result.get("orders_delivered"), **stats}


def main() -> None:
    args = sys.argv[1:]
    jobs = 6
    if "--jobs" in args:
        i = args.index("--jobs")
        jobs = int(args[i + 1])
        del args[i : i + 2]
    out, levels = args[0], args[1:]
    tasks = [(lv, on) for lv in levels for on in (False, True)]
    with ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(lambda t: run(*t), tasks))
    report = {}
    for (lv, on), r in zip(tasks, results):
        report.setdefault(lv, {})["on" if on else "off"] = r
    Path(out).write_text(json.dumps(report, indent=1))
    better = [lv for lv, r in report.items() if (r["on"]["score"] or 0) > (r["off"]["score"] or 0)]
    worse = [lv for lv, r in report.items() if (r["on"]["score"] or 0) < (r["off"]["score"] or 0)]
    chosen = sum(r["on"][k]["chosen"] for r in report.values() for k in ("leave-out", "swap"))
    print(f"levels {len(report)}  better {len(better)}  worse {len(worse)}  "
          f"alternatives chosen in real runs {chosen}")
    for lv in better + worse:
        r = report[lv]
        print(Path(lv).stem, "off", r["off"]["score"], "on", r["on"]["score"],
              {k: r["on"][k] for k in ("leave-out", "swap")})


if __name__ == "__main__":
    main()
