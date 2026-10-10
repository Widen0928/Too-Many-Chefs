"""Write randomised variants of a practice level, for testing robustness.

Shuffles the cells of the top and bottom rows (storages, stations, counters
keep their kinds, just not their places), puts the chefs on random floor
tiles, scales the time limits and sometimes adds orders.

Usage: uv run python tools/fuzz_family.py LEVEL OUT_DIR [--count N]
"""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("level")
    parser.add_argument("out")
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.level).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows0 = [r.split("|")[1:-1] for r in base["layout"].splitlines()]
    symbols = {str(a["symbol"]) for a in base["legend"]["agents"]}
    stem = Path(args.level).stem
    for seed in range(args.count):
        rng = random.Random(f"{stem}:{seed}")
        level = copy.deepcopy(base)
        rows = [list(r) for r in rows0]
        for y in (0, len(rows) - 1):
            rng.shuffle(rows[y])
        floor = []
        for y, row in enumerate(rows):
            for x, cell in enumerate(row):
                if cell.strip() in symbols:
                    row[x] = "   "
                if cell.strip() == "" or cell.strip() in symbols:
                    floor.append((y, x))
        floor = [(y, x) for y, x in floor if 0 < y < len(rows) - 1]
        for agent, (y, x) in zip(base["legend"]["agents"], rng.sample(floor, len(symbols))):
            rows[y][x] = f" {agent['symbol']} "
        level["layout"] = "\n".join("|" + "|".join(r) + "|" for r in rows)
        scale = rng.choice([1.1, 1.0, 0.9, 0.8])
        orders = level["state"]["orders"]
        for i, o in enumerate(orders):
            if isinstance(o, dict) and "time_limit" in o:
                o["time_limit"] = max(15, int(o["time_limit"] * scale))
        rules = level.setdefault("rules", {})
        if "default_order_time_limit" in rules:
            rules["default_order_time_limit"] = int(rules["default_order_time_limit"] * scale)
        if rng.random() < 0.3:
            extra = copy.deepcopy(rng.choice(orders))
            if isinstance(extra, dict) and "time_limit" in extra:
                extra["time_limit"] += 60
            orders.append(extra)
        level["name"] = f"{level.get('name', stem)} fuzz {seed}"
        (out / f"{stem}__{seed:02d}.yaml").write_text(yaml.safe_dump(level, sort_keys=False))
    print("wrote", args.count, "variants of", stem)


if __name__ == "__main__":
    main()
