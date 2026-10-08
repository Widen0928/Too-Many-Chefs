"""Write variants of the practice levels to stress-test the controller.

Variants: mirrored layouts, more orders, and chefs moved to other floor tiles.

Usage: uv run python tools/fuzz_levels.py OUT_DIR [glob ...] [--seeds N]
"""

from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import yaml


def cells(layout: str) -> list[list[str]]:
    rows = []
    for line in layout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")[1:-1]
        rows.append(parts)
    return rows


def render(rows: list[list[str]]) -> str:
    width = max(len(c) for row in rows for c in row)
    return "\n".join(
        "|" + "|".join(c.ljust(width) for c in row) + "|" for row in rows
    )


def agent_symbols(level: dict) -> set[str]:
    return {str(a["symbol"]) for a in level.get("legend", {}).get("agents", [])}


def move_agents(rows: list[list[str]], symbols: set[str], rng: random.Random) -> None:
    empty = [(y, x) for y, row in enumerate(rows) for x, c in enumerate(row) if not c.strip()]
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c.strip() in symbols and empty:
                ny, nx = empty.pop(rng.randrange(len(empty)))
                rows[ny][nx] = c
                rows[y][x] = " " * len(c)


def variants(level: dict, seeds: int):
    base = cells(level["layout"])
    yield "hflip", [list(reversed(r)) for r in base], None
    yield "vflip", list(reversed([list(r) for r in base])), None
    yield "rot", [list(reversed(r)) for r in reversed(base)], None
    yield "more_orders", [list(r) for r in base], "more"
    for seed in range(seeds):
        rows = [list(r) for r in base]
        move_agents(rows, agent_symbols(level), random.Random(seed))
        yield f"agents{seed}", rows, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("patterns", nargs="*", default=["levels/[012]_*/*.yaml"])
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    catalog = out / "catalog"
    if not catalog.exists():
        catalog.symlink_to(Path("catalog").resolve())
    for path in sorted({p for pat in args.patterns for p in glob.glob(pat)}):
        level = yaml.safe_load(Path(path).read_text())
        for name, rows, extra in variants(level, args.seeds):
            new = dict(level)
            new["layout"] = render(rows)
            if extra == "more":
                state = dict(new["state"])
                orders = list(state["orders"])
                state["orders"] = orders + orders[: max(1, len(orders) // 2)]
                new["state"] = state
            target = out / f"{Path(path).parent.name[:1]}_{Path(path).stem}__{name}.yaml"
            target.write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
