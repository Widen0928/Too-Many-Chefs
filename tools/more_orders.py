"""Write variants of levels with more orders and a plate dispenser.

Usage: uv run python tools/more_orders.py OUT_DIR N [glob ...] [--stretch K]
Extra orders repeat the level's own, with deadlines spaced like the originals.
"""
from __future__ import annotations

import argparse
import copy
import glob
import re
from pathlib import Path

import yaml


def symbols(level: dict) -> set[str]:
    used = set()
    for entries in level.get("legend", {}).values():
        if isinstance(entries, list):
            used |= {str(e.get("symbol")) for e in entries if isinstance(e, dict)}
    return used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("n", type=int)
    parser.add_argument("patterns", nargs="*", default=["levels/[12]_*/*.yaml"])
    parser.add_argument("--stretch", type=float, default=1.0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "catalog").exists():
        (out / "catalog").symlink_to(Path("catalog").resolve())
    for path in sorted({p for pat in args.patterns for p in glob.glob(pat)}):
        level = yaml.safe_load(Path(path).read_text())
        e = copy.deepcopy(level)
        orders = [o if isinstance(o, dict) else {"food": o} for o in level["state"]["orders"]]
        limits = [o.get("time_limit") for o in orders]
        known = [l for l in limits if l]
        gap = (max(known) - min(known)) / max(1, len(known) - 1) if len(known) > 1 else 40
        gap = gap or 40
        new = []
        for i in range(args.n):
            o = dict(orders[i % len(orders)])
            if o.get("time_limit"):
                base = o["time_limit"] + (i // len(orders)) * gap * len(orders)
                o["time_limit"] = int(base * args.stretch)
            new.append(o)
        e["state"]["orders"] = new
        legend = e["legend"]
        if "plate_dispensers" in legend:
            for d in legend["plate_dispensers"]:
                d["plate_count"] = max(d.get("plate_count", 1), 2)
        else:
            sym = next(c for c in "DQZWJ" if c not in symbols(level))
            legend["plate_dispensers"] = [{"symbol": sym, "plate_count": 2}]
            e["layout"] = e["layout"].replace("-p", sym + " ")
        name = f"{Path(path).parent.name[:1]}_{Path(path).stem}__n{args.n}"
        (out / f"{name}.yaml").write_text(yaml.safe_dump(e, sort_keys=False))


if __name__ == "__main__":
    main()
