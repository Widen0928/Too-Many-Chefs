"""Copy the PDDL model into the Part 4 submission folder.

The contest plays Part 4 from ``controllers/open/submission`` alone, mounted
into a fresh checkout, so the open controller cannot import our
``controllers/pddl/submission`` code there. This copies it into
``controllers/open/submission/planner`` (relative imports keep working).

Usage: uv run python tools/sync_open.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "controllers" / "pddl" / "submission"
TARGET = ROOT / "controllers" / "open" / "submission" / "planner"
FILES = [
    "kitchen.py",
    "dispatch.py",
    "from_pddl.py",
    "problem_generator.py",
    "overcooked.pddl",
    "config.py",
    "rollout.py",
    "strategy.py",
]


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "__init__.py").write_text(
        '"""A copy of controllers/pddl/submission (see tools/sync_open.py)."""\n'
    )
    for name in FILES:
        shutil.copy2(SOURCE / name, TARGET / name)
    print(f"copied {len(FILES)} files to {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
