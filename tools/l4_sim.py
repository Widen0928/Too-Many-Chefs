"""Drive the Part 4 controller in one process on the tournament kitchen.

Other chefs stand still unless --example is given (then they wander like the
example controller). Prints score and per-tick debug.

Usage: uv run python tools/l4_sim.py [--ticks N] [--chef Ada] [--debug]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.open.submission import controller as ctl  # noqa: E402
from simulator.configuration.load import load, load_configuration  # noqa: E402
from simulator.context import Context  # noqa: E402
from simulator.entities import Agent, OvercookedState  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=300)
    p.add_argument("--chefs", default="Ada")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--events", action="store_true")
    p.add_argument("--walkers", default="", help="chefs driven by the example controller")
    p.add_argument("--focus", default="", help="report only this chef's score")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10**9)
    p.add_argument("--level", default="levels/3_too_many_chefs/0_full_menu.yaml")
    p.add_argument("--base", default="", help="another submission folder to compare against")
    p.add_argument("--base-chefs", default="", help="chefs driven by the --base controller")
    a = p.parse_args()
    env = load(load_configuration(a.level))
    controllers = {}
    for name in a.chefs.split(","):
        c = ctl.OpenController()
        c.set_controlled_agents({name})
        controllers[name] = c
    from controllers.open.example import ExampleController
    if a.base:
        base = _load_base(a.base)
        for name in [b for b in a.base_chefs.split(",") if b]:
            c = base.OpenController()
            c.set_controlled_agents({name})
            controllers[name] = c
    walkers = {}
    for name in [w for w in a.walkers.split(",") if w]:
        w = ExampleController()
        w.set_controlled_agents({name})
        walkers[name] = w
    orig = ctl.OpenController._act
    def act(self, environment):
        try:
            return orig(self, environment)
        except Exception:
            traceback.print_exc()
            raise
    ctl.OpenController._act = act
    slow = 0
    for t in range(a.ticks):
        by_chef: dict[str, list] = {}
        for name, c in controllers.items():
            start = time.monotonic()
            acts = c.get_actions(env, Context())
            dt = time.monotonic() - start
            slow = max(slow, dt)
            by_chef[name] = list(acts)
            if a.debug and a.start <= t <= a.end:
                ag = env.get_entity_as(name, Agent) if env.get_entity(name) else None
                me = [x for x in env.get_entities_of_type(Agent) if x.id == name][0]
                nxt = c._plan[0] if c._plan else None
                print(t, (me.x, me.y), [type(x).__name__ for x in acts], len(c._plan), nxt and (type(nxt.mutation).__name__, nxt.location))
            elif a.events:
                key = (c._target, bool(c._plan), c._future is not None)
                if key != getattr(c, "_dbg", None):
                    c._dbg = key
                    st = env.get_first_entity_of_type(OvercookedState)
                    print(t, name, "target", c._target and c._target.split('/')[-1], "plan", len(c._plan), "planning", c._future is not None, "score", st.score, "orders", [(e.name.split('/')[-1], e.revealed_at + (e.time_limit or 0)) for e in st.order_queue.visible])
        for wname, w in walkers.items():
            by_chef[wname] = list(w.get_actions(env, Context()))
        # The host's order: the sides take turns to go first.
        from simulator.match.host import _turn_order
        sides = [[c for c in ("Ada", "Bo") if c in by_chef], [c for c in ("Cyd", "Dev") if c in by_chef]]
        sides = [side for side in sides if side]
        actions = [x for chef in _turn_order(sides, env.timestep) for x in by_chef[chef]]
        env = env.step(actions)
        pass
    state = env.get_first_entity_of_type(OvercookedState)
    if a.focus:
        print("FOCUS", a.focus, state.score_by_agent.get(a.focus, 0))
    print("score", state.score, "by agent", state.score_by_agent, "delivered", state.order_queue.delivered_count, f"slowest tick {slow*1000:.0f} ms")
    for c in controllers.values():
        c.shutdown()


def _load_base(folder: str):
    """Import another copy of the submission under its own package name."""
    import importlib
    import importlib.util

    name = "base_submission"
    spec = importlib.util.spec_from_file_location(
        name, Path(folder) / "__init__.py", submodule_search_locations=[folder]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(name + ".controller")


if __name__ == "__main__":
    main()
