"""Whole-level trials must preserve the controller's deferred replans."""

from types import SimpleNamespace

from controllers.pddl.submission.strategy import play_out
from simulator.entities import Agent, Equipment


def test_replan_requested_while_carrying_station_runs_after_placing_it(monkeypatch) -> None:
    planned = []

    class FakeEnvironment:
        def __init__(self, timestep=0):
            self.timestep = timestep

        def get_first_entity_of_type(self, cls):
            return SimpleNamespace(
                score=0, visible_orders=["dish"],
                order_queue=SimpleNamespace(pending=[]),
            )

        def get_entities_of_type(self, cls):
            return [SimpleNamespace(
                id="chef", x=0, y=0,
                held_item_id="pan" if self.timestep == 1 else None,
            )] if cls is Agent else []

        def get_entity(self, entity_id):
            return Equipment(id="pan", name="pan") if entity_id == "pan" else None

        def has_pending_effects(self):
            return False

        def step(self, actions):
            return FakeEnvironment(self.timestep + 1)

    class FakeGenerator:
        def should_replan(self, env):
            # Like an expired-order event: _reset consumes the trigger, so it
            # is no longer returned after the station has been put down.
            return env.timestep == 1

        def to_pddl(self, env):
            planned.append(env.timestep)

    class FakeTranslator:
        def __init__(self, env):
            pass

        def from_pddl(self, steps):
            return []

    class FakePolicy:
        def get_mutations(self, plans, env, context):
            if env.timestep < 3:
                return [object()], plans
            return [], {"chef": []}

    monkeypatch.setattr("controllers.navigation.DefaultMutationInterpolationPolicy", FakePolicy)
    monkeypatch.setattr("controllers.pddl.submission.problem_generator.DefaultProblemGenerator", FakeGenerator)
    monkeypatch.setattr("controllers.pddl.submission.from_pddl.DefaultFromPDDL", FakeTranslator)
    monkeypatch.setattr("controllers.pddl.solvers.solve_problem_with_logging", lambda _: [])
    monkeypatch.setattr("controllers.pddl.controller.group_mutations_by_agent", lambda **_: {"chef": [object()]})

    play_out(FakeEnvironment(), 3)

    # Match PDDLController._replan_due: defer at t=1, then replan at t=2.
    assert planned == [0, 2]
