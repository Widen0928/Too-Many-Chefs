"""Both planner copies support templates with and without legacy deadlocks."""

from importlib import import_module

import pytest

from controllers import navigation
from controllers.navigation import traffic
from simulator.entities import Agent, OvercookedState
from simulator.entities.orders import OrderEntry, OrderQueue
from simulator.environment import Environment


class LegacyDeadlockError(RuntimeError):
    pass


@pytest.fixture(params=[
    ("controllers.pddl.submission", "simulate"),
    ("controllers.pddl.submission", "play_out"),
    ("controllers.open.submission.planner", "simulate"),
    ("controllers.open.submission.planner", "play_out"),
])
def trial(request, monkeypatch):
    package, operation = request.param
    rollout = import_module(f"{package}.rollout")
    environment = Environment().with_entity(Agent(id="chef", x=0, y=0)).with_entity(
        OvercookedState(order_queue=OrderQueue(visible=[OrderEntry(name="soup")])),
    )
    if operation == "simulate":
        run = lambda: rollout.simulate(environment, [])
    else:
        generator = import_module(f"{package}.problem_generator")
        translator = import_module(f"{package}.from_pddl")
        monkeypatch.setattr(generator.DefaultProblemGenerator, "to_pddl", lambda *_: None)
        monkeypatch.setattr(translator.DefaultFromPDDL, "from_pddl", lambda *_: [])
        monkeypatch.setattr("controllers.pddl.solvers.solve_problem_with_logging", lambda _: [])
        strategy = import_module(f"{package}.strategy")
        run = lambda: strategy.play_out(environment, 3)
    return rollout, operation, run


def policy_returning_or_raising(monkeypatch, error=None):
    class Policy:
        def get_mutations(self, plans, environment, context):
            if error is not None:
                raise error
            return [], plans

    monkeypatch.setattr(navigation, "DefaultMutationInterpolationPolicy", Policy)


def test_new_template_runs_without_removed_deadlock_class(trial, monkeypatch):
    rollout, operation, run = trial
    monkeypatch.delattr(traffic, "AgentDeadlockError", raising=False)
    policy_returning_or_raising(monkeypatch)
    assert rollout.navigation_deadlock_errors() == ()
    result = run()
    if operation == "simulate":
        assert result.finished and not result.crashed
    else:
        assert result == (0, 0)


def test_legacy_deadlock_still_ends_the_trial(trial, monkeypatch):
    rollout, operation, run = trial
    monkeypatch.setattr(traffic, "AgentDeadlockError", LegacyDeadlockError, raising=False)
    policy_returning_or_raising(monkeypatch, LegacyDeadlockError("blocked"))
    assert rollout.navigation_deadlock_errors() == (LegacyDeadlockError,)
    result = run()
    if operation == "simulate":
        assert result.crashed and not result.finished
    else:
        assert result == (0, 0)


@pytest.mark.parametrize("legacy", [False, True])
def test_unrelated_navigation_errors_are_not_swallowed(trial, monkeypatch, legacy):
    _, _, run = trial
    if legacy:
        monkeypatch.setattr(traffic, "AgentDeadlockError", LegacyDeadlockError, raising=False)
    else:
        monkeypatch.delattr(traffic, "AgentDeadlockError", raising=False)
    policy_returning_or_raising(monkeypatch, ValueError("unrelated failure"))
    with pytest.raises(ValueError, match="unrelated failure"):
        run()
