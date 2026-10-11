"""Regression checks for semantic rollout cache isolation."""

import os
import subprocess
import sys

import pytest

from controllers.pddl.submission import from_pddl
from simulator.entities import (
    Agent,
    Bounds,
    Delivery,
    Equipment,
    Food,
    OvercookedState,
    PlateDispenser,
)
from simulator.entities.equipment import CookingTimer
from simulator.entities.orders import OrderEntry, OrderQueue
from simulator.environment import Environment


def kitchen() -> Environment:
    return (
        Environment()
        .with_entity(Bounds(id="bounds", width=3, height=3))
        .with_entity(Agent(id="chef", x=1, y=1))
        .with_entity(OvercookedState(
            id="state",
            order_queue=OrderQueue(
                visible=[OrderEntry(name="soup", time_limit=30)],
                pending=[OrderEntry(name="salad", reward=70)],
            ),
        ))
    )


@pytest.mark.parametrize("change", [
    {"tipped": {"": {"soup": 1}}},
    {"pending": [OrderEntry(name="salad", reward=71)]},
    {"strict_ordering": True},
    {"order_reveal": "sequential"},
])
def test_order_accounting_changes_invalidate_rollout_cache(change) -> None:
    original = kitchen()
    state = original.get_first_entity_of_type(OvercookedState)
    changed = original.replace_entity(state.copy_with(
        order_queue=state.order_queue.copy_with(**change),
    ))
    assert from_pddl._cache_key(original, []) != from_pddl._cache_key(changed, [])


def test_scoring_rules_invalidate_rollout_cache() -> None:
    original = kitchen()
    state = original.get_first_entity_of_type(OvercookedState)
    changed = original.replace_entity(state.copy_with(
        scoring=state.scoring.copy_with(produce_ordered_item_reward=5),
    ))
    assert from_pddl._cache_key(original, []) != from_pddl._cache_key(changed, [])


def test_action_limit_changes_invalidate_rollout_cache() -> None:
    original = kitchen()
    assert from_pddl._cache_key(original, []) != from_pddl._cache_key(
        original.copy_with(actions_per_step=2), [],
    )


@pytest.mark.parametrize('entity, change', [
    (Delivery(id='delivery', x=2, y=2), {'accepted_item_names': {'salad'}}),
    (PlateDispenser(id='plates', x=2, y=2), {'returns_dirty': True}),
])
def test_station_rules_invalidate_rollout_cache(entity, change) -> None:
    original = kitchen().with_entity(entity)
    changed = original.replace_entity(entity.copy_with(**change))
    assert from_pddl._cache_key(original, []) != from_pddl._cache_key(changed, [])


def test_random_food_ids_do_not_prevent_trial_and_execution_cache_hits() -> None:
    def cooking_kitchen(food_id: str) -> Environment:
        return kitchen().with_entity(
            Food(id=food_id, name="soup", x=None, y=None),
        ).with_entity(Equipment(
            id="pan", name="pan", x=2, y=2, held_item_id=food_id,
            cooking=CookingTimer(food_id=food_id, elapsed=3, started_by="chef"),
        ))

    assert from_pddl._cache_key(cooking_kitchen("trial-food"), []) == (
        from_pddl._cache_key(cooking_kitchen("execution-food"), [])
    )


def test_stale_timer_does_not_share_cache_with_active_timer() -> None:
    original = kitchen().with_entity(
        Food(id="food", name="soup", x=None, y=None),
    ).with_entity(Equipment(
        id="pan", x=2, y=2, held_item_id="food",
        cooking=CookingTimer(food_id="food", elapsed=3),
    ))
    pan = original.get_entity_as("pan", Equipment)
    changed = original.replace_entity(pan.copy_with(
        cooking=pan.cooking.copy_with(food_id="previous-food"),
    ))
    assert from_pddl._cache_key(original, []) != from_pddl._cache_key(changed, [])


@pytest.mark.parametrize("submission_file", [
    "controllers/pddl/submission/from_pddl.py",
    "controllers/open/submission/planner/from_pddl.py",
])
def test_runtime_source_invalidates_namespace_once_per_worker(tmp_path, monkeypatch, submission_file) -> None:
    paths = [
        submission_file,
        "controllers/pddl/controller.py",
        "controllers/navigation/policy.py",
        "simulator/mutations/move_agent.py",
    ]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original runtime\n")
    monkeypatch.setattr(from_pddl, "__file__", str(tmp_path / paths[0]))
    from_pddl._cache_namespace.cache_clear()
    try:
        old = from_pddl._cache_namespace()
        (tmp_path / paths[-1]).write_text("# moving now changes orientation\n")
        assert from_pddl._cache_namespace() == old
        from_pddl._cache_namespace.cache_clear()  # a newly started worker
        assert from_pddl._cache_namespace() != old
    finally:
        from_pddl._cache_namespace.cache_clear()


def test_namespace_is_part_of_rollout_key(monkeypatch) -> None:
    original = kitchen()
    monkeypatch.setattr(from_pddl, "_cache_namespace", lambda: "old-runtime")
    old = from_pddl._cache_key(original, [])
    monkeypatch.setattr(from_pddl, "_cache_namespace", lambda: "new-runtime")
    assert from_pddl._cache_key(original, []) != old


def test_cache_key_matches_across_hash_seeds_and_random_state_ids() -> None:
    script = """
from controllers.pddl.submission.from_pddl import _cache_key
from simulator.entities import OvercookedState
from simulator.entities.recipes import CombineRecipe
from tests.test_submission_cache import kitchen

env = kitchen()
old = env.get_first_entity_of_type(OvercookedState)
state = OvercookedState(
    order_queue=old.order_queue,
    combine_recipes=[CombineRecipe(ingredients=frozenset({'a', 'b', 'c'}), output='soup')],
)
print(_cache_key(env.without_entity(old.id).with_entity(state), []))
"""
    keys = []
    for seed in ("1", "8675309"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, check=True, timeout=10,
        )
        keys.append(result.stdout.strip())
    assert keys[0] != "None"
    assert keys[0] == keys[1]


def test_nested_sets_have_a_canonical_json_representation() -> None:
    assert from_pddl._canonical_json({"rules": [frozenset({"b", "a"})]}) == (
        '{"rules":[["a","b"]]}'
    )
