"""Batching is an optional whole-run policy, never a change to strict queues."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from controllers.pddl.submission import config, from_pddl, problem_generator, strategy
from controllers.pddl.submission.kitchen import Kitchen
from simulator.configuration import load, load_configuration
from simulator.entities import OvercookedState
from simulator.entities.orders import OrderEntry, OrderQueue


def queue_kitchen(*, strict=False, reveal="all"):
    entries = [OrderEntry(name=name, time_limit=deadline) for name, deadline in (
        ("hamburger", 42), ("cheeseburger", 66),
        ("hamburger", 80), ("cheeseburger", 105),
    )]
    queue = OrderQueue.from_orders(entries, strict_ordering=strict, order_reveal=reveal)
    return SimpleNamespace(state=SimpleNamespace(order_queue=queue),
                           environment=SimpleNamespace(timestep=0)), entries


def burning_kitchen():
    path = Path(__file__).parents[1] / "levels/2_overcooked/2_1_burger_burning_2p.yaml"
    return load(load_configuration(path))


def test_batching_preserves_deadline_order_within_each_dish_and_does_not_mutate_input():
    kitchen, entries = queue_kitchen()
    original = list(entries)
    batched = problem_generator.batch_orders(kitchen, entries)
    assert batched == [entries[0], entries[2], entries[1], entries[3]]
    assert entries == original
    assert all(actual is expected for actual, expected in zip(
        batched, [entries[0], entries[2], entries[1], entries[3]], strict=True,
    ))


def test_first_appearing_dish_keeps_priority_over_alphabetical_order():
    kitchen, _ = queue_kitchen()
    entries = [OrderEntry(name=name) for name in ("z", "a", "z", "b", "a")]
    assert [entry.name for entry in problem_generator.batch_orders(kitchen, entries)] == [
        "z", "z", "a", "a", "b",
    ]


@pytest.mark.parametrize("kwargs", [
    {"strict": True}, {"reveal": "sequential"},
    {"strict": True, "reveal": "sequential"},
])
def test_strict_or_sequential_queues_cannot_be_batched(kwargs):
    kitchen, entries = queue_kitchen(**kwargs)
    assert problem_generator.batch_orders(kitchen, entries) == entries


@pytest.mark.parametrize("names", [[], ["a"], ["a", "a"], ["a", "b", "c"]])
def test_batching_without_interleaved_duplicates_preserves_queue(names):
    kitchen, _ = queue_kitchen()
    entries = [OrderEntry(name=name) for name in names]
    assert problem_generator.batch_orders(kitchen, entries) == entries


def test_goal_horizon_is_applied_after_batching_the_earliest_deadline_queue(monkeypatch):
    kitchen, entries = queue_kitchen()
    kitchen.state.order_queue = kitchen.state.order_queue.copy_with(
        visible=[entries[2], entries[3], entries[0], entries[1]],
    )
    monkeypatch.setattr(problem_generator, "feasible_orders", lambda _kitchen, orders: orders)
    regular = list(problem_generator.goal_sets(kitchen, 3))
    batched = list(problem_generator.goal_sets(kitchen, 3, batch=True))
    assert regular[0][0] == [entries[0], entries[1], entries[2]]
    assert batched[0][0] == [entries[0], entries[2], entries[1]]
    assert batched[1][0] == [entries[0], entries[2]]


@pytest.mark.parametrize("selected", ["3rb", "3rbw", "3rbswap"])
def test_reordering_is_restored_for_batched_policy_in_a_fresh_worker(tmp_path, monkeypatch, selected):
    environment = burning_kitchen()
    monkeypatch.setattr(strategy, "_DIR", tmp_path)
    strategy.remember(strategy.fingerprint(environment), selected)
    code = """
import json, sys
from pathlib import Path
from controllers.pddl.submission import from_pddl, strategy
from simulator.environment import Environment
strategy._DIR = Path(sys.argv[1])
environment = Environment.model_validate_json(sys.stdin.read())
print(json.dumps(from_pddl._reorder_for(environment)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        input=environment.model_dump_json(), text=True, capture_output=True,
        check=True, cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert json.loads(result.stdout) is True


def test_prefetch_policy_is_restored_in_a_fresh_worker(tmp_path, monkeypatch):
    environment = burning_kitchen()
    monkeypatch.setattr(strategy, "_DIR", tmp_path)
    strategy.remember(strategy.fingerprint(environment), "swapbuns")
    code = """
import json, sys
from pathlib import Path
from controllers.pddl.submission import from_pddl, strategy
from simulator.environment import Environment
strategy._DIR = Path(sys.argv[1])
environment = Environment.model_validate_json(sys.stdin.read())
print(json.dumps(from_pddl._prefetch_for(environment)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        input=environment.model_dump_json(), text=True, capture_output=True,
        check=True, cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert json.loads(result.stdout) is True


def test_pair_policy_is_restored_in_a_fresh_worker(tmp_path, monkeypatch):
    environment = burning_kitchen()
    monkeypatch.setattr(strategy, "_DIR", tmp_path)
    strategy.remember(strategy.fingerprint(environment), "2buns")
    code = """
import json, sys
from pathlib import Path
from controllers.pddl.submission import from_pddl, strategy
from simulator.environment import Environment
strategy._DIR = Path(sys.argv[1])
environment = Environment.model_validate_json(sys.stdin.read())
print(json.dumps(from_pddl._pair_for(environment)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        input=environment.model_dump_json(), text=True, capture_output=True,
        check=True, cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert json.loads(result.stdout) is True


def test_batch_suffix_sets_horizon_and_only_applies_to_its_selected_policy(monkeypatch):
    environment = burning_kitchen()
    monkeypatch.setattr(from_pddl, "REORDER", [False, False])
    monkeypatch.setattr(problem_generator, "tip_problem", lambda *_: None)
    calls = []

    def goals(_kitchen, horizon=None, **kwargs):
        calls.append((horizon, kwargs))
        yield [], 1

    monkeypatch.setattr(problem_generator, "goal_sets", goals)
    generator = problem_generator.DefaultProblemGenerator()
    generator.forced_horizon = "3rb"
    generator.to_pddl(environment)
    assert from_pddl.REORDER == [True, True]
    generator.forced_horizon = 3
    generator.to_pddl(environment)
    assert calls == [(3, {"batch": True}), (3, {})]
    assert from_pddl.REORDER == [False, True]


def test_timed_burning_batch_candidate_preserves_original_first_two_and_is_not_duplicated():
    options = [3, "3r", 2, None, "3w"]
    selected = strategy.strategy_options(burning_kitchen(), options)
    assert selected == [3, "3r", "3rb", "3safer", "1r", "swap", "4rq1", "4rq4", 2, "3w", "3safe", "3rbprep", "swapbuns", "1buns", "2buns", "4rq2", "4rq5", None]
    assert options == [3, "3r", 2, None, "3w"]
    assert strategy.strategy_options(burning_kitchen(), selected).count("3rb") == 1
    assert strategy.strategy_options(burning_kitchen(), selected).count("swap") == 1
    assert strategy.strategy_options(burning_kitchen(), selected).count("swapbuns") == 1
    assert strategy.strategy_options(burning_kitchen(), selected).count("1buns") == 1
    assert strategy.strategy_options(burning_kitchen(), selected).count("2buns") == 1
    for pattern in ("4rq2", "4rq1", "4rq4"):
        assert strategy.strategy_options(burning_kitchen(), selected).count(pattern) == 1
    assert strategy.strategy_options(burning_kitchen(), selected).count("4rq5") == 1


def test_four_ticket_permutation_serves_cheeseburgers_first():
    kitchen = Kitchen(burning_kitchen())
    entries, _ = next(problem_generator.goal_sets(kitchen, horizon=4, order_pattern=5))
    assert [entry.name.rsplit("/", 1)[-1] for entry in entries] == [
        "cheeseburger", "cheeseburger", "hamburger", "hamburger"
    ]


def test_parallel_candidate_can_deliver_any_visible_ticket():
    kitchen = Kitchen(burning_kitchen())
    entries = problem_generator.queued_orders(kitchen)
    serial = problem_generator.build_problem(kitchen, entries, fast=True)
    parallel = problem_generator.build_problem(
        kitchen, entries, fast=True, parallel_orders=True
    )
    assert "(next_order o0 o1)" in serial
    assert "(next_order o0 o1)" not in parallel
    for index in range(len(entries)):
        assert f"(open o{index})" in parallel
        assert f"(next_order o{index} o_end)" in parallel


@pytest.mark.parametrize("restriction", [
    "strict", "sequential", "untimed", "one_order", "no_cooking_chain", "no_reorder", "disabled",
    "all_identical", "all_unique",
])
def test_batch_candidate_is_not_added_outside_supported_kitchens(monkeypatch, restriction):
    environment = burning_kitchen()
    state = environment.get_first_entity_of_type(OvercookedState)
    queue = state.order_queue
    options = [3, "3r", 2]
    if restriction == "strict":
        queue = queue.copy_with(strict_ordering=True)
    elif restriction == "sequential":
        queue = queue.copy_with(order_reveal="sequential")
    elif restriction == "untimed":
        queue = queue.copy_with(visible=[entry.copy_with(time_limit=None) for entry in queue.visible])
    elif restriction == "one_order":
        queue = queue.copy_with(visible=queue.visible[:1])
    elif restriction == "all_identical":
        queue = queue.copy_with(visible=[
            entry.copy_with(name=queue.visible[0].name) for entry in queue.visible
        ])
    elif restriction == "all_unique":
        queue = queue.copy_with(visible=[
            entry.copy_with(name=f"distinct_{index}") for index, entry in enumerate(queue.visible)
        ])
    elif restriction == "no_cooking_chain":
        state = state.copy_with(cook_recipes=[
            recipe for recipe in state.cook_recipes if recipe.output != "burnt_meat"
        ])
    elif restriction == "no_reorder":
        options = [3, 2]
    elif restriction == "disabled":
        monkeypatch.setattr(config, "CONFIG", replace(config.CONFIG, select_order_batches=False))
    environment = environment.replace_entity(state.copy_with(order_queue=queue))
    assert "3rb" not in strategy.strategy_options(environment, options)
    assert "4rq5" not in strategy.strategy_options(environment, options)
