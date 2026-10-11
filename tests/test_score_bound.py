"""Exercise the ceiling against the simulator's real scoring transitions."""

from __future__ import annotations

import pytest

from controllers.pddl.submission.score_bound import remaining_score_upper_bound
from simulator.entities import Agent, OvercookedState, ScoringState
from simulator.entities.orders import OrderEntry, OrderGenerator, OrderQueue
from simulator.environment import Environment
from tests.scoring_states import CHOPPED_TOMATO, SALAD, TOMATO, scoring_state

LETTUCE = "catalog/food/lettuce"


def environment_with(state: OvercookedState, *agents: Agent) -> Environment:
    return Environment(entities={item.id: item for item in (state, *agents)})


def prepare_salad(state: OvercookedState) -> OvercookedState:
    # Extra attempts exercise the shared per-food quota rather than assuming
    # repeated retrieval or preparation can earn extra points.
    for _ in range(4):
        state = state.notify_item_retrieved(TOMATO)
        state = state.notify_item_retrieved(LETTUCE)
        state = state.notify_item_prepared(CHOPPED_TOMATO)
        state = state.notify_item_prepared(SALAD)
    return state


@pytest.mark.parametrize("reveal", ["all", "sequential"])
def test_duplicate_orders_and_released_tips_reach_default_reward_bound(reveal):
    start = scoring_state(reveal=reveal).copy_with(scoring=ScoringState())
    bound = remaining_score_upper_bound(environment_with(start))
    state = start
    for timestep in range(2):
        state = prepare_salad(state)
        state = state.notify_item_delivered(SALAD, timestep=timestep)
        assert state.score - start.score <= bound
    assert state.score - start.score == bound == 108
    assert remaining_score_upper_bound(environment_with(state)) == 0


@pytest.mark.parametrize("reveal", ["all", "sequential"])
def test_unequal_tip_rewards_are_conservatively_bounded(reveal):
    start = scoring_state(reveal=reveal)
    bound = remaining_score_upper_bound(environment_with(start))
    state = start
    for timestep in range(2):
        state = prepare_salad(state)
        state = state.notify_item_delivered(SALAD, timestep=timestep)
    assert state.score - start.score == 124
    assert bound == 140
    assert state.score - start.score <= bound


def test_dish_also_used_as_component_cannot_exceed_max_tip():
    start = scoring_state().copy_with(
        order_queue=OrderQueue.from_orders(
            [OrderEntry(name=SALAD, reward=50), OrderEntry(name=CHOPPED_TOMATO, reward=90)]
        )
    )
    bound = remaining_score_upper_bound(environment_with(start))
    state = prepare_salad(start)
    state = state.notify_item_delivered(CHOPPED_TOMATO, timestep=0)
    state = prepare_salad(state)
    state = state.notify_item_delivered(SALAD, timestep=1)
    assert bound == 170
    assert state.score - start.score <= bound


def test_expiry_releases_tips_without_creating_unbounded_slots():
    start = scoring_state(reveal="sequential", time_limit=1)
    bound = remaining_score_upper_bound(environment_with(start))
    state = prepare_salad(start).advance_to_timestep(2)
    assert state.order_queue.expired_count == 1
    state = prepare_salad(state)
    state = state.notify_item_delivered(SALAD, timestep=2)
    assert state.score - start.score == 74
    assert state.score - start.score <= bound


def test_bound_is_an_increment_and_ignores_already_paid_tips():
    state = scoring_state().copy_with(scoring=ScoringState(score=500))
    state = state.notify_item_retrieved(TOMATO)
    bound = remaining_score_upper_bound(environment_with(state))
    initial_score = state.score
    for timestep in range(2):
        state = prepare_salad(state)
        state = state.notify_item_delivered(SALAD, timestep=timestep)
    assert bound == 108
    assert state.score - initial_score == 107
    assert state.score - initial_score <= bound


@pytest.mark.parametrize("teams", [(None, None), ("", None)])
def test_unnamed_chefs_share_one_tip_pool(teams):
    state = scoring_state()
    agents = [Agent(team=team) for team in teams]
    assert remaining_score_upper_bound(environment_with(state, *agents)) == 140


@pytest.mark.parametrize("teams", [("red",), ("red", "blue"), ("red", None)])
def test_named_teams_disable_the_bound(teams):
    agents = [Agent(team=team) for team in teams]
    assert remaining_score_upper_bound(environment_with(scoring_state(), *agents)) is None


def test_generated_orders_disable_the_bound():
    state = scoring_state().copy_with(
        order_queue=OrderQueue.from_orders(
            [], generator=OrderGenerator(pool=[OrderEntry(name=SALAD)])
        )
    )
    assert remaining_score_upper_bound(environment_with(state)) is None


def test_missing_kitchen_state_disables_the_bound():
    assert remaining_score_upper_bound(Environment()) is None
