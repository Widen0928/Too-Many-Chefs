"""A conservative ceiling on points still available in a finite kitchen."""

from __future__ import annotations

from simulator.entities import Agent, OvercookedState
from simulator.environment import Environment


def remaining_score_upper_bound(environment: Environment) -> int | None:
    """Bound future score increments, or return None if no bound is supported.

    Each finite order can pay its delivery reward once. For each food in its
    recipe closure (including the dish), it also permits at most one tip per
    team. Retrieval and preparation share that food's tip counter. Removing
    an order releases only its own one counter slot; it cannot create more
    slots than the finite list of visible and pending orders supplies.

    Count every recipe route, give every food the largest tip reward, and do
    not subtract tips already paid. Impossible recipes and past payments
    therefore only make the bound looser. The result excludes initial score:
    it is compared with the increment returned by strategy.play_out.

    Named teams are deliberately unsupported. Besides their separate tip
    pools, initial unattended cooking can pay the unnamed pool as well.
    Generated orders also have no finite bound here.
    """
    state = environment.get_first_entity_of_type(OvercookedState)
    if state is None or state.order_queue.generator is not None:
        return None
    if any(agent.team for agent in environment.get_entities_of_type(Agent)):
        return None

    scoring = state.scoring
    tip = max(
        scoring.retrieve_order_component_reward,
        scoring.produce_order_component_reward,
        scoring.produce_ordered_item_reward,
    )
    return sum(
        entry.reward + tip * (1 + len(state._order_components(entry.name)))
        for entry in (*state.order_queue.visible, *state.order_queue.pending)
    )
