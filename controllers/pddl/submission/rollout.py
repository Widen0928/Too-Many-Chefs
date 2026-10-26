"""Try a plan out before committing to it.

Execution is deterministic: planning costs no timesteps, and the controller
turns a plan into moves with a fixed navigation policy. So a plan can be run
forward on a copy of the kitchen, with the very same policy and simulator, to
see what it would achieve: how many orders it serves, how many points it
earns, how long it takes, and whether the chefs end up jammed. Older navigation
policies raise ``AgentDeadlockError`` when chefs cannot escape a dead end;
rollouts still catch that error when the installed template provides it.

``from_pddl.py`` builds a few variations of the same plan (who does which
job) and keeps the one whose rollout scores best.
"""

from __future__ import annotations

from dataclasses import dataclass

from simulator.context import Context
from simulator.entities import Agent, OvercookedState
from simulator.environment import Environment
from simulator.mutations import MutationWithLocation

# Rollouts stop after this many timesteps.
MAX_TICKS = 600


def navigation_deadlock_errors() -> tuple[type[Exception], ...]:
    """Catch legacy deadlocks without requiring the removed template class."""
    from controllers.navigation import traffic

    error = getattr(traffic, "AgentDeadlockError", None)
    return (error,) if error is not None else ()


@dataclass(frozen=True)
class Outcome:
    crashed: bool
    delivered: int
    points: int
    expired: int
    # Reward of the orders that expired: points the run can no longer earn.
    lost: int
    # Whether the plan ran to its end (rather than getting stuck on the way).
    finished: bool
    ticks: int

    def key(self) -> tuple:
        """Bigger is better: no crash, then points earned minus points lost,
        then finishing the plan, then speed."""
        return (not self.crashed, self.points - self.lost, self.finished, -self.ticks)


def simulate(
    environment: Environment,
    mutations: list[MutationWithLocation],
    max_ticks: int = MAX_TICKS,
    old_tips: bool = False,
) -> Outcome:
    """Run ``mutations`` forward on a copy of the kitchen.

    ``old_tips`` scores as before the 2026-10-26 template: the tip for the
    food made by taking food from storage onto a held plate or pan is not
    counted. Only this rollout's points change, never the kitchen's; plans
    chosen this way are still played out under the real rules.
    """
    from controllers.navigation import DefaultMutationInterpolationPolicy
    from controllers.pddl.controller import group_mutations_by_agent

    deadlock_errors = navigation_deadlock_errors()
    start = _state(environment)
    agent_ids = [
        agent.id
        for agent in sorted(
            environment.get_entities_of_type(Agent),
            key=lambda a: (
                a.y if a.y is not None else -1,
                a.x if a.x is not None else -1,
                a.id,
            ),
        )
    ]
    plans = group_mutations_by_agent(mutations=mutations, agent_ids=agent_ids)
    policy = DefaultMutationInterpolationPolicy()
    context = Context()
    env = environment
    crashed = False
    finished = False
    adjust = 0
    for _ in range(max_ticks):
        state = _state(env)
        if not state.visible_orders and not state.order_queue.pending:
            finished = True
            break
        try:
            # With no actions, the controller simply asks again without time
            # passing; the policy's path cache sometimes needs that retry.
            for _attempt in range(3):
                actions, plans = policy.get_mutations(plans, env, context)
                if actions:
                    break
        except deadlock_errors:
            crashed = True
            break
        if not actions:
            if all(not queue for queue in plans.values()):
                finished = True
                break
            if not env.has_pending_effects():
                break  # Stuck: the real controller would replan here.
        if old_tips:
            adjust += _new_template_tips(env, actions)
        env = env.step(actions)
    end = _state(env)
    expired = end.order_queue.expired_count - start.order_queue.expired_count
    entries = [*start.order_queue.visible, *start.order_queue.pending]
    reward = sum(e.reward for e in entries) / len(entries) if entries else 0
    return Outcome(
        crashed=crashed,
        delivered=end.order_queue.delivered_count - start.order_queue.delivered_count,
        points=end.score - start.score - adjust,
        expired=expired,
        lost=int(expired * reward),
        finished=finished,
        ticks=env.timestep - environment.timestep,
    )


def _state(environment: Environment) -> OvercookedState:
    state = environment.get_first_entity_of_type(OvercookedState)
    if state is None:
        raise ValueError("The environment has no OvercookedState")
    return state


def _new_template_tips(environment: Environment, actions: list) -> int:
    """Points ``actions`` would earn only under the 2026-10-26 rules: a
    chef taking food from storage onto the plate or pan it holds now also
    gets the tip for the food that makes (beyond the fetch tip it got
    before)."""
    from simulator.entities import Equipment
    from simulator.mutations import TakeFromStorage

    takers = [
        action.agent_id
        for action in actions
        if isinstance(action, TakeFromStorage)
        and isinstance(_held(environment, action.agent_id), Equipment)
    ]
    if not takers:
        return 0
    before = _state(environment).scoring
    after = _state(environment.step(actions)).scoring
    fetch = before.retrieve_order_component_reward
    return sum(
        max(0, after.score_for(agent) - before.score_for(agent) - fetch) for agent in takers
    )


def _held(environment: Environment, agent_id: str):
    agent = environment.get_entity(agent_id)
    held = getattr(agent, "held_item_id", None)
    return environment.get_entity(held) if held else None
