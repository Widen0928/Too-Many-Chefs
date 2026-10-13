from dataclasses import replace
from types import SimpleNamespace

import pytest

from controllers.pddl.submission import config, from_pddl, problem_generator, strategy
from simulator.entities import OvercookedState
from simulator.entities.orders import OrderEntry, OrderQueue
from simulator.environment import Environment


def environment(*, strict=False, reveal='all', timed=True):
    return Environment().with_entity(OvercookedState(order_queue=OrderQueue.from_orders(
        [OrderEntry(name='burger', time_limit=60 if timed else None),
         OrderEntry(name='cake', time_limit=100 if timed else None)],
        strict_ordering=strict, order_reveal=reveal,
    )))


def test_swap_policies_follow_all_original_candidates(monkeypatch):
    monkeypatch.setattr(config, 'CONFIG', replace(config.CONFIG, select_deadline_alternatives=True))
    options = [3, '3r', 2, None, 1, '2r', '3safe', 4, '1r', 'r', '4r']
    assert strategy.strategy_options(environment(), options) == [
        *options, '3swap', '3rswap', '2rswap',
    ]
    assert options[-1] == '4r'  # do not mutate the caller's list


@pytest.mark.parametrize('kwargs', [
    {'strict': True}, {'reveal': 'sequential'}, {'timed': False},
])
def test_swap_policies_are_not_added_when_unnecessary_or_illegal(kwargs, monkeypatch):
    monkeypatch.setattr(config, 'CONFIG', replace(config.CONFIG, select_deadline_alternatives=True))
    assert not any(
        isinstance(option, str) and option.endswith('swap')
        for option in strategy.strategy_options(environment(**kwargs), [3, '3r', 2])
    )


def test_experimental_swap_policies_are_disabled_by_default():
    assert strategy.strategy_options(environment(), [3, '3r', 2]) == [3, '3r', 2]


def test_six_identical_orders_try_tip_safe_before_other_horizons():
    queue = OrderQueue.from_orders([OrderEntry(name='cake', time_limit=100)] * 6)
    env = Environment().with_entity(OvercookedState(order_queue=queue))
    options = [3, '3r', 2, '3safe', None]
    assert strategy.strategy_options(env, options) == [3, '3safe', '3tipfarm', '3r', 2, None]
    assert options == [3, '3r', 2, '3safe', None]


@pytest.mark.parametrize('bound, expected_calls', [(51, [3]), (None, [3, '3r'])])
def test_selection_stops_only_after_a_supported_score_bound(monkeypatch, bound, expected_calls):
    from controllers.pddl.submission import score_bound

    calls = []
    monkeypatch.setattr(strategy, 'fingerprint', lambda _: 'episode')
    monkeypatch.setattr(strategy, 'remembered', lambda _: 'missing')
    monkeypatch.setattr(strategy, 'remember', lambda *_: None)
    monkeypatch.setattr(score_bound, 'remaining_score_upper_bound', lambda _: bound)

    def play(_, option):
        calls.append(option)
        return (51, -100)

    monkeypatch.setattr(strategy, 'play_out', play)
    assert strategy.choose_horizon(environment(), [3, '3r']) == 3
    assert calls == expected_calls


@pytest.mark.parametrize('swap_score, expected', [(49, 3), (51, '3swap')])
def test_whole_level_points_decide_whether_to_enable_swapping(monkeypatch, swap_score, expected):
    monkeypatch.setattr(strategy, 'fingerprint', lambda _: 'episode')
    monkeypatch.setattr(strategy, 'remembered', lambda _: 'missing')
    remembered = []
    monkeypatch.setattr(strategy, 'remember', lambda *args: remembered.append(args))
    monkeypatch.setattr(strategy, 'strategy_options', lambda *_: [3, '3swap'])
    monkeypatch.setattr(strategy, 'play_out', lambda _, option: (
        (50, -100) if option == 3 else (swap_score, -20)
    ))
    assert strategy.choose_horizon(environment(), [3]) == expected
    assert remembered == [('episode', expected)]


@pytest.mark.parametrize('policy', ['3rswap', '3rw'])
def test_translator_recovers_reordering_from_swap_policy_in_another_worker(monkeypatch, policy):
    monkeypatch.setattr(from_pddl, 'REORDER', [False, False])
    monkeypatch.setattr(strategy, 'fingerprint', lambda _: 'episode')
    monkeypatch.setattr(strategy, 'remembered', lambda _: policy)
    assert from_pddl._reorder_for(environment())


@pytest.mark.parametrize('policy, swaps, reorder', [
    ('3r', False, True), ('3rswap', True, True), ('3rw', True, True), ('3w', True, False),
    ('swap', True, False),
])
def test_selected_policy_controls_swapping_in_planning_worker(monkeypatch, policy, swaps, reorder):
    env = environment()
    state = env.get_first_entity_of_type(OvercookedState)
    kitchen = SimpleNamespace(state=state)
    entries = state.order_queue.visible
    built = []
    monkeypatch.setattr(problem_generator, 'CONFIG', replace(
        problem_generator.CONFIG, models='fast', deadline_alternatives=False,
    ))
    monkeypatch.setattr(problem_generator, 'Kitchen', lambda *_: kitchen)
    monkeypatch.setattr(problem_generator, 'build_domain', lambda: '(domain)')
    monkeypatch.setattr(problem_generator, 'tip_problem', lambda _: None)
    monkeypatch.setattr(problem_generator, 'goal_sets', lambda *_: [(entries, 1)])
    monkeypatch.setattr(problem_generator, 'probe', lambda *_: [object()])
    monkeypatch.setattr(from_pddl, 'REORDER', [False, False])
    monkeypatch.setattr(from_pddl, 'LAST_EXPIRED', [1])

    def build(_, orders, **kwargs):
        built.append([entry.name for entry in orders])
        return '(problem)'

    monkeypatch.setattr(problem_generator, 'build_problem', build)
    monkeypatch.setattr(problem_generator, 'evaluate', lambda *_: (True, 50, True, -20))
    generator = problem_generator.DefaultProblemGenerator()
    generator.forced_horizon = policy
    generator.to_pddl(env)
    assert built == [['burger', 'cake']] + ([['cake', 'burger']] if swaps else [])
    assert from_pddl.REORDER == [reorder, True]
