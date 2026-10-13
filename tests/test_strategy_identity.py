import pickle

from controllers.pddl.submission.strategy import fingerprint
from controllers.pddl.submission import from_pddl, strategy
from simulator.entities import OvercookedState
from simulator.entities.orders import OrderEntry, OrderQueue
from simulator.environment import Environment


def test_strategy_selection_is_isolated_between_same_layout_episodes():
    first = Environment().with_entity(OvercookedState(name='same kitchen'))
    second = Environment().with_entity(OvercookedState(name='same kitchen'))
    assert fingerprint(first) != fingerprint(second)


def test_strategy_identity_survives_worker_serialization_and_order_changes():
    original = Environment().with_entity(OvercookedState(
        name='same kitchen',
        order_queue=OrderQueue(visible=[OrderEntry(name='burger')]),
    ))
    worker_copy = pickle.loads(pickle.dumps(original))
    state = worker_copy.get_first_entity_of_type(OvercookedState)
    later = worker_copy.replace_entity(state.copy_with(
        order_queue=state.order_queue.copy_with(visible=[], delivered_count=1),
    )).copy_with(timestep=20)
    assert fingerprint(original) == fingerprint(worker_copy) == fingerprint(later)


def test_prep_strategy_flags_survive_separate_translator_process(monkeypatch):
    monkeypatch.setattr(strategy, "fingerprint", lambda _: "episode")
    monkeypatch.setattr(strategy, "remembered", lambda _: "3rbprep")
    monkeypatch.setattr(from_pddl, "REORDER", [False, False])
    monkeypatch.setattr(from_pddl, "NEXT_PREP", [False, False])
    assert from_pddl._reorder_for(object())
    assert from_pddl._next_prep_for(object())
