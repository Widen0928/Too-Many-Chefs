from types import SimpleNamespace

import pytest

from controllers.pddl.submission.problem_generator import can_swap_orders
from simulator.entities.orders import OrderEntry, OrderQueue


@pytest.mark.parametrize(
    ('strict', 'reveal', 'allowed'),
    [(False, 'all', True), (True, 'all', False),
     (False, 'sequential', False), (True, 'sequential', False)],
)
def test_swap_respects_delivery_rules(strict, reveal, allowed):
    entries = [OrderEntry(name='burger'), OrderEntry(name='cake')]
    queue = OrderQueue.from_orders(entries, strict_ordering=strict, order_reveal=reveal)
    kitchen = SimpleNamespace(state=SimpleNamespace(order_queue=queue))
    assert can_swap_orders(kitchen, entries) is allowed


def test_swap_requires_both_orders_to_be_visible():
    first, second = OrderEntry(name='burger'), OrderEntry(name='cake')
    queue = OrderQueue(visible=[first], pending=[second])
    kitchen = SimpleNamespace(state=SimpleNamespace(order_queue=queue))
    assert not can_swap_orders(kitchen, [first, second])
    assert not can_swap_orders(kitchen, [first])
