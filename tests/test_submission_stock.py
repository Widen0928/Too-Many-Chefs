"""Scarce ingredients must not be spent on a cheaper interchangeable job."""

from controllers.pddl.submission.kitchen import Kitchen
from controllers.pddl.submission.problem_generator import feasible_orders, queued_orders
from controllers.pddl.submission.strategy import play_out
from simulator.configuration import configuration_from_dict
from simulator.configuration.load import load
from simulator.entities import CombineRecipe, OvercookedState


def scarce_kitchen(orders, *, cherries=1):
    """Both dishes need a cherry; their other ingredient is unlimited."""
    return load(configuration_from_dict({
        "layout": (
            "| -r| A | B | -p| -p| * |\n"
            "| 1 |   |   |   |   |   |\n"
            f"| {'-r' if cherries == 2 else '-'} | - | - | X | - | - |"
        ),
        "state": {"orders": orders},
        "legend": {
            "agents": [{"symbol": "1", "orientation": "n"}],
            "counters": [{"symbol": "-"}],
            "plates": [{"symbol": "p"}],
            "bins": [{"symbol": "X"}],
            "deliveries": [{"symbol": "*"}],
            "storages": [
                {"symbol": "A", "food_name": "apple"},
                {"symbol": "B", "food_name": "bun"},
            ],
            "foods": [
                {"symbol": "r", "name": "cherry"},
                {"name": "apple"}, {"name": "bun"},
                {"name": "premium", "raw": True, "deliverable": True},
                {"name": "cheap", "raw": True, "deliverable": True},
            ],
        },
        "recipes": {"combine": [
            {"ingredients": ["cherry", "apple"], "to_make": "premium"},
            {"ingredients": ["cherry", "bun"], "to_make": "cheap"},
        ]},
    }))


def selected(environment):
    kitchen = Kitchen(environment)
    return [entry.name for entry in feasible_orders(kitchen, queued_orders(kitchen))]


def test_finite_ingredient_goes_to_higher_reward_ticket():
    env = scarce_kitchen([
        {"food": "premium", "reward": 100},
        {"food": "cheap", "reward": 10},
    ])
    assert selected(env) == ["premium"]


def test_same_rewards_preserve_existing_subset_selection():
    env = scarce_kitchen([
        {"food": "premium", "reward": 50},
        {"food": "cheap", "reward": 50},
    ])
    assert selected(env) == ["cheap"]


def test_all_orders_are_kept_when_stock_suffices():
    env = scarce_kitchen([
        {"food": "premium", "reward": 100},
        {"food": "cheap", "reward": 10},
    ], cherries=2)
    assert selected(env) == ["premium", "cheap"]


def test_larger_delivery_reward_can_outweigh_serving_more_dishes():
    env = scarce_kitchen([
        {"food": "premium", "reward": 100},
        {"food": "cheap", "reward": 10},
        {"food": "cheap", "reward": 10},
    ], cherries=2)
    state = env.get_first_entity_of_type(OvercookedState)
    # A one-name combine recipe combines two copies of that food.
    state = state.copy_with(combine_recipes=[
        CombineRecipe(ingredients=frozenset({"cherry"}), output="premium"),
        state.combine_recipes[1],
    ])
    assert selected(env.replace_entity(state)) == ["premium"]


def test_cannot_target_a_more_expensive_later_ticket_for_same_dish():
    env = scarce_kitchen([
        {"food": "premium", "reward": 10},
        {"food": "premium", "reward": 100},
        {"food": "cheap", "reward": 20},
    ])
    # A premium dish would serve the first premium ticket and earn only 10.
    assert selected(env) == ["cheap"]


def test_finite_stock_selection_improves_real_simulator_score():
    env = scarce_kitchen([
        {"food": "premium", "reward": 100},
        {"food": "cheap", "reward": 10},
    ])
    # The count-only policy used the sole cherry for the cheap dish and
    # finished on 13 points. This exercises actual planning and execution.
    assert play_out(env, 3)[0] >= 100
