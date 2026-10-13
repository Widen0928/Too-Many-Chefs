"""A moved cooking episode must stay together across separately assigned jobs."""

import pytest

from controllers.pddl.interfaces import PDDLPlanStep
from controllers.pddl.submission.dispatch import Options, dispatch
from controllers.pddl.submission.from_pddl import DefaultFromPDDL
from controllers.pddl.submission.kitchen import Kitchen, chef_symbol
from controllers.pddl.submission.rollout import simulate
from simulator.configuration.configuration import Configuration
from simulator.configuration.load import load
from simulator.entities import Agent


def kitchen_with_twin_boards(*, holding=False):
    agent = {"symbol": "1", "orientation": "n"}
    if holding:
        agent["held_item"] = {"name": "catalog/food/coconut", "raw": True}
    environment = load(Configuration.from_dict({
        "layout": "| -/|   |   |   | C | -/| -p|\n"
                  "|   |   |   |   |   | 1 |   |\n"
                  "|   |   |   |   |   |   | * |",
        "state": {"orders": ["catalog/food/coconut_juice"]},
        "legend": {
            "agents": [agent],
            "counters": [{"symbol": "-"}],
            "plates": [{"symbol": "p"}],
            "deliveries": [{"symbol": "*"}],
            "equipment": [{"symbol": "/", "name": "catalog/equipment/cutboard"}],
            "storages": [{"symbol": "C", "food_name": "catalog/food/coconut"}],
            "foods": [{"name": "catalog/food/coconut"},
                      {"name": "catalog/food/coconut_juice"}],
        },
        "recipes": {"cook": [{"ingredient": "catalog/food/coconut",
                                "with": "catalog/equipment/cutboard",
                                "to_make": "catalog/food/coconut_juice"}],
                    "combine": []},
    }))
    chef = chef_symbol(environment.get_first_entity_of_type(Agent).id)
    steps = [] if holding else [
        PDDLPlanStep("take_food", (chef, "loc_4_0", "f_coconut")),
    ]
    steps.extend([
        PDDLPlanStep("put_in_station", (chef, "s_0_0_cutboard", "loc_0_0", "f_coconut")),
        PDDLPlanStep("cook", (chef, "s_0_0_cutboard", "loc_0_0",
                              "f_coconut", "f_coconut_juice")),
        PDDLPlanStep("pick_up", (chef, "loc_6_0", "p_empty")),
        PDDLPlanStep("station_onto_plate", (chef, "s_0_0_cutboard", "loc_0_0",
                                            "f_coconut_juice", "p_empty", "p_coconut_juice")),
        PDDLPlanStep("deliver", (chef, "loc_6_2", "p_coconut_juice", "o0", "o_end")),
    ])
    return environment, steps


def test_first_job_uses_the_chefs_actual_position_for_station_choice():
    environment, steps = kitchen_with_twin_boards(holding=True)
    chosen = dispatch(Kitchen(environment), steps)
    assert chosen[0].parameters[1:3] == ("s_5_0_cutboard", "loc_5_0")


@pytest.mark.parametrize("reorder", [False, True])
def test_station_move_is_retained_for_cooking_and_plating_jobs(reorder):
    environment, steps = kitchen_with_twin_boards()
    chosen = dispatch(Kitchen(environment), steps, Options(reorder=reorder))
    station_steps = [step for step in chosen if step.name in {
        "put_in_station", "cook", "station_onto_plate",
    }]
    assert [step.parameters[1:3] for step in station_steps] == [
        ("s_5_0_cutboard", "loc_5_0"),
    ] * 3
    mutations = DefaultFromPDDL(environment).translate_all(Kitchen(environment), chosen, False)
    result = simulate(environment, mutations)
    assert result.finished
    assert result.delivered == 1
    assert not result.crashed
