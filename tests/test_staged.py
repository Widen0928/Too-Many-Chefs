"""Planning orders one at a time (controllers/pddl/submission/staged.py)."""

from controllers.pddl.interfaces import PDDLPlanStep, PDDLProblem
from controllers.pddl.submission import staged
from controllers.pddl.submission.strategy import strategy_flags

DOMAIN = """
(define (domain toy)
  (:predicates (at ?x) (served ?o) (open ?o) (next_order ?o ?p))
  (:action move :parameters (?a ?b) :precondition (and (at ?a)) :effect (and (at ?b) (not (at ?a))))
  (:action serve :parameters (?o ?p) :precondition (and (open ?o) (next_order ?o ?p) (at home))
   :effect (and (served ?o) (not (open ?o)) (open ?p)))
)
"""
PROBLEM = """(define (problem p) (:domain toy)
  (:objects home away o0 o1 o_end)
  (:init
    (at away) (open o0) (next_order o0 o1) (next_order o1 o_end)
  )
  (:goal
    (and (served o0) (served o1))
  )
)
"""


def test_progress_applies_effects_and_rejects_inapplicable_steps():
    init = {("at", "away"), ("open", "o0"), ("next_order", "o0", "o1")}
    after = staged.progress(DOMAIN, init, [PDDLPlanStep("move", ("away", "home"))])
    assert ("at", "home") in after and ("at", "away") not in after
    assert staged.progress(DOMAIN, init, [PDDLPlanStep("serve", ("o0", "o1"))]) is None


def test_staged_plan_solves_each_order_from_the_previous_state():
    goals = []

    def solve(problem: PDDLProblem):
        goals.append(problem.problem.split("(:goal")[1].split()[1:3])
        if "(at home)" in problem.problem:
            return [PDDLPlanStep("serve", ("o1", "o_end"))]
        return [PDDLPlanStep("move", ("away", "home")), PDDLPlanStep("serve", ("o0", "o1"))]

    plan = staged.staged_plan(PDDLProblem(domain=DOMAIN, problem=PROBLEM), 2, solve)
    assert [step.name for step in plan] == ["move", "serve", "serve"]
    assert goals == [["(served", "o0)"], ["(served", "o1)"]]
    assert staged.staged_plan(PDDLProblem(domain=DOMAIN, problem=PROBLEM), 2, lambda p: []) is None


def test_staged_suffix_is_parsed_first():
    assert strategy_flags("4rg") == {
        "reorder": True, "prefetch": False, "pair": False, "next_prep": False, "staged": True,
        "old_tips": False,
    }
    assert strategy_flags("3r")["staged"] is False
    assert strategy_flags("4rgo") == {**strategy_flags("4rg"), "old_tips": True}
