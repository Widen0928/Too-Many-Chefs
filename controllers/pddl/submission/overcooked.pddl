; Too Many Chefs: kitchen domain shared by every level.
;
; problem_generator.py describes one kitchen as a problem for this domain and
; from_pddl.py turns each planned action into one simulator mutation.
;
; Modelling choices
; - Movement is not modelled. The controller walks a chef to the tile next to
;   an action's target, so an action only needs the chef to be able to reach
;   that tile: (reach ?c ?l), computed from the floor plan.
; - Food and plates are described by type, not by identity. Two eggs are the
;   same object `f_egg`; where they are tells them apart. A plate is described
;   by its state: `p_empty`, `p_dirty`, or `p_<food>` for a plate holding food.
; - A station (cutboard, pan, oven, mixing bowl, ...) is an object of its own,
;   because a stove, its pan and an oven may share one tile.
; - Recipe rules are compiled into static facts by problem_generator.py,
;   which mirrors the simulator's own resolution order, so every action here
;   corresponds to exactly one legal simulator mutation.
; - Only positive preconditions are used; "empty" conditions are explicit
;   predicates (hand_empty, free, st_empty).
(define (domain overcooked)
  (:requirements :strips :typing)

  (:types
    chef loc station thing count order - object
    food pstate - thing
  )

  (:predicates
    ; --- chefs ---
    (reach ?c - chef ?l - loc)
    (hand_empty ?c - chef)
    (holds ?c - chef ?t - thing)

    ; --- counters ---
    (free ?l - loc)
    (on ?t - thing ?l - loc)
    (placeable ?t - thing)
    (combine_counter ?f - food ?g - food ?o - food)

    ; --- storages, delivery, bins and sinks ---
    (storage ?l - loc ?f - food)
    (delivery ?l - loc)
    (bin ?l - loc)
    (sink ?l - loc)

    ; --- stations ---
    (st_at ?s - station ?l - loc)
    (st_empty ?s - station)
    (in ?f - food ?s - station)
    (fixed ?s - station)
    ; A station cooks only while it stands where it works: fixed stations
    ; always, a pan or bowl only on its stove or mixer. Food can still be
    ; put in or taken out of a pan standing on a bare counter.
    (usable ?s - station)
    (portable ?s - station)
    (holds_st ?c - chef ?s - station)
    (st_on_counter ?s - station ?l - loc)
    (support_for ?s - station ?l - loc)
    (base_free ?l - loc)
    (accepts ?s - station ?f - food)
    (merge_in ?s - station ?f - food ?e - food ?o - food)
    (merge_hand ?s - station ?f - food ?e - food ?o - food)
    (cook_by_hand ?s - station ?f - food ?o - food)
    (cook_auto ?s - station ?f - food ?o - food)
    (st_to_plate ?s - station ?p - pstate ?f - food ?q - pstate)
    ; Food in a station that would burn it is collected by the chef who put
    ; it in (tending), so nobody wanders off and leaves it to burn. Other
    ; stations are open to anyone.
    (open_station ?s - station)
    (tending ?c - chef ?s - station)

    ; --- plates ---
    (plating ?p - pstate ?f - food ?q - pstate)
    (plate_of ?p - pstate ?f - food)
    (empty_plate ?p - pstate)
    (dirty_plate ?p - pstate)
    (dispenser ?l - loc)
    (fast_plating)
    (infinite_dispenser ?l - loc)
    (clean_n ?l - loc ?n - count)
    (dirty_n ?l - loc ?n - count)
    (succ ?n - count ?m - count)
    (zero ?n - count)
    (returns_clean ?l - loc)
    (returns_dirty ?l - loc)
    (returns_nowhere)

    ; --- orders ---
    (wants ?o - order ?p - pstate)
    (open ?o - order)
    (next_order ?o - order ?o2 - order)
    (served ?o - order)
  )

  ; =========================================================================
  ; Storages and plate dispensers
  ; =========================================================================

  (:action take_food
    :parameters (?c - chef ?l - loc ?f - food)
    :precondition (and (reach ?c ?l) (hand_empty ?c) (storage ?l ?f))
    :effect (and (holds ?c ?f) (not (hand_empty ?c)))
  )

  ; Take an ingredient straight onto the plate the chef is holding.
  ; Only onto an empty plate: a dish built straight from a storage onto a
  ; plate is not credited as prepared, which loses its tip.
  (:action take_food_onto_plate
    :parameters (?c - chef ?l - loc ?f - food ?p - pstate ?q - pstate)
    :precondition (and (reach ?c ?l) (holds ?c ?p) (empty_plate ?p) (storage ?l ?f)
                       (plating ?p ?f ?q))
    :effect (and (holds ?c ?q) (not (holds ?c ?p)))
  )

  ; Add an ingredient from a storage straight to the dish on the held plate.
  ; Quicker, but the simulator does not credit the new dish as prepared, so
  ; its tip is lost; the problem allows it (fast_plating) only when time
  ; matters more than tips.
  (:action take_food_onto_dish
    :parameters (?c - chef ?l - loc ?f - food ?p - pstate ?q - pstate)
    :precondition (and (reach ?c ?l) (holds ?c ?p) (fast_plating) (storage ?l ?f)
                       (plating ?p ?f ?q))
    :effect (and (holds ?c ?q) (not (holds ?c ?p)))
  )

  (:action take_plate_infinite
    :parameters (?c - chef ?l - loc ?p - pstate)
    :precondition (and (reach ?c ?l) (hand_empty ?c) (infinite_dispenser ?l)
                       (empty_plate ?p))
    :effect (and (holds ?c ?p) (not (hand_empty ?c)))
  )

  (:action take_plate_clean
    :parameters (?c - chef ?l - loc ?n - count ?m - count ?p - pstate)
    :precondition (and (reach ?c ?l) (hand_empty ?c) (dispenser ?l)
                       (clean_n ?l ?m) (succ ?n ?m) (empty_plate ?p))
    :effect (and (holds ?c ?p) (not (hand_empty ?c))
                 (clean_n ?l ?n) (not (clean_n ?l ?m)))
  )

  ; A dispenser hands out clean plates before dirty ones.
  (:action take_plate_dirty
    :parameters (?c - chef ?l - loc ?z - count ?n - count ?m - count ?p - pstate)
    :precondition (and (reach ?c ?l) (hand_empty ?c) (dispenser ?l)
                       (clean_n ?l ?z) (zero ?z)
                       (dirty_n ?l ?m) (succ ?n ?m) (dirty_plate ?p))
    :effect (and (holds ?c ?p) (not (hand_empty ?c))
                 (dirty_n ?l ?n) (not (dirty_n ?l ?m)))
  )

  ; =========================================================================
  ; Counters
  ; =========================================================================

  (:action pick_up
    :parameters (?c - chef ?l - loc ?t - thing)
    :precondition (and (reach ?c ?l) (hand_empty ?c) (on ?t ?l))
    :effect (and (holds ?c ?t) (not (hand_empty ?c))
                 (free ?l) (not (on ?t ?l)))
  )

  ; Only raw food and plates may rest on a bare counter.
  (:action put_down
    :parameters (?c - chef ?l - loc ?t - thing)
    :precondition (and (reach ?c ?l) (holds ?c ?t) (free ?l) (placeable ?t))
    :effect (and (hand_empty ?c) (not (holds ?c ?t))
                 (on ?t ?l) (not (free ?l)))
  )

  ; Held food combined with food on a counter; the result stays there.
  (:action combine_on_counter
    :parameters (?c - chef ?l - loc ?f - food ?g - food ?o - food)
    :precondition (and (reach ?c ?l) (holds ?c ?f) (on ?g ?l)
                       (combine_counter ?f ?g ?o))
    :effect (and (hand_empty ?c) (not (holds ?c ?f))
                 (on ?o ?l) (not (on ?g ?l)))
  )

  ; Held food added to a plate standing on a counter.
  (:action food_onto_counter_plate
    :parameters (?c - chef ?l - loc ?f - food ?p - pstate ?q - pstate)
    :precondition (and (reach ?c ?l) (holds ?c ?f) (on ?p ?l) (plating ?p ?f ?q))
    :effect (and (hand_empty ?c) (not (holds ?c ?f))
                 (on ?q ?l) (not (on ?p ?l)))
  )

  ; Food on a counter scooped onto the held plate.
  (:action counter_food_onto_plate
    :parameters (?c - chef ?l - loc ?f - food ?p - pstate ?q - pstate)
    :precondition (and (reach ?c ?l) (holds ?c ?p) (on ?f ?l) (plating ?p ?f ?q))
    :effect (and (holds ?c ?q) (not (holds ?c ?p))
                 (free ?l) (not (on ?f ?l)))
  )

  ; The food on a plate standing on a counter moved onto the held empty
  ; plate. (A held plate with food on it would pour its food onto the
  ; counter plate instead.)
  (:action counter_plate_onto_plate
    :parameters (?c - chef ?l - loc ?r - pstate ?f - food ?p - pstate ?q - pstate
                 ?e - pstate)
    :precondition (and (reach ?c ?l) (holds ?c ?p) (empty_plate ?p) (on ?r ?l)
                       (plate_of ?r ?f) (plating ?p ?f ?q) (empty_plate ?e))
    :effect (and (holds ?c ?q) (not (holds ?c ?p))
                 (on ?e ?l) (not (on ?r ?l)))
  )

  ; =========================================================================
  ; Stations
  ; =========================================================================

  (:action put_in_station
    :parameters (?c - chef ?s - station ?l - loc ?f - food)
    :precondition (and (reach ?c ?l) (st_at ?s ?l) (holds ?c ?f) (st_empty ?s)
                       (accepts ?s ?f))
    :effect (and (hand_empty ?c) (not (holds ?c ?f))
                 (in ?f ?s) (not (st_empty ?s)) (tending ?c ?s))
  )

  ; Held food mixed into what the station holds; the result stays inside.
  (:action add_to_station
    :parameters (?c - chef ?s - station ?l - loc ?f - food ?e - food ?o - food)
    :precondition (and (reach ?c ?l) (st_at ?s ?l) (holds ?c ?f) (in ?e ?s)
                       (merge_in ?s ?f ?e ?o))
    :effect (and (hand_empty ?c) (not (holds ?c ?f))
                 (in ?o ?s) (not (in ?e ?s)))
  )

  ; Held food combined with the station's food; the chef keeps the result.
  (:action combine_from_station
    :parameters (?c - chef ?s - station ?l - loc ?f - food ?e - food ?o - food)
    :precondition (and (reach ?c ?l) (st_at ?s ?l) (holds ?c ?f) (in ?e ?s)
                       (merge_hand ?s ?f ?e ?o) (open_station ?s))
    :effect (and (holds ?c ?o) (not (holds ?c ?f))
                 (st_empty ?s) (not (in ?e ?s)))
  )

  ; The same, for the chef tending a station that would burn the food.
  (:action combine_from_station_tended
    :parameters (?c - chef ?s - station ?l - loc ?f - food ?e - food ?o - food)
    :precondition (and (reach ?c ?l) (st_at ?s ?l) (holds ?c ?f) (in ?e ?s)
                       (merge_hand ?s ?f ?e ?o) (tending ?c ?s))
    :effect (and (holds ?c ?o) (not (holds ?c ?f))
                 (st_empty ?s) (not (in ?e ?s)) (not (tending ?c ?s)))
  )

  ; Take food out of a station that cannot be lifted, such as a cutboard.
  (:action take_out
    :parameters (?c - chef ?s - station ?l - loc ?f - food)
    :precondition (and (reach ?c ?l) (st_at ?s ?l) (hand_empty ?c) (fixed ?s)
                       (in ?f ?s))
    :effect (and (holds ?c ?f) (not (hand_empty ?c))
                 (st_empty ?s) (not (in ?f ?s)))
  )

  ; Chop, or press cook on a station that does not cook by itself.
  (:action cook
    :parameters (?c - chef ?s - station ?l - loc ?f - food ?o - food)
    :precondition (and (usable ?s) (reach ?c ?l) (st_at ?s ?l) (in ?f ?s)
                       (cook_by_hand ?s ?f ?o))
    :effect (and (in ?o ?s) (not (in ?f ?s)))
  )

  ; Timed stations (pan, pot, oven, ...) cook without a chef.
  (:action cook_by_itself
    :parameters (?s - station ?f - food ?o - food)
    :precondition (and (usable ?s) (in ?f ?s) (cook_auto ?s ?f ?o))
    :effect (and (in ?o ?s) (not (in ?f ?s)))
  )

  ; Station food moved onto the held plate.
  (:action station_onto_plate
    :parameters (?c - chef ?s - station ?l - loc ?f - food ?p - pstate ?q - pstate)
    :precondition (and (reach ?c ?l) (st_at ?s ?l) (holds ?c ?p) (in ?f ?s)
                       (st_to_plate ?s ?p ?f ?q) (open_station ?s))
    :effect (and (holds ?c ?q) (not (holds ?c ?p))
                 (st_empty ?s) (not (in ?f ?s)))
  )

  ; The same, for the chef tending a station that would burn the food.
  (:action station_onto_plate_tended
    :parameters (?c - chef ?s - station ?l - loc ?f - food ?p - pstate ?q - pstate)
    :precondition (and (reach ?c ?l) (st_at ?s ?l) (holds ?c ?p) (in ?f ?s)
                       (st_to_plate ?s ?p ?f ?q) (tending ?c ?s))
    :effect (and (holds ?c ?q) (not (holds ?c ?p))
                 (st_empty ?s) (not (in ?f ?s)) (not (tending ?c ?s)))
  )

  ; The held plate's food poured into an empty station.
  (:action plate_into_station
    :parameters (?c - chef ?s - station ?l - loc ?p - pstate ?f - food ?e - pstate)
    :precondition (and (reach ?c ?l) (st_at ?s ?l) (holds ?c ?p) (plate_of ?p ?f)
                       (st_empty ?s) (accepts ?s ?f) (empty_plate ?e))
    :effect (and (holds ?c ?e) (not (holds ?c ?p))
                 (in ?f ?s) (not (st_empty ?s)) (tending ?c ?s))
  )

  ; The held plate's food mixed into what the station holds.
  (:action plate_add_to_station
    :parameters (?c - chef ?s - station ?l - loc ?p - pstate ?f - food ?g - food
                 ?o - food ?e - pstate)
    :precondition (and (reach ?c ?l) (st_at ?s ?l) (holds ?c ?p) (plate_of ?p ?f)
                       (in ?g ?s) (merge_in ?s ?f ?g ?o) (empty_plate ?e))
    :effect (and (holds ?c ?e) (not (holds ?c ?p))
                 (in ?o ?s) (not (in ?g ?s)))
  )

  ; Pick up a pan, pot or bowl (with whatever is in it).
  (:action lift_station_from_counter
    :parameters (?c - chef ?s - station ?l - loc)
    :precondition (and (reach ?c ?l) (hand_empty ?c) (portable ?s) (st_at ?s ?l)
                       (st_on_counter ?s ?l))
    :effect (and (holds_st ?c ?s) (not (hand_empty ?c))
                 (not (st_at ?s ?l)) (not (st_on_counter ?s ?l)) (free ?l))
  )

  (:action lift_station_from_support
    :parameters (?c - chef ?s - station ?l - loc)
    :precondition (and (reach ?c ?l) (hand_empty ?c) (portable ?s) (st_at ?s ?l)
                       (usable ?s))
    :effect (and (holds_st ?c ?s) (not (hand_empty ?c))
                 (not (st_at ?s ?l)) (not (usable ?s)) (base_free ?l))
  )

  (:action set_station_on_support
    :parameters (?c - chef ?s - station ?l - loc)
    :precondition (and (reach ?c ?l) (holds_st ?c ?s) (support_for ?s ?l)
                       (base_free ?l))
    :effect (and (hand_empty ?c) (not (holds_st ?c ?s))
                 (st_at ?s ?l) (usable ?s) (not (base_free ?l)))
  )

  (:action set_station_on_counter
    :parameters (?c - chef ?s - station ?l - loc)
    :precondition (and (reach ?c ?l) (holds_st ?c ?s) (free ?l))
    :effect (and (hand_empty ?c) (not (holds_st ?c ?s))
                 (st_at ?s ?l) (st_on_counter ?s ?l) (not (free ?l)))
  )

  ; =========================================================================
  ; Bins, sinks and delivery
  ; =========================================================================

  (:action discard_food
    :parameters (?c - chef ?l - loc ?f - food)
    :precondition (and (reach ?c ?l) (bin ?l) (holds ?c ?f))
    :effect (and (hand_empty ?c) (not (holds ?c ?f)))
  )

  (:action discard_from_plate
    :parameters (?c - chef ?l - loc ?p - pstate ?f - food ?e - pstate)
    :precondition (and (reach ?c ?l) (bin ?l) (holds ?c ?p) (plate_of ?p ?f)
                       (empty_plate ?e))
    :effect (and (holds ?c ?e) (not (holds ?c ?p)))
  )

  (:action wash
    :parameters (?c - chef ?l - loc ?p - pstate ?e - pstate)
    :precondition (and (reach ?c ?l) (sink ?l) (holds ?c ?p) (dirty_plate ?p)
                       (empty_plate ?e))
    :effect (and (holds ?c ?e) (not (holds ?c ?p)))
  )

  ; Serving an order. The used plate goes back to the first dispenser, or
  ; disappears when the kitchen has none. ?o2 is the order that opens next
  ; when orders must be served in sequence.
  (:action deliver
    :parameters (?c - chef ?l - loc ?p - pstate ?o - order ?o2 - order)
    :precondition (and (reach ?c ?l) (delivery ?l) (holds ?c ?p) (wants ?o ?p)
                       (open ?o) (next_order ?o ?o2) (returns_nowhere))
    :effect (and (hand_empty ?c) (not (holds ?c ?p))
                 (served ?o) (not (open ?o)) (open ?o2))
  )

  (:action deliver_return_clean
    :parameters (?c - chef ?l - loc ?p - pstate ?o - order ?o2 - order
                 ?d - loc ?n - count ?m - count)
    :precondition (and (reach ?c ?l) (delivery ?l) (holds ?c ?p) (wants ?o ?p)
                       (open ?o) (next_order ?o ?o2)
                       (returns_clean ?d) (clean_n ?d ?n) (succ ?n ?m))
    :effect (and (hand_empty ?c) (not (holds ?c ?p))
                 (served ?o) (not (open ?o)) (open ?o2)
                 (clean_n ?d ?m) (not (clean_n ?d ?n)))
  )

  (:action deliver_return_dirty
    :parameters (?c - chef ?l - loc ?p - pstate ?o - order ?o2 - order
                 ?d - loc ?n - count ?m - count)
    :precondition (and (reach ?c ?l) (delivery ?l) (holds ?c ?p) (wants ?o ?p)
                       (open ?o) (next_order ?o ?o2)
                       (returns_dirty ?d) (dirty_n ?d ?n) (succ ?n ?m))
    :effect (and (hand_empty ?c) (not (holds ?c ?p))
                 (served ?o) (not (open ?o)) (open ?o2)
                 (dirty_n ?d ?m) (not (dirty_n ?d ?n)))
  )
)
