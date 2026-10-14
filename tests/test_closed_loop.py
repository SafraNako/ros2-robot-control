import math

import pytest

from robot_kontrol import scenarios as sc
from robot_kontrol.sim import World


def test_safety_layer_prevents_every_collision_in_random_cluttered_rooms():
    # 20 rooms with 8 pillars, random start/goal, a goal controller with no obstacle awareness at all (forward only)
    without = sc.collision_experiment(20, safety=False, seed=100, t_max=60.0)
    with_safety = sc.collision_experiment(20, safety=True, seed=100, t_max=60.0)
    assert without["collision"] >= 5                              # the experiment is meaningful: unguarded, a good fraction of the runs hit something
    assert with_safety["collision"] == 0
    assert with_safety["reached"] >= 5                            # and the guarded robot still gets to many goals
    assert sum(with_safety.values()) == 20


def test_charging_a_wall_at_full_speed_stops_short_without_contact():
    world = World(segments=[(4.0, -5.0, 4.0, 5.0)])
    run = sc.run_go_to_goal((0, 0, 0), (8.0, 0, 0), world, safety=True, dt=0.02, t_max=40.0)
    assert not run.collided and run.v.max() > 0.45                # it did go fast
    gap = 4.0 - run.pose[:, 0].max() - 0.18                       # wall to the edge of the 0.18 m disc
    assert 0.0 < gap < 0.2                                        # measured 0.06 m: stops close, but not touching
    assert abs(run.v[-1]) < 1e-3                                  # and ends at rest


def test_without_safety_the_same_charge_hits_the_wall():
    world = World(segments=[(4.0, -5.0, 4.0, 5.0)])
    assert sc.run_go_to_goal((0, 0, 0), (8.0, 0, 0), world, safety=False, dt=0.02, t_max=40.0).collided


def test_a_blocked_robot_can_still_reverse_and_turn():
    world = World(segments=[(4.0, -5.0, 4.0, 5.0)])
    run = sc.run_go_to_goal((0, 0, 0), (8.0, 0, 0), world, safety=True, dt=0.02, t_max=30.0)
    assert run.limited_steps > 0
    # from where it stopped, a goal behind it is reached in reverse straight away (reversing is never limited)
    back = sc.run_go_to_goal(run.final, (run.final[0] - 2.0, 0.0, run.final[2]), world, safety=True, dt=0.02, t_max=60.0)
    assert back.done and not back.collided


def test_the_guard_does_not_hold_back_a_robot_in_free_space():
    run = sc.run_go_to_goal((0, 0, 0), (3.0, 2.0, 0.0), World(), safety=True, dt=0.02)
    free = sc.run_go_to_goal((0, 0, 0), (3.0, 2.0, 0.0), World(), safety=False, dt=0.02)
    assert run.done and run.limited_steps == 0
    assert run.duration == pytest.approx(free.duration)
    assert math.hypot(run.final[0] - 3.0, run.final[1] - 2.0) < 0.03
