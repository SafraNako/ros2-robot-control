import math

import numpy as np
import pytest

from robot_kontrol import scenarios as sc
from robot_kontrol.geometry import wrap
from robot_kontrol.go_to_goal import GoToGoal, gains_are_stable


def path_length(run):
    return float(np.sum(np.hypot(*np.diff(run.pose[:, :2], axis=0).T)))


def test_stability_conditions():
    assert gains_are_stable(0.6, 1.8, -0.5)
    assert not gains_are_stable(-0.1, 1.8, -0.5)                 # k_rho > 0
    assert not gains_are_stable(0.6, 1.8, 0.5)                   # k_beta < 0
    assert not gains_are_stable(0.6, 0.6, -0.5)                  # k_alpha - k_rho > 0 is strict
    assert not gains_are_stable(0.6, 0.5, -0.5)


@pytest.mark.parametrize("gains", [dict(k_rho=-1.0), dict(k_beta=0.3), dict(k_alpha=0.5)])
def test_unstable_gains_are_refused(gains):
    with pytest.raises(ValueError):
        GoToGoal(**gains)


def test_straight_approach_decays_exponentially_at_k_rho():
    # aligned with the goal and its heading: alpha = beta = 0, so v = k_rho * rho and rho(t) = rho0 exp(-k_rho t) (up to the 1 ms Euler step of the loop)
    ctl = GoToGoal(v_max=100.0, w_max=100.0)
    run = sc.run_go_to_goal((0, 0, 0), (4.0, 0, 0), ideal=True, safety=False, dt=0.001, t_max=3.0, controller=ctl)
    rho = 4.0 - run.pose[:, 0]
    assert np.all(np.abs(run.pose[:, 1]) < 1e-12)                # it never leaves the line
    assert rho[-1] == pytest.approx(4.0 * math.exp(-0.6 * run.t[-1]), rel=2e-3)
    assert np.polyfit(run.t[:2000], np.log(rho[:2000]), 1)[0] == pytest.approx(-0.6, rel=1e-2)


def test_converges_from_random_starts_and_the_distance_never_grows():
    rng = np.random.default_rng(3)
    for _ in range(25):
        start = (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-math.pi, math.pi))
        goal = (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-math.pi, math.pi))
        run = sc.run_go_to_goal(start, goal, ideal=True, safety=False, dt=0.01, t_max=240.0)
        assert run.done
        assert math.hypot(run.final[0] - goal[0], run.final[1] - goal[1]) < 0.03
        assert abs(wrap(run.final[2] - goal[2])) < 0.05
        rho = np.hypot(run.pose[:, 0] - goal[0], run.pose[:, 1] - goal[1])
        reached = int(np.argmax(rho < 0.03))
        assert np.max(np.diff(rho[:reached + 1])) <= 1e-12       # rho' = -v cos(alpha) <= 0 while driving


def test_converges_with_motor_lag_and_the_limiter():
    rng = np.random.default_rng(4)
    for _ in range(8):
        start = (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-math.pi, math.pi))
        goal = (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-math.pi, math.pi))
        run = sc.run_go_to_goal(start, goal, safety=False, dt=0.02, t_max=240.0)
        assert run.done and math.hypot(run.final[0] - goal[0], run.final[1] - goal[1]) < 0.03


def test_a_goal_behind_is_reached_in_reverse_along_the_straight_line():
    rev = sc.run_go_to_goal((0, 0, 0), (-3, 0, 0), ideal=True, safety=False, dt=0.01, controller=GoToGoal(allow_reverse=True))
    fwd = sc.run_go_to_goal((0, 0, 0), (-3, 0, 0), ideal=True, safety=False, dt=0.01, controller=GoToGoal(allow_reverse=False))
    assert rev.done and fwd.done
    assert rev.v[1] < 0 < fwd.v[1]
    assert path_length(rev) == pytest.approx(3.0 - 0.03, abs=0.01)   # straight back, stopping at the tolerance
    assert abs(rev.final[2]) < 1e-6                               # and already at the goal heading, no turn needed
    assert path_length(fwd) > 2.5 * path_length(rev)              # forward-only has to loop around
    assert rev.duration < 0.6 * fwd.duration


def test_final_turn_on_the_spot():
    ctl = GoToGoal(k_final=1.5, w_max=1.0, yaw_tol=0.05)
    v, w, done = ctl.command((1.0, 1.0, 0.0), (1.0, 1.0, 1.0))
    assert (v, done) == (0.0, False) and w == pytest.approx(1.0)  # 1.5 * 1.0 clipped to w_max
    v, w, done = ctl.command((1.0, 1.0, 0.8), (1.0, 1.0, 1.0))
    assert (v, done) == (0.0, False) and w == pytest.approx(1.5 * 0.2)
    assert ctl.command((1.0, 1.0, 0.98), (1.0, 1.0, 1.0)) == (0.0, 0.0, True)
    assert ctl.command((1.0, 1.0, math.pi - 0.01), (1.0, 1.0, -math.pi + 0.01))[2] is True   # the heading error is wrapped


def test_commands_respect_the_speed_limits_and_keep_the_curve():
    ctl = GoToGoal(v_max=0.3, w_max=0.8)
    rng = np.random.default_rng(5)
    for _ in range(300):
        pose = (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-3, 3))
        v, w, _ = ctl.command(pose, (0.0, 0.0, 0.0))
        assert abs(v) <= 0.3 + 1e-12 and abs(w) <= 0.8 + 1e-12
