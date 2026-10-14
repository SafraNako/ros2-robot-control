import math

import numpy as np
import pytest

from robot_kontrol import kinematics as K


def analytic_arc(v, w, t):
    """Pose after time t from the origin with heading 0 at constant (v, w), w != 0: the closed form of the circular arc."""
    return (v / w * math.sin(w * t), v / w * (1 - math.cos(w * t)), w * t)


def run(step, v, w, t, n):
    pose = (0.0, 0.0, 0.0)
    for _ in range(n):
        pose = step(pose, v, w, t / n)
    return pose


def position_error(pose, ref):
    return math.hypot(pose[0] - ref[0], pose[1] - ref[1])


def test_wheel_speeds_and_body_twist_are_inverse():
    drive = K.DiffDrive(0.07, 0.4)
    rng = np.random.default_rng(0)
    for _ in range(200):
        v, w = rng.uniform(-1, 1), rng.uniform(-3, 3)
        assert drive.body_twist(*drive.wheel_speeds(v, w)) == pytest.approx((v, w))


def test_wheel_speeds_of_special_motions():
    drive = K.DiffDrive(0.05, 0.30)
    wl, wr = drive.wheel_speeds(0.0, 2.0)                       # turning on the spot: wheels counter-rotate equally
    assert wl == pytest.approx(-wr) and wr == pytest.approx(0.5 * 0.30 * 2.0 / 0.05)
    wl, wr = drive.wheel_speeds(0.5, 0.0)                       # straight: equal wheels, v = r * omega
    assert wl == wr == pytest.approx(0.5 / 0.05)


def test_exact_integrator_follows_the_arc_for_any_step_size():
    ref = analytic_arc(0.5, 0.8, 3.0)
    for n in (1, 3, 10, 1000):
        assert position_error(run(K.integrate_exact, 0.5, 0.8, 3.0, n), ref) < 1e-12
        assert run(K.integrate_exact, 0.5, 0.8, 3.0, n)[2] == pytest.approx(ref[2])
    # a negative turn rate mirrors the arc
    x, y, th = run(K.integrate_exact, 0.5, -0.8, 3.0, 7)
    assert (x, y, th) == pytest.approx((ref[0], -ref[1], -ref[2]))


def test_exact_integrator_is_a_semigroup_and_continuous_at_w_zero():
    one = K.integrate_exact((1.0, -2.0, 0.7), 0.4, 0.9, 2.0)
    two = K.integrate_exact(K.integrate_exact((1.0, -2.0, 0.7), 0.4, 0.9, 1.0), 0.4, 0.9, 1.0)
    assert one == pytest.approx(two)
    assert K.integrate_exact((0, 0, 0), 1.0, 1e-8, 1.0) == pytest.approx(K.integrate_exact((0, 0, 0), 1.0, 0.0, 1.0), abs=1e-7)
    assert K.integrate_exact((0, 0, 0), 1.0, 0.0, 2.0) == (2.0, 0.0, 0.0)


def test_euler_is_first_order_and_midpoint_second_order():
    ref = analytic_arc(0.5, 0.8, 3.0)
    for step, order in ((K.integrate_euler, 1), (K.integrate_midpoint, 2)):
        coarse = position_error(run(step, 0.5, 0.8, 3.0, 50), ref)
        fine = position_error(run(step, 0.5, 0.8, 3.0, 100), ref)
        assert coarse / fine == pytest.approx(2.0 ** order, rel=0.05)
    # and the exact integrator beats both by many orders of magnitude at the same step
    exact = position_error(run(K.integrate_exact, 0.5, 0.8, 3.0, 50), ref)
    assert exact < 1e-9 < position_error(run(K.integrate_midpoint, 0.5, 0.8, 3.0, 50), ref)


def test_full_circle_returns_to_start():
    # one full period at any step count closes for every integrator (the heading samples are equally spaced), so this is a sanity check only
    w = 0.5
    for step in (K.integrate_exact, K.integrate_euler, K.integrate_midpoint):
        x, y, _ = run(step, 0.5, w, 2 * math.pi / w, 64)
        assert math.hypot(x, y) < 1e-9


def test_odometry_reproduces_the_commanded_arc():
    drive = K.DiffDrive()
    odo = K.Odometry(drive.track)
    for _ in range(3000):
        odo.update(*drive.wheel_travel(0.5, 0.8, 0.001))
    ref = analytic_arc(0.5, 0.8, 3.0)
    assert position_error(odo.pose, ref) < 1e-9
    assert odo.pose[2] == pytest.approx(K.wrap(ref[2]))


def test_odometry_special_cases_and_chunking():
    drive = K.DiffDrive(0.05, 0.30)
    spin = K.Odometry(drive.track)
    spin.update(-0.1, 0.1)                                      # on the spot: the position must not move
    assert spin.pose[:2] == pytest.approx((0.0, 0.0), abs=1e-12)
    assert spin.pose[2] == pytest.approx(0.2 / 0.30)
    fine, coarse = K.Odometry(drive.track), K.Odometry(drive.track)
    for _ in range(1000):
        fine.update(*drive.wheel_travel(0.3, -0.5, 0.002))
    coarse.update(*drive.wheel_travel(0.3, -0.5, 2.0))          # a constant twist gives the same pose in one tick or a thousand
    assert fine.pose == pytest.approx(coarse.pose, abs=1e-9)


def test_odometry_heading_stays_wrapped():
    odo = K.Odometry(0.3)
    for _ in range(2000):
        odo.update(-0.01, 0.01)
        assert -math.pi < odo.pose[2] <= math.pi
