import math

import pytest

from robot_kontrol.pid import PID
from robot_kontrol.scenarios import pid_step_response


def test_proportional_only():
    pid = PID(2.0)
    assert pid.update(1.0, 0.25, 0.01) == pytest.approx(1.5)


def test_integral_accumulates_error_times_time():
    pid = PID(0.0, ki=3.0)
    for _ in range(100):
        out = pid.update(1.0, 0.5, 0.01)                         # constant error 0.5 for 1 s
    assert pid.integral == pytest.approx(0.5)
    assert out == pytest.approx(3.0 * 0.5)


def test_derivative_acts_on_the_measurement_so_a_setpoint_jump_does_not_kick():
    a, b = PID(1.0, 0.0, 0.5), PID(1.0, 0.0, 0.5)
    for _ in range(50):
        a.update(0.0, 0.2, 0.01)
        b.update(0.0, 0.2, 0.01)
    jumped, steady = a.update(5.0, 0.2, 0.01), b.update(0.0, 0.2, 0.01)
    assert jumped - steady == pytest.approx(5.0)                 # only the proportional term sees the jump


def test_filtered_derivative_of_a_ramp_converges_to_the_slope():
    pid = PID(0.0, 0.0, 1.0, tau_d=0.05)
    rate, dt, out = 2.0, 0.001, 0.0
    for k in range(1000):                                        # 20 filter time constants
        out = pid.update(0.0, rate * k * dt, dt)
    assert out == pytest.approx(-1.0 * rate, rel=0.01)


def test_first_order_plant_p_control_has_the_textbook_steady_state_error():
    # y' = (-y + u) / T under u = kp (r - y): y_ss = kp / (1 + kp) r
    kp, r = 4.0, 1.0
    pid, y = PID(kp), 0.0
    for _ in range(4000):
        y += 0.005 * (-y + pid.update(r, y, 0.005))
    assert y == pytest.approx(kp / (1 + kp) * r, abs=1e-6)


def test_pi_control_removes_the_steady_state_error():
    pid, y = PID(2.0, 2.0), 0.0
    for _ in range(8000):
        y += 0.005 * (-y + pid.update(1.0, y, 0.005))
    assert y == pytest.approx(1.0, abs=1e-4)


def test_output_is_always_within_its_limits():
    pid = PID(10.0, 5.0, 1.0, -0.7, 0.9)
    for k in range(500):
        assert -0.7 <= pid.update(math.sin(k / 10) * 5, 0.0, 0.01) <= 0.9


def test_anti_windup_removes_the_overshoot_after_a_saturated_rise():
    plain, guarded = pid_step_response(False), pid_step_response(True)
    assert plain["overshoot"] > 0.2                              # measured 0.25: the integral wound up while the input saturated
    assert guarded["overshoot"] < 1e-3
    assert plain["integral_peak"] > 2 * plain["integral_steady"]
    assert guarded["integral_peak"] == pytest.approx(guarded["integral_steady"], rel=1e-3)
    for r in (plain, guarded):                                   # both settle at the setpoint, with the integral at u_ss / ki
        assert r["final"] == pytest.approx(1.5, abs=1e-4)
        assert r["integral_final"] == pytest.approx(r["integral_steady"], rel=1e-3)


def test_bad_dt_is_rejected():
    for dt in (0.0, -0.01):
        with pytest.raises(ValueError):
            PID(1.0).update(1.0, 0.0, dt)
