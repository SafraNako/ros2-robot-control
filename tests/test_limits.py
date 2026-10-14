import numpy as np
import pytest

from robot_kontrol.limits import VelocityLimiter, clamp_twist


def test_clamp_keeps_the_curvature_and_respects_the_limits():
    rng = np.random.default_rng(0)
    for _ in range(500):
        v, w = rng.uniform(-3, 3), rng.uniform(-6, 6)
        cv, cw = clamp_twist(v, w, 0.5, 1.5)
        assert abs(cv) <= 0.5 + 1e-12 and abs(cw) <= 1.5 + 1e-12
        if abs(v) > 1e-6:
            assert cw * v == pytest.approx(w * cv)               # same w / v
        if abs(v) <= 0.5 and abs(w) <= 1.5:
            assert (cv, cw) == (v, w)


def test_clamp_examples():
    assert clamp_twist(1.0, 0.0, 0.5, 1.5) == (0.5, 0.0)
    assert clamp_twist(0.2, 3.0, 0.5, 1.5) == pytest.approx((0.1, 1.5))
    assert clamp_twist(-1.0, 2.0, 0.5, 1.5) == pytest.approx((-0.5, 1.0))


def test_limiter_never_exceeds_the_slew_or_speed_limits():
    lim = VelocityLimiter(v_max=0.6, w_max=1.5, a_max=0.8, alpha_max=3.0, v_min=-0.3)
    rng = np.random.default_rng(1)
    dt, prev = 0.02, (0.0, 0.0)
    for _ in range(5000):
        v, w = lim.apply(rng.uniform(-2, 2), rng.uniform(-5, 5), dt)
        assert abs(v - prev[0]) <= 0.8 * dt + 1e-12 and abs(w - prev[1]) <= 3.0 * dt + 1e-12
        assert -0.3 - 1e-12 <= v <= 0.6 + 1e-12 and abs(w) <= 1.5 + 1e-12
        prev = (v, w)


def test_limiter_ramps_at_exactly_the_acceleration_limit():
    lim = VelocityLimiter(v_max=0.6, a_max=0.8)
    t = 0.0
    while lim.v < 0.6 - 1e-12:
        lim.apply(0.6, 0.0, 0.01)
        t += 0.01
    assert t == pytest.approx(0.6 / 0.8, abs=0.011)             # v_max / a_max seconds
    lim.reset()
    assert (lim.v, lim.w) == (0.0, 0.0)


def test_backwards_is_limited_to_v_min():
    lim = VelocityLimiter(v_min=-0.3, a_max=100.0)
    assert lim.apply(-1.0, 0.0, 1.0)[0] == pytest.approx(-0.3)
