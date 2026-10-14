import math

import numpy as np
import pytest

from robot_kontrol import geometry as geo


def test_wrap_range_and_boundaries():
    assert geo.wrap(math.pi) == pytest.approx(math.pi)
    assert geo.wrap(-math.pi) == pytest.approx(math.pi)         # the interval is (-pi, pi]
    assert geo.wrap(3 * math.pi) == pytest.approx(math.pi)
    assert geo.wrap(0.0) == 0.0
    a = np.random.default_rng(0).uniform(-50, 50, 5000)
    w = geo.wrap(a)
    assert np.all(w > -math.pi) and np.all(w <= math.pi)
    # the wrapped angle differs from the original by a whole number of turns
    k = (a - w) / (2 * math.pi)
    assert np.allclose(k, np.round(k), atol=1e-9)


def test_angle_diff_is_the_shortest_rotation():
    assert geo.angle_diff(0.1, 2 * math.pi - 0.1) == pytest.approx(0.2)
    assert geo.angle_diff(2 * math.pi - 0.1, 0.1) == pytest.approx(-0.2)
    assert geo.angle_diff(1.0, 1.0) == 0.0


def test_yaw_quaternion_round_trip():
    for yaw in np.linspace(-3.1, 3.1, 41):
        x, y, z, w = geo.yaw_to_quaternion(yaw)
        assert (x, y) == (0.0, 0.0)
        assert x * x + y * y + z * z + w * w == pytest.approx(1.0)
        assert geo.quaternion_to_yaw(x, y, z, w) == pytest.approx(yaw)


def euler_to_quaternion(roll, pitch, yaw):
    """The standard ZYX (yaw-pitch-roll) conversion, written out independently of the module under test."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2))
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def test_yaw_is_recovered_from_a_tilted_pose():
    rng = np.random.default_rng(1)
    for _ in range(200):
        roll, pitch, yaw = rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(-3.1, 3.1)
        assert geo.quaternion_to_yaw(*euler_to_quaternion(roll, pitch, yaw)) == pytest.approx(yaw, abs=1e-9)


def test_body_and_world_frames_are_inverse():
    rng = np.random.default_rng(2)
    for _ in range(100):
        x, y, th = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-3, 3)
        px, py = rng.uniform(-5, 5, 6), rng.uniform(-5, 5, 6)
        bx, by = geo.to_body(x, y, th, px, py)
        wx, wy = geo.to_world(x, y, th, bx, by)
        assert np.allclose(wx, px) and np.allclose(wy, py)
        # rotations and translations keep distances
        assert np.allclose(np.hypot(bx, by), np.hypot(px - x, py - y))


def test_a_point_straight_ahead_has_body_x_equal_to_its_distance():
    bx, by = geo.to_body(1.0, 2.0, math.pi / 2, 1.0, 5.0)       # robot faces +y, the point is 3 m ahead
    assert bx == pytest.approx(3.0) and by == pytest.approx(0.0, abs=1e-12)
    bx, by = geo.to_body(0.0, 0.0, 0.0, 0.0, 1.0)               # a point to the left has positive body y
    assert (bx, by) == (pytest.approx(0.0, abs=1e-12), pytest.approx(1.0))
