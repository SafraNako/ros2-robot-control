import math

import numpy as np
import pytest

from robot_kontrol import safety as sf

RADIUS = 0.22


def brute_force_free_distance(px, py, v, w, radius, horizon, step=1e-3):
    """Drive the disc along the commanded arc in 1 mm steps and report the first arc length at which a point is within the radius."""
    kappa = w / v
    s = 0.0
    while s <= horizon:
        if abs(kappa) < 1e-6:
            cx, cy = s, 0.0
        else:
            cx, cy = math.sin(kappa * s) / kappa, (1 - math.cos(kappa * s)) / kappa
        if np.any(np.hypot(px - cx, py - cy) <= radius):
            return s
        s += step
    return horizon


def test_scan_points_drops_invalid_returns_and_converts_polar_to_cartesian():
    x, y = sf.scan_points(0.0, math.pi / 2, [1.0, 2.0, math.inf, math.nan, 0.01, 99.0, 0.0], range_min=0.05, range_max=10.0)
    assert x == pytest.approx([1.0, 0.0], abs=1e-12) and y == pytest.approx([0.0, 2.0])


def test_straight_free_distance_is_the_contact_distance():
    assert sf.free_distance([2.0], [0.0], 0.3, 0.0, RADIUS, 3.0) == pytest.approx(2.0 - RADIUS)
    off = 0.1                                                     # offset to the side: the disc touches before the centre line reaches the point
    assert sf.free_distance([2.0], [off], 0.3, 0.0, RADIUS, 3.0) == pytest.approx(2.0 - math.sqrt(RADIUS ** 2 - off ** 2))
    assert sf.free_distance([2.0], [0.3], 0.3, 0.0, RADIUS, 3.0) == 3.0          # outside the corridor
    assert sf.free_distance([-2.0], [0.0], 0.3, 0.0, RADIUS, 3.0) == 3.0         # behind
    assert sf.free_distance([2.0, 1.0, 5.0], [0.0, 0.0, 0.0], 0.3, 0.0, RADIUS, 3.0) == pytest.approx(1.0 - RADIUS)   # the nearest of several


def test_a_point_already_inside_the_footprint_gives_zero_and_no_forward_command_gives_the_horizon():
    assert sf.free_distance([0.1], [0.1], 0.3, 0.5, RADIUS, 3.0) == 0.0
    assert sf.free_distance([0.1], [0.1], 0.3, 0.0, RADIUS, 3.0) == 0.0
    assert sf.free_distance([1.0], [0.0], 0.0, 1.0, RADIUS, 3.0) == 3.0          # turning on the spot is not limited
    assert sf.free_distance([1.0], [0.0], -0.3, 0.0, RADIUS, 3.0) == 3.0         # neither is reversing
    assert sf.free_distance([], [], 0.3, 0.0, RADIUS, 3.0) == 3.0


def test_arc_free_distance_closed_form_for_a_point_on_the_path():
    # a point on the centre line of an arc of radius R at angle phi; the disc touches it when the chord is `radius`: delta = 2 asin(radius / 2R)
    for R in (0.8, 1.5, 4.0):
        for phi in (0.6, 1.2, 2.0):
            px, py = R * math.sin(phi), R * (1 - math.cos(phi))
            expected = R * (phi - 2 * math.asin(RADIUS / (2 * R)))
            assert sf.free_distance([px], [py], 0.4, 0.4 / R, RADIUS, 20.0) == pytest.approx(expected, abs=1e-9)
            assert sf.free_distance([px], [-py], 0.4, -0.4 / R, RADIUS, 20.0) == pytest.approx(expected, abs=1e-9)   # mirrored


def test_matches_brute_force_stepping_on_random_scenes():
    rng = np.random.default_rng(1)
    for _ in range(300):
        m = int(rng.integers(1, 8))
        px, py = rng.uniform(-1, 4, m), rng.uniform(-3, 3, m)
        v = rng.uniform(0.05, 0.6)
        w = rng.uniform(-1.5, 1.5) * rng.choice([0.0, 1.0, 1.0, 1.0])
        fast = sf.free_distance(px, py, v, w, RADIUS, 3.0)
        slow = brute_force_free_distance(px, py, v, w, RADIUS, 3.0)
        assert abs(fast - slow) < 2e-3                            # the brute force steps in 1 mm


def test_mirror_symmetry_and_monotonicity_in_the_radius():
    rng = np.random.default_rng(2)
    for _ in range(200):
        px, py = rng.uniform(-1, 4, 5), rng.uniform(-3, 3, 5)
        v, w = rng.uniform(0.05, 0.6), rng.uniform(-1.5, 1.5)
        assert sf.free_distance(px, -py, v, -w, RADIUS, 3.0) == pytest.approx(sf.free_distance(px, py, v, w, RADIUS, 3.0), abs=1e-9)
        assert sf.free_distance(px, py, v, w, 0.3, 3.0) <= sf.free_distance(px, py, v, w, 0.2, 3.0) + 1e-12


def test_safe_speed_is_the_speed_that_stops_exactly_in_the_free_room():
    guard = sf.ScanSafety(decel=0.8, margin=0.1)
    for free in (0.3, 1.0, 2.5):
        v = guard.safe_speed(free)
        assert v * v / (2 * 0.8) == pytest.approx(free - 0.1)       # braking distance v^2 / 2a equals the room left
    assert guard.safe_speed(0.1) == 0.0 and guard.safe_speed(0.0) == 0.0


def test_limit_scales_the_command_keeping_its_curvature():
    guard = sf.ScanSafety()
    R, phi = 2.0, 0.25                                            # a point on the arc of radius 2 that (0.6, 0.3) follows, 0.25 rad along it
    px, py = np.array([R * math.sin(phi)]), np.array([R * (1 - math.cos(phi))])
    v, w, free, limited = guard.limit(0.6, 0.3, px, py)
    assert free == pytest.approx(R * (phi - 2 * math.asin(guard.radius / (2 * R))))
    assert limited and v < 0.6 and w / v == pytest.approx(0.3 / 0.6)
    assert v == pytest.approx(guard.safe_speed(free))
    assert not guard.limit(0.6, 0.3, np.array([1.0]), np.array([0.0]))[3]          # (1, 0) lies 0.236 m off that arc, more than the 0.22 m radius
    clear = guard.limit(0.2, 0.1, np.array([5.0]), np.array([0.0]))
    assert clear[:2] == (0.2, 0.1) and clear[3] is False
    assert guard.limit(-0.2, 0.4, px, py) == (-0.2, 0.4, guard.horizon, False)   # reversing passes untouched
    assert guard.limit(0.0, 0.9, px, py)[:2] == (0.0, 0.9)                      # so does turning on the spot
