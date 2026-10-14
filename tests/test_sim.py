import math

import numpy as np
import pytest

from robot_kontrol import scenarios as sc
from robot_kontrol.sim import DiffDriveSim, World, world_from_flat


def sphere_trace(world, x, y, angle, max_range):
    """Distance to the first obstacle by sphere tracing along the ray (steps of the distance to the nearest surface): independent of ``raycast``."""
    t, dx, dy = 0.0, math.cos(angle), math.sin(angle)
    for _ in range(5000):
        d = world.distance_to_obstacles(x + t * dx, y + t * dy)
        if d < 1e-7:
            return t
        t += d
        if t > max_range:
            return math.inf
    return t


def test_first_order_lag_matches_the_analytic_step_response_for_any_dt():
    tau = 0.15
    for dt in (0.1, 0.01, 0.001):
        sim = DiffDriveSim(tau_v=tau)
        for _ in range(int(round(1.0 / dt))):
            sim.step(0.5, 0.0, dt)
        assert sim.v == pytest.approx(0.5 * (1 - math.exp(-1.0 / tau)), abs=1e-12)
        assert sim.x == pytest.approx(0.5 * (1.0 - tau * (1 - math.exp(-1.0 / tau))), abs=1e-12)   # the integral of v, exact even for dt = 0.1


def test_curved_motion_with_lag_converges_at_second_order():
    def pose(dt):
        sim = DiffDriveSim()
        for _ in range(int(round(3.0 / dt))):
            sim.step(0.5, 0.8, dt)
        return np.array(sim.pose)
    ref = pose(1e-4)
    errors = [np.max(np.abs(pose(dt) - ref)) for dt in (0.1, 0.05, 0.025)]
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.15) and errors[1] / errors[2] == pytest.approx(4.0, rel=0.15)
    assert errors[0] < 1e-3


def test_no_lag_means_the_command_is_executed_at_once():
    sim = DiffDriveSim(tau_v=0.0, tau_w=0.0)
    sim.step(0.5, 0.8, 1.0)
    assert (sim.v, sim.w) == (0.5, 0.8)
    assert sim.pose == pytest.approx((0.5 / 0.8 * math.sin(0.8), 0.5 / 0.8 * (1 - math.cos(0.8)), 0.8))


def test_raycast_known_geometry():
    world = World(circles=[(3.0, 0.0, 1.0)], segments=[(5.0, -5.0, 5.0, 5.0)])
    r = world.raycast(0.0, 0.0, [0.0, math.pi, math.pi / 2, 0.3, 0.45], 10.0)
    assert r[0] == pytest.approx(2.0)                             # the circle, not the wall behind it
    assert math.isinf(r[1]) and math.isinf(r[2])                  # away from everything / parallel to the wall and off its end
    assert r[3] == pytest.approx(3 * math.cos(0.3) - math.sqrt(1 - (3 * math.sin(0.3)) ** 2))   # 3 sin(0.3) = 0.89 < 1: grazes the circle's side
    assert r[4] == pytest.approx(5.0 / math.cos(0.45))            # 3 sin(0.45) = 1.3 > 1: misses the circle, hits the wall
    assert math.isinf(World(segments=[(5.0, -5.0, 5.0, 5.0)]).raycast(0.0, 0.0, [0.0], 4.0)[0])    # beyond max_range


def test_raycast_matches_sphere_tracing_in_random_worlds():
    rng = np.random.default_rng(5)
    hits = 0
    for _ in range(300):
        world = sc.random_world(rng, n_circles=int(rng.integers(1, 8)))
        x, y = sc.random_free_point(world, rng, clearance=0.05)
        angle = rng.uniform(-math.pi, math.pi)
        fast = float(world.raycast(x, y, [angle], 8.0)[0])
        slow = sphere_trace(world, x, y, angle, 8.0)
        assert math.isinf(fast) == math.isinf(slow)
        if not math.isinf(fast):
            hits += 1
            assert fast == pytest.approx(slow, abs=1e-4)
    assert hits > 200


def test_distance_to_obstacles():
    world = World(circles=[(0.0, 0.0, 1.0)], segments=[(3.0, -1.0, 3.0, 1.0)])
    assert world.distance_to_obstacles(2.0, 0.0) == pytest.approx(1.0)       # to the circle (1) and to the wall (1)
    assert world.distance_to_obstacles(0.0, 0.5) == pytest.approx(-0.5)      # inside the circle
    assert World(segments=[(3.0, -1.0, 3.0, 1.0)]).distance_to_obstacles(4.0, 2.0) == pytest.approx(math.sqrt(2.0))   # nearest point is an end point


def test_scan_is_expressed_in_the_robot_frame():
    world = World(segments=[(-5.0, 4.0, 5.0, 4.0)])                # a wall at y = 4
    sim = DiffDriveSim(0.0, 0.0, math.pi / 2)                      # facing it
    a0, inc, ranges = sim.scan(world, n=360, fov=2 * math.pi, max_range=10.0)
    beam = int(round((0.0 - a0) / inc))                            # relative angle 0 = straight ahead
    assert ranges[beam] == pytest.approx(4.0)
    side = int(round((math.pi / 2 - a0) / inc))                    # to the robot's left = world -x: nothing there
    assert math.isinf(ranges[side])


def test_collision_detection_uses_the_robot_radius():
    world = World(circles=[(1.0, 0.0, 0.5)])
    assert not DiffDriveSim(0.0, 0.0, 0.0, radius=0.18).collided(world)      # 0.5 m from the surface
    assert DiffDriveSim(0.4, 0.0, 0.0, radius=0.18).collided(world)          # 0.1 m from the surface < 0.18


def test_world_from_flat_parsing():
    world = world_from_flat([1.0, 2.0, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    assert world.circles == [(1.0, 2.0, 0.5)]                       # the zero-radius circle (ROS "none") and the zero-length wall are dropped
    assert world.segments == [(0.0, 0.0, 1.0, 1.0)]
    for bad in (dict(circles=[1.0, 2.0]), dict(walls=[1.0, 2.0, 3.0])):
        with pytest.raises(ValueError):
            world_from_flat(**bad)
