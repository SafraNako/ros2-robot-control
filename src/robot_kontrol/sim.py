"""A small deterministic simulator: a differential-drive robot with motor lag in a plane with circular and wall obstacles, and a lidar.

It is the test bench of the controllers (closed-loop checks) and the engine of the ROS simulation node. The pose is advanced on the exact arc of the
*actual* velocities, which follow the commanded ones through first-order motor dynamics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import wrap
from .kinematics import integrate_exact


@dataclass
class World:
    """Obstacles: ``circles`` as ``(cx, cy, r)`` and ``segments`` as ``(x0, y0, x1, y1)`` (walls)."""

    circles: list = field(default_factory=list)
    segments: list = field(default_factory=list)

    def distance_to_obstacles(self, x, y):
        """Distance from ``(x, y)`` to the nearest obstacle surface (negative inside a circle)."""
        best = math.inf
        for cx, cy, r in self.circles:
            best = min(best, math.hypot(x - cx, y - cy) - r)
        for x0, y0, x1, y1 in self.segments:
            ab = np.array([x1 - x0, y1 - y0])
            ap = np.array([x - x0, y - y0])
            t = min(1.0, max(0.0, float(ap @ ab) / max(float(ab @ ab), 1e-18)))
            best = min(best, float(np.hypot(*(ap - ab * t))))
        return best

    def raycast(self, x, y, angles, max_range):
        """Range along each ray (angles in the world frame) to the first obstacle; ``inf`` where nothing is hit within ``max_range``."""
        angles = np.asarray(angles, float)
        dx, dy = np.cos(angles), np.sin(angles)
        best = np.full(len(angles), np.inf)
        for cx, cy, r in self.circles:
            fx, fy = x - cx, y - cy
            b = fx * dx + fy * dy
            c = fx * fx + fy * fy - r * r
            disc = b * b - c
            with np.errstate(invalid="ignore"):
                t = np.where(disc >= 0, -b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
            t = np.where(t >= 0, t, np.inf)
            best = np.minimum(best, t)
        for x0, y0, x1, y1 in self.segments:
            ex, ey = x1 - x0, y1 - y0
            denom = dx * ey - dy * ex
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((x0 - x) * ey - (y0 - y) * ex) / denom
                u = ((x0 - x) * dy - (y0 - y) * dx) / denom
            ok = (np.abs(denom) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)
            best = np.minimum(best, np.where(ok, t, np.inf))
        best[best > max_range] = np.inf
        return best


def world_from_flat(circles=(), walls=()):
    """A ``World`` from flat number lists, the shape ROS parameters have: ``circles`` is ``cx, cy, r, ...`` and ``walls`` is ``x0, y0, x1, y1, ...``.

    Circles with a radius of zero or less are ignored (ROS cannot declare an empty numeric array, so ``[0, 0, 0]`` stands for "none").
    """
    circles, walls = list(circles), list(walls)
    if len(circles) % 3 or len(walls) % 4:
        raise ValueError("circles need 3 numbers each (cx, cy, r) and walls 4 each (x0, y0, x1, y1)")
    return World(
        circles=[tuple(circles[i:i + 3]) for i in range(0, len(circles), 3) if circles[i + 2] > 0],
        segments=[tuple(walls[i:i + 4]) for i in range(0, len(walls), 4) if (walls[i], walls[i + 1]) != (walls[i + 2], walls[i + 3])],
    )


# a small demo room: 10 m x 10 m with two pillars (circles are cx, cy, r; walls are x0, y0, x1, y1), the default world of the CLI and of the ROS sim node
DEMO_CIRCLES = [2.0, 1.0, 0.4, -1.5, 2.5, 0.5]
DEMO_WALLS = [-5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, 5.0, 5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0, -5.0]


def demo_world():
    return world_from_flat(DEMO_CIRCLES, DEMO_WALLS)


@dataclass
class DiffDriveSim:
    """State ``(x, y, theta)`` and the actual speeds ``(v, w)``, which follow the commands with time constants ``tau_v`` and ``tau_w``."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    w: float = 0.0
    tau_v: float = 0.15
    tau_w: float = 0.10
    time: float = 0.0
    radius: float = 0.18
    mean_twist: tuple = (0.0, 0.0)      # the average (v, w) over the last step

    def step(self, v_cmd, w_cmd, dt):
        """Advance by ``dt`` seconds. The lag is solved exactly: the speeds at the end of the step and their averages over it are closed forms, and the
        pose moves along the exact arc of the averages, so the result does not depend on ``dt`` being small (straight motion is exact for any ``dt``)."""
        v_end, v_mean = self._lag(self.v, v_cmd, self.tau_v, dt)
        w_end, w_mean = self._lag(self.w, w_cmd, self.tau_w, dt)
        self.x, self.y, self.theta = integrate_exact((self.x, self.y, self.theta), v_mean, w_mean, dt)
        self.theta = wrap(self.theta)
        self.v, self.w = v_end, w_end
        self.mean_twist = (v_mean, w_mean)
        self.time += dt

    @staticmethod
    def _lag(current, command, tau, dt):
        """End value and step average of a first-order response ``current -> command`` with time constant ``tau``."""
        if tau <= 0:
            return command, command
        decay = math.exp(-dt / tau)
        gap = current - command
        return command + gap * decay, command + gap * tau * (1.0 - decay) / dt

    @property
    def pose(self):
        return (self.x, self.y, self.theta)

    def scan(self, world, n=360, fov=2 * math.pi, max_range=6.0):
        """A lidar scan in the robot frame: ``(angle_min, angle_increment, ranges)`` with ``inf`` for no return."""
        increment = fov / n
        angles = -fov / 2 + increment * np.arange(n)
        return -fov / 2, increment, world.raycast(self.x, self.y, angles + self.theta, max_range)

    def collided(self, world):
        return world.distance_to_obstacles(self.x, self.y) < self.radius
