"""Scan-based speed limiting: how far the robot can drive on its commanded arc before a lidar point touches it, and the speed it can afford.

The robot is a disc of radius ``radius`` centred on the scan origin. For a command ``(v, w)`` with ``v > 0`` its centre moves on a circular arc of
curvature ``w / v`` (a straight line when ``w`` is tiny); ``free_distance`` is the arc length until the disc first touches a scan point, found in
closed form. The speed it may use is the one it can still stop from within that distance at the braking deceleration ``decel``, with a ``margin``.
Reversing and turning on the spot are not limited (there is no rear scan, and a disc turning in place sweeps nothing new).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


def scan_points(angle_min, angle_increment, ranges, range_min=0.0, range_max=math.inf):
    """Cartesian scan points ``(x, y)`` in the sensor frame from the fields of a ``sensor_msgs/LaserScan``; invalid and out-of-range returns are dropped."""
    r = np.asarray(ranges, float)
    angles = angle_min + angle_increment * np.arange(len(r))
    ok = np.isfinite(r) & (r >= range_min) & (r <= range_max) & (r > 0)
    return r[ok] * np.cos(angles[ok]), r[ok] * np.sin(angles[ok])


def free_distance(points_x, points_y, v, w, radius, horizon=math.inf):
    """Arc length (m) the robot centre can travel at ``(v, w)`` before the disc of the given radius touches a point; ``horizon`` if none is hit.

    Returns 0 when a point is already within ``radius`` of the centre.
    """
    px, py = np.asarray(points_x, float), np.asarray(points_y, float)
    if len(px) == 0 or v <= 0:
        return horizon
    kappa = w / v
    if abs(kappa) < 1e-6:                          # straight: a corridor of half-width radius ahead
        inside = (np.abs(py) <= radius) & (px >= 0)
        starts_in = np.hypot(px, py) <= radius
        if starts_in.any():
            return 0.0
        if not inside.any():
            return horizon
        return float(min(horizon, np.min(px[inside] - np.sqrt(np.maximum(radius * radius - py[inside] ** 2, 0.0)))))
    s = 1.0 if kappa > 0 else -1.0
    big_r = 1.0 / abs(kappa)
    cy = s * big_r
    dx, dy = px, py - cy
    dc = np.hypot(dx, dy)
    if np.any(np.hypot(px, py) <= radius):
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        c0 = (dc * dc + big_r * big_r - radius * radius) / (2.0 * big_r * dc)
    reachable = np.isfinite(c0) & (c0 <= 1.0) & (c0 > -1.0)
    if not reachable.any():
        return horizon
    delta = np.arccos(np.clip(c0[reachable], -1.0, 1.0))
    gamma0 = -s * math.pi / 2.0
    gamma_p = np.arctan2(dy[reachable], dx[reachable])
    center = (s * (gamma_p - gamma0)) % TWO_PI
    start = center - delta
    hit = np.where((start <= 0.0) | (center + delta >= TWO_PI), 0.0, start)
    return float(min(horizon, np.min(hit) * big_r))


@dataclass
class ScanSafety:
    radius: float = 0.22            # robot footprint (disc) radius plus a little clearance
    decel: float = 0.8              # braking deceleration the robot can really deliver (m/s^2)
    margin: float = 0.10            # distance left between the stopped robot and the obstacle
    horizon: float = 3.0            # the distance beyond which nothing matters

    def safe_speed(self, free):
        """The largest forward speed from which the robot can still stop within ``free - margin``."""
        room = free - self.margin
        return 0.0 if room <= 0 else math.sqrt(2.0 * self.decel * room)

    def limit(self, v, w, points_x, points_y):
        """``(v, w, free, limited)``: the command scaled down (curvature kept) to the speed the scan allows."""
        if v <= 0:
            return v, w, self.horizon, False
        free = free_distance(points_x, points_y, v, w, self.radius, self.horizon)
        allowed = self.safe_speed(free)
        if v <= allowed:
            return v, w, free, False
        k = allowed / v
        return allowed, w * k, free, True
