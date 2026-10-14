"""Plane geometry the controllers share: angle wrapping, quaternions about the vertical axis, frame changes."""
from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap(angle):
    """Wrap an angle (or array) to ``(-pi, pi]``."""
    a = (np.asarray(angle, dtype=float) + math.pi) % TWO_PI - math.pi
    a = np.where(a <= -math.pi, a + TWO_PI, a)
    return float(a) if np.ndim(angle) == 0 else a


def angle_diff(a, b):
    """The signed shortest rotation taking ``b`` to ``a``."""
    return wrap(np.asarray(a, float) - np.asarray(b, float))


def yaw_to_quaternion(yaw):
    """``(x, y, z, w)`` of a rotation by ``yaw`` about the z axis (the order ROS messages use)."""
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def quaternion_to_yaw(x, y, z, w):
    """Yaw of a general unit quaternion (roll and pitch are ignored)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def to_body(x, y, theta, px, py):
    """Coordinates of the world points ``(px, py)`` in the frame of a robot at ``(x, y, theta)``."""
    dx, dy = np.asarray(px, float) - x, np.asarray(py, float) - y
    c, s = math.cos(theta), math.sin(theta)
    return c * dx + s * dy, -s * dx + c * dy


def to_world(x, y, theta, bx, by):
    """World coordinates of points given in the frame of a robot at ``(x, y, theta)``."""
    c, s = math.cos(theta), math.sin(theta)
    bx, by = np.asarray(bx, float), np.asarray(by, float)
    return x + c * bx - s * by, y + s * bx + c * by
