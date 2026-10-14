"""Differential-drive kinematics and odometry.

A pose is ``(x, y, theta)``; a body twist ``(v, w)`` is forward speed (m/s) and turn rate (rad/s). For constant ``(v, w)`` the motion is
an exact circular arc, and ``integrate_exact`` follows it (``integrate_euler`` is the first-order alternative the tests compare against).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import wrap


@dataclass
class DiffDrive:
    """Two driven wheels of radius ``wheel_radius`` whose contact points are ``track`` apart."""

    wheel_radius: float = 0.05
    track: float = 0.30

    def wheel_speeds(self, v, w):
        """Left and right wheel angular speeds (rad/s) for a body twist."""
        half = 0.5 * self.track * w
        return (v - half) / self.wheel_radius, (v + half) / self.wheel_radius

    def body_twist(self, wl, wr):
        """Body twist ``(v, w)`` of the given wheel angular speeds."""
        vl, vr = wl * self.wheel_radius, wr * self.wheel_radius
        return 0.5 * (vl + vr), (vr - vl) / self.track

    def wheel_travel(self, v, w, dt):
        """Distances (m) the left and right wheels roll in ``dt`` at constant ``(v, w)``."""
        half = 0.5 * self.track * w
        return (v - half) * dt, (v + half) * dt


def integrate_exact(pose, v, w, dt):
    """Pose after ``dt`` at constant ``(v, w)``: the exact circular arc (a straight line when ``w`` is tiny)."""
    x, y, th = pose
    if abs(w) < 1e-9:
        return (x + v * dt * math.cos(th), y + v * dt * math.sin(th), th + w * dt)
    r = v / w
    th2 = th + w * dt
    return (x + r * (math.sin(th2) - math.sin(th)), y - r * (math.cos(th2) - math.cos(th)), th2)


def integrate_euler(pose, v, w, dt):
    """Forward-Euler step (the heading is held at its old value over the step): first order in ``dt``."""
    x, y, th = pose
    return (x + v * dt * math.cos(th), y + v * dt * math.sin(th), th + w * dt)


def integrate_midpoint(pose, v, w, dt):
    """Midpoint step (the heading at the middle of the step is used): second order in ``dt``."""
    x, y, th = pose
    mid = th + 0.5 * w * dt
    return (x + v * dt * math.cos(mid), y + v * dt * math.sin(mid), th + w * dt)


class Odometry:
    """Dead reckoning from wheel travel: each update turns the left and right distances into an exact arc."""

    def __init__(self, track, pose=(0.0, 0.0, 0.0)):
        self.track = track
        self.pose = tuple(pose)

    def update(self, dl, dr):
        """Advance by the distances ``dl`` and ``dr`` (m) the wheels rolled since the last call; returns the new pose."""
        ds = 0.5 * (dl + dr)
        dth = (dr - dl) / self.track
        x, y, th = self.pose
        if abs(dth) < 1e-12:
            x, y = x + ds * math.cos(th), y + ds * math.sin(th)
        else:
            r = ds / dth
            x, y = x + r * (math.sin(th + dth) - math.sin(th)), y - r * (math.cos(th + dth) - math.cos(th))
        self.pose = (x, y, wrap(th + dth))
        return self.pose
