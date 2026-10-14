"""Velocity limits: a hard clamp that keeps the curvature, and an acceleration limiter."""
from __future__ import annotations

from dataclasses import dataclass


def clamp_twist(v, w, v_max, w_max):
    """Scale ``(v, w)`` down, both by the same factor, until ``|v| <= v_max`` and ``|w| <= w_max`` (the curvature ``w / v`` is kept)."""
    k = 1.0
    if abs(v) > v_max:
        k = min(k, v_max / abs(v))
    if abs(w) > w_max:
        k = min(k, w_max / abs(w))
    return v * k, w * k


def _slew(current, target, max_step):
    delta = target - current
    if delta > max_step:
        return current + max_step
    if delta < -max_step:
        return current - max_step
    return target


@dataclass
class VelocityLimiter:
    """Clamp a command to the speed limits and move towards it no faster than the acceleration limits allow."""

    v_max: float = 0.6
    w_max: float = 1.5
    a_max: float = 0.8              # m/s^2
    alpha_max: float = 3.0          # rad/s^2
    v_min: float = -0.3             # the robot may drive backwards, more slowly
    v: float = 0.0
    w: float = 0.0

    def reset(self, v=0.0, w=0.0):
        self.v, self.w = v, w

    def apply(self, v_cmd, w_cmd, dt):
        """The command the robot is allowed to execute during the next ``dt`` seconds; also stored as the current state."""
        v_t, w_t = clamp_twist(v_cmd, w_cmd, self.v_max, self.w_max)
        v_t = max(v_t, self.v_min)
        self.v = _slew(self.v, v_t, self.a_max * dt)
        self.w = _slew(self.w, w_t, self.alpha_max * dt)
        return self.v, self.w
