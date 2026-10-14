"""Go to a pose with the polar-coordinate controller of Aicardi, Casalino, Bicchi & Balestrino (1995).

In the frame of the goal, with ``rho`` the distance to it, ``alpha`` the angle between the robot's heading and the line to the goal and
``beta`` the angle of that line in the goal frame, the law ``v = k_rho rho``, ``w = k_alpha alpha + k_beta beta`` drives the pose to the origin
exponentially when ``k_rho > 0``, ``k_beta < 0`` and ``k_alpha - k_rho > 0`` (that is checked, not assumed). A goal behind the robot is
approached in reverse. Speed limits scale ``(v, w)`` together so the curve stays the same. When the position is reached the robot turns on the
spot to the goal heading.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import wrap
from .limits import clamp_twist


def gains_are_stable(k_rho, k_alpha, k_beta):
    """The three conditions that make the closed loop locally exponentially stable."""
    return k_rho > 0 and k_beta < 0 and k_alpha - k_rho > 0


@dataclass
class GoToGoal:
    k_rho: float = 0.6
    k_alpha: float = 1.8
    k_beta: float = -0.5
    k_final: float = 1.5            # gain of the final turn on the spot
    v_max: float = 0.5
    w_max: float = 1.5
    pos_tol: float = 0.03
    yaw_tol: float = 0.05
    allow_reverse: bool = True

    def __post_init__(self):
        if not gains_are_stable(self.k_rho, self.k_alpha, self.k_beta):
            raise ValueError("unstable gains: need k_rho > 0, k_beta < 0 and k_alpha - k_rho > 0")

    def command(self, pose, goal):
        """``(v, w, done)`` for the robot at ``pose = (x, y, theta)`` heading for ``goal = (gx, gy, gtheta)``."""
        x, y, th = pose
        gx, gy, gth = goal
        dx, dy = gx - x, gy - y
        rho = math.hypot(dx, dy)
        if rho < self.pos_tol:
            err = wrap(gth - th)
            if abs(err) < self.yaw_tol:
                return 0.0, 0.0, True
            return 0.0, max(-self.w_max, min(self.w_max, self.k_final * err)), False
        reverse = False
        theta = th
        alpha = wrap(math.atan2(dy, dx) - theta)
        if self.allow_reverse and abs(alpha) > 0.5 * math.pi:
            reverse, theta = True, th + math.pi
            alpha = wrap(math.atan2(dy, dx) - theta)
        # beta = -(heading - goal heading) - alpha; driving in reverse flips the robot frame and the goal frame by pi alike, so the true headings are used
        beta = wrap(-(th - gth) - alpha)
        v = self.k_rho * rho
        w = self.k_alpha * alpha + self.k_beta * beta
        v, w = clamp_twist(v, w, self.v_max, self.w_max)
        return (-v if reverse else v), w, False
