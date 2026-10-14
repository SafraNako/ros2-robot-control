"""Pure-pursuit path following (Coulter, 1992) and the path helpers it needs.

The robot steers along the circle through its own position, tangent to its heading, that passes through the point of the path one lookahead
distance ahead: curvature ``kappa = 2 y / L^2`` with ``y`` the lateral offset of that point. On a circular path of radius ``R`` the robot's own
circle *is* the path, so the steady-state tracking error is zero for any lookahead (the tests check that, and the decay of an initial offset).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import to_body


def line_path(x0, y0, x1, y1, step=0.05):
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step) + 1)
    return np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])


def circle_path(cx, cy, radius, step=0.05, turns=1.0, start_angle=0.0):
    n = max(8, int(2 * math.pi * radius * turns / step))
    a = start_angle + np.linspace(0.0, 2 * math.pi * turns, n)
    return np.column_stack([cx + radius * np.cos(a), cy + radius * np.sin(a)])


def figure_eight_path(width=2.0, height=1.2, step=0.05):
    n = int(2 * math.pi * max(width, height) * 1.6 / step)
    t = np.linspace(0.0, 2 * math.pi, n)
    return np.column_stack([width * np.sin(t), height * np.sin(t) * np.cos(t)])


def path_length(path):
    path = np.asarray(path, float)
    return float(np.sum(np.hypot(*np.diff(path, axis=0).T)))


def cross_track_error(path, x, y):
    """Distance from ``(x, y)`` to the polyline ``path`` (distance to the nearest segment, not only to its vertices)."""
    p = np.asarray(path, float)
    a, b = p[:-1], p[1:]
    ab = b - a
    ap = np.array([x, y]) - a
    denom = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18)
    t = np.clip(np.einsum("ij,ij->i", ap, ab) / denom, 0.0, 1.0)
    proj = a + ab * t[:, None]
    return float(np.min(np.hypot(*(proj - np.array([x, y])).T)))


def _segment_circle(a, b, c, radius):
    """Parameters ``t`` in [0, 1] where the segment a->b meets the circle (centre c, given radius), the later one first."""
    d = b - a
    f = a - c
    qa = d @ d
    if qa < 1e-18:
        return []
    qb = 2 * f @ d
    qc = f @ f - radius * radius
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        return []
    root = math.sqrt(disc)
    ts = [(-qb + root) / (2 * qa), (-qb - root) / (2 * qa)]
    return [t for t in ts if 0.0 <= t <= 1.0]


def _along(path, start, distance):
    """The point ``distance`` of arc length beyond vertex ``start`` (the last point if the path ends first)."""
    seg = np.hypot(*np.diff(path[start:], axis=0).T)
    cum = np.cumsum(seg)
    k = int(np.searchsorted(cum, distance))
    if k >= len(seg):
        return path[-1]
    a, b = path[start + k], path[start + k + 1]
    before = cum[k] - seg[k]
    return a + (b - a) * ((distance - before) / max(seg[k], 1e-18))


@dataclass
class PurePursuit:
    lookahead: float = 0.5
    v_nominal: float = 0.4
    v_min: float = 0.05
    w_max: float = 1.5
    a_lat_max: float = 0.5          # lateral acceleration allowed in a curve (m/s^2)
    decel: float = 0.4              # braking towards the end of the path (m/s^2)
    goal_tol: float = 0.05

    def __post_init__(self):
        self.index = 0              # progress along the path: the search never goes back
        self._acquired = False

    def reset(self):
        self.index = 0
        self._acquired = False

    def _closest(self, path, x, y):
        """Advance the progress index to the closest vertex.

        The first call looks along the whole path; afterwards only a stretch of arc length ``max(3 lookahead, 1 m)`` ahead is searched, so a point of the
        path that happens to be close in space but far along it (the crossing of a figure eight, the second lap of a circle) cannot be jumped to.
        """
        if not self._acquired:
            self.index = int(np.argmin(np.hypot(path[:, 0] - x, path[:, 1] - y)))
            self._acquired = True
            return
        window = path[self.index:self.index + 400]
        reach = max(3.0 * self.lookahead, 1.0)
        window = window[:2 + int(np.searchsorted(np.cumsum(np.hypot(*np.diff(window, axis=0).T)), reach))]
        self.index += int(np.argmin(np.hypot(window[:, 0] - x, window[:, 1] - y)))

    def lookahead_point(self, path, x, y):
        """First point of the path, beyond the closest one (progress is advanced to it first), at distance ``lookahead`` from the robot.

        When the robot is farther from the path than ``lookahead`` the circle meets nothing; then the point one ``lookahead`` of arc length beyond the
        closest vertex is used (aiming at the far end of a long path would make the correction vanishingly small).
        """
        self._closest(path, x, y)
        return self._lookahead(path, x, y)

    def _lookahead(self, path, x, y):
        c = np.array([x, y])
        for i in range(self.index, len(path) - 1):
            hits = _segment_circle(path[i], path[i + 1], c, self.lookahead)
            if hits:
                t = max(hits)
                return path[i] + (path[i + 1] - path[i]) * t
        return _along(path, self.index, self.lookahead)

    def command(self, pose, path):
        """``(v, w, done)`` for the robot at ``pose`` following ``path`` (an array of points)."""
        path = np.asarray(path, float)
        x, y, th = pose
        self._closest(path, x, y)
        left = float(np.sum(np.hypot(*np.diff(path[self.index:], axis=0).T)))      # path length still to go (not the straight line: a loop ends where it starts)
        if math.hypot(path[-1][0] - x, path[-1][1] - y) < self.goal_tol and left <= max(self.lookahead, 2.0 * self.goal_tol):
            return 0.0, 0.0, True
        px, py = self._lookahead(path, x, y)
        bx, by = to_body(x, y, th, px, py)
        dist2 = float(bx * bx + by * by)
        kappa = 2.0 * float(by) / max(dist2, 1e-9)
        if bx < 0:                               # the point is behind: turn on the spot towards it
            return 0.0, math.copysign(self.w_max, by if by != 0 else 1.0), False
        v = self.v_nominal
        if abs(kappa) > 1e-6:
            v = min(v, math.sqrt(self.a_lat_max / abs(kappa)))
        v = min(v, math.sqrt(2.0 * self.decel * max(left - self.goal_tol, 0.0)))               # constant deceleration, ending at the goal tolerance
        v = max(v, self.v_min)
        w = kappa * v
        if abs(w) > self.w_max:
            v, w = v * self.w_max / abs(w), math.copysign(self.w_max, w)
        return v, w, False
