"""Closed-loop runs of the controllers in the simulator: the one place the loops are written, shared by the CLI, the plots, the checks and the tests.

Every run uses the same chain as the ROS pipeline: controller -> scan safety (optional) -> velocity limiter -> simulated robot with motor lag.
``ideal=True`` removes the motor lag (the robot executes the command at once), which is what the closed-form comparisons need.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .go_to_goal import GoToGoal
from .limits import VelocityLimiter
from .pid import PID
from .pure_pursuit import PurePursuit, cross_track_error
from .safety import ScanSafety, scan_points
from .sim import DiffDriveSim, World


@dataclass
class Run:
    t: np.ndarray
    pose: np.ndarray                 # (n, 3)
    v: np.ndarray                    # actual forward speed
    w: np.ndarray
    done: bool
    collided: bool
    limited_steps: int = 0
    world: World = field(default_factory=World)
    progress: np.ndarray = None      # pure pursuit: the index of the path vertex the controller considered the closest at each sample

    @property
    def final(self):
        return tuple(self.pose[-1])

    @property
    def duration(self):
        return float(self.t[-1])


def _new_sim(start, ideal):
    return DiffDriveSim(*start, tau_v=0.0 if ideal else 0.15, tau_w=0.0 if ideal else 0.10)


def _record(sim, store):
    store.append((sim.time, sim.x, sim.y, sim.theta, sim.v, sim.w))


def _pack(store, done, collided, limited, world, progress=None):
    a = np.array(store)
    return Run(a[:, 0], a[:, 1:4], a[:, 4], a[:, 5], done, collided, limited, world, None if progress is None else np.array(progress))


def run_go_to_goal(start, goal, world=None, safety=True, ideal=False, dt=0.02, t_max=120.0, controller=None, stop_on_collision=True):
    """Drive from ``start`` to ``goal`` (both ``(x, y, theta)``); ``safety`` switches the scan limiter on."""
    world = world if world is not None else World()
    ctl = controller or GoToGoal()
    lim, guard, sim = VelocityLimiter(), ScanSafety(), _new_sim(start, ideal)
    store, done, collided, limited = [], False, False, 0
    _record(sim, store)
    for _ in range(int(t_max / dt)):
        v, w, done = ctl.command(sim.pose, goal)
        if done:
            break
        if safety:
            a0, inc, ranges = sim.scan(world, 360, 2 * math.pi, 6.0)
            v, w, _, was_limited = guard.limit(v, w, *scan_points(a0, inc, ranges, 0.05, 6.0))
            limited += int(was_limited)
        if not ideal:
            v, w = lim.apply(v, w, dt)
        sim.step(v, w, dt)
        _record(sim, store)
        if sim.collided(world):
            collided = True
            if stop_on_collision:
                break
    return _pack(store, done, collided, limited, world)


def run_pure_pursuit(path, start, ideal=False, dt=0.02, t_max=120.0, controller=None):
    """Follow ``path`` from ``start``; the run ends at the end of the path or after ``t_max``."""
    ctl = controller or PurePursuit()
    ctl.reset()
    lim, sim = VelocityLimiter(), _new_sim(start, ideal)
    store, done, progress = [], False, []
    _record(sim, store)
    for _ in range(int(t_max / dt)):
        v, w, done = ctl.command(sim.pose, path)
        progress.append(ctl.index)
        if done:
            break
        if not ideal:
            v, w = lim.apply(v, w, dt)
        sim.step(v, w, dt)
        _record(sim, store)
    progress.append(ctl.index)
    return _pack(store, done, False, 0, World(), progress)


def cross_track(run, path, reach=1.5):
    """Cross-track error of every sample of a run.

    Measured against the part of the path within ``reach`` metres of arc length of where the controller was along it (the ``progress`` of the run), so
    that where the path comes back close to itself (the crossing of a figure eight, the second lap of a circle) the robot is judged against the branch
    it is following and not against the other one.
    """
    path = np.asarray(path, float)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    out = np.empty(len(run.pose))
    for k, (x, y, _) in enumerate(run.pose):
        if run.progress is None:
            out[k] = cross_track_error(path, x, y)
            continue
        c = cum[min(run.progress[k], len(cum) - 1)]
        lo, hi = np.searchsorted(cum, c - reach), np.searchsorted(cum, c + reach) + 1
        out[k] = cross_track_error(path[max(lo - 1, 0):hi + 1], x, y)
    return out


def random_world(rng, n_circles=8, half=5.0):
    """A square room with ``n_circles`` pillars of radius 0.2-0.6."""
    circles = [(rng.uniform(-half + 1, half - 1), rng.uniform(-half + 1, half - 1), rng.uniform(0.2, 0.6)) for _ in range(n_circles)]
    walls = [(-half, -half, half, -half), (half, -half, half, half), (half, half, -half, half), (-half, half, -half, -half)]
    return World(circles, walls)


def random_free_point(world, rng, half=5.0, clearance=0.6):
    while True:
        p = (float(rng.uniform(-half + 0.5, half - 0.5)), float(rng.uniform(-half + 0.5, half - 0.5)))
        if world.distance_to_obstacles(*p) > clearance:
            return p


def collision_experiment(n_worlds, safety, seed=0, t_max=90.0, dt=0.05):
    """Random start and goal in random cluttered rooms, forward-only go-to-goal: counts of ``collision``, ``reached`` and ``stalled`` runs."""
    counts = {"collision": 0, "reached": 0, "stalled": 0}
    for k in range(n_worlds):
        rng = np.random.default_rng(seed + k)
        world = random_world(rng)
        s, g = random_free_point(world, rng), random_free_point(world, rng)
        start = (s[0], s[1], float(rng.uniform(-math.pi, math.pi)))
        run = run_go_to_goal(start, (g[0], g[1], 0.0), world, safety=safety, dt=dt, t_max=t_max, controller=GoToGoal(allow_reverse=False))
        counts["collision" if run.collided else "reached" if run.done else "stalled"] += 1
    return counts


def pid_step_response(anti_windup, kp=2.0, ki=2.0, plant_gain=2.0, plant_tau=1.0, u_max=1.0, setpoint=1.5, t_max=30.0, dt=0.005):
    """A PI loop around the first-order plant ``y' = (-y + plant_gain u) / plant_tau`` whose input saturates at ``+-u_max``.

    The plant can reach ``plant_gain u_max`` = 2 but only ``setpoint / plant_gain`` = 0.75 of the input is needed to hold the setpoint, so the input
    saturates during the rise and the integral can wind up. In the steady state the integral must equal ``u_ss / ki`` with ``u_ss = setpoint / plant_gain``.
    """
    pid = PID(kp, ki, 0.0, -u_max, u_max, anti_windup=anti_windup)
    y, ts, ys, us, integrals = 0.0, [], [], [], []
    for k in range(int(t_max / dt)):
        u = pid.update(setpoint, y, dt)
        y += dt * (-y + plant_gain * u) / plant_tau
        ts.append((k + 1) * dt), ys.append(y), us.append(u), integrals.append(pid.integral)
    ys = np.array(ys)
    return {"t": np.array(ts), "y": ys, "u": np.array(us), "overshoot": float(ys.max() - setpoint), "final": float(ys[-1]),
            "integral_final": float(pid.integral), "integral_peak": float(max(integrals)), "integral_steady": setpoint / plant_gain / ki}
