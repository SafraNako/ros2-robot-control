"""Self-validation: every estimator and controller in the package is checked against something it did not produce itself.

Each check returns a ``Check(name, passed, detail)``. The references are closed forms (the circular arc, the first-order step response, the
steady state of a P-controlled lag), brute-force stepping and sphere tracing for the geometry, a planted exponential rate for go-to-goal, and
a closed-loop experiment in which the same random rooms are run with and without the safety layer. ``robot dogrulama`` prints them all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import kinematics as K
from . import pure_pursuit as PP
from . import safety as S
from . import scenarios as SC
from .geometry import wrap
from .go_to_goal import GoToGoal
from .limits import VelocityLimiter
from .mux import CmdMux
from .pid import PID
from .sim import DiffDriveSim


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _arc(v, w, t):
    return (v / w * math.sin(w * t), v / w * (1 - math.cos(w * t)))


def _run(step, v, w, t, n):
    pose = (0.0, 0.0, 0.0)
    for _ in range(n):
        pose = step(pose, v, w, t / n)
    return pose


def check_integrators():
    ref = _arc(0.5, 0.8, 3.0)
    err = {f.__name__: math.hypot(*(np.array(_run(f, 0.5, 0.8, 3.0, 100)[:2]) - ref)) for f in (K.integrate_exact, K.integrate_euler, K.integrate_midpoint)}
    order = {}
    for f in (K.integrate_euler, K.integrate_midpoint):
        e1 = math.hypot(*(np.array(_run(f, 0.5, 0.8, 3.0, 50)[:2]) - ref))
        e2 = math.hypot(*(np.array(_run(f, 0.5, 0.8, 3.0, 100)[:2]) - ref))
        order[f.__name__] = math.log2(e1 / e2)
    ok = err["integrate_exact"] < 1e-12 and abs(order["integrate_euler"] - 1) < 0.1 and abs(order["integrate_midpoint"] - 2) < 0.1
    return Check("integrators vs the analytic arc", ok,
                 f"exact error {err['integrate_exact']:.1e} m; Euler order {order['integrate_euler']:.2f}, midpoint order {order['integrate_midpoint']:.2f}")


def check_odometry():
    drive = K.DiffDrive()
    odo = K.Odometry(drive.track)
    for _ in range(3000):
        odo.update(*drive.wheel_travel(0.5, 0.8, 0.001))
    ref = _arc(0.5, 0.8, 3.0)
    err = math.hypot(odo.pose[0] - ref[0], odo.pose[1] - ref[1])
    return Check("wheel odometry vs the analytic arc", err < 1e-9 and abs(wrap(odo.pose[2] - 2.4)) < 1e-9, f"position error {err:.1e} m after 3 s")


def check_sim_lag():
    tau, worst = 0.15, 0.0
    for dt in (0.1, 0.01, 0.001):
        sim = DiffDriveSim(tau_v=tau)
        for _ in range(int(round(1.0 / dt))):
            sim.step(0.5, 0.0, dt)
        worst = max(worst, abs(sim.v - 0.5 * (1 - math.exp(-1 / tau))), abs(sim.x - 0.5 * (1 - tau * (1 - math.exp(-1 / tau)))))
    return Check("simulated motor lag vs the first-order step response", worst < 1e-12, f"worst error {worst:.1e} for dt = 0.1, 0.01, 0.001 s")


def check_pid_steady_state():
    kp = 4.0
    pid, y = PID(kp), 0.0
    for _ in range(4000):
        y += 0.005 * (-y + pid.update(1.0, y, 0.005))
    pi, z = PID(2.0, 2.0), 0.0
    for _ in range(8000):
        z += 0.005 * (-z + pi.update(1.0, z, 0.005))
    ok = abs(y - kp / (1 + kp)) < 1e-6 and abs(z - 1.0) < 1e-4
    return Check("PID steady state of a first-order plant", ok, f"P: {y:.6f} (theory {kp / (1 + kp):.6f}); PI: {z:.6f} (theory 1)")


def check_anti_windup():
    plain, guarded = SC.pid_step_response(False), SC.pid_step_response(True)
    ok = plain["overshoot"] > 0.2 and guarded["overshoot"] < 1e-3 and abs(guarded["integral_peak"] - guarded["integral_steady"]) < 1e-3
    return Check("anti-windup removes the overshoot of a saturated rise", ok,
                 f"overshoot {plain['overshoot']:.3f} without, {max(guarded['overshoot'], 0):.3f} with; integral peak {plain['integral_peak']:.3f} vs {guarded['integral_peak']:.3f} "
                 f"(steady {guarded['integral_steady']:.3f})")


def check_go_to_goal(n=25):
    ctl = GoToGoal(v_max=100.0, w_max=100.0)
    run = SC.run_go_to_goal((0, 0, 0), (4.0, 0, 0), ideal=True, safety=False, dt=0.001, t_max=3.0, controller=ctl)
    rate = np.polyfit(run.t[:2000], np.log(4.0 - run.pose[:2000, 0]), 1)[0]
    rng = np.random.default_rng(3)
    failures = 0
    for _ in range(n):
        s = (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-math.pi, math.pi))
        g = (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-math.pi, math.pi))
        r = SC.run_go_to_goal(s, g, ideal=True, safety=False, dt=0.01, t_max=240.0)
        failures += not (r.done and math.hypot(r.final[0] - g[0], r.final[1] - g[1]) < 0.03 and abs(wrap(r.final[2] - g[2])) < 0.05)
    ok = abs(rate + 0.6) < 0.01 and failures == 0
    return Check("go-to-goal: decay rate k_rho and convergence from random poses", ok, f"fitted rate {rate:.4f} (theory -0.6); {n - failures}/{n} runs reached position and heading")


def check_reverse():
    r = SC.run_go_to_goal((0, 0, 0), (-3, 0, 0), ideal=True, safety=False, dt=0.01, controller=GoToGoal(allow_reverse=True))
    length = float(np.sum(np.hypot(*np.diff(r.pose[:, :2], axis=0).T)))
    ok = r.done and abs(length - 2.97) < 0.01 and abs(r.final[2]) < 1e-6 and r.v[1] < 0
    return Check("go-to-goal drives straight back to a goal behind it", ok, f"path {length:.3f} m for a 3 m goal (stops at the 0.03 m tolerance), heading error {abs(r.final[2]):.1e}")


def check_pure_pursuit():
    R = 2.0
    circle = PP.circle_path(0, 0, R, turns=1.0)
    ideal = SC.run_pure_pursuit(circle, (R, 0, math.pi / 2), ideal=True, controller=PP.PurePursuit(lookahead=0.6, v_nominal=0.3))
    e_circle = SC.cross_track(ideal, circle).max()
    eight = PP.figure_eight_path()
    run = SC.run_pure_pursuit(eight, (0, 0, 0.6), controller=PP.PurePursuit(lookahead=0.5, v_nominal=0.35))
    travelled = float(np.sum(np.hypot(*np.diff(run.pose[:, :2], axis=0).T)))
    line = PP.line_path(0, 0, 30, 0)
    off = SC.run_pure_pursuit(line, (0, 0.6, 0.0), ideal=True, controller=PP.PurePursuit(lookahead=0.5, v_nominal=0.4))
    e_line = SC.cross_track(off, line)
    ok = (e_circle < 5e-4 and run.done and abs(travelled / PP.path_length(eight) - 1) < 0.03 and SC.cross_track(run, eight).max() < 0.06
          and e_line[500] < 0.02 and e_line[1000] < 5e-4)
    return Check("pure pursuit: zero error on a circle, both loops of a figure eight, offset decay", ok,
                 f"circle max error {e_circle * 1000:.2f} mm; figure eight {travelled:.2f} of {PP.path_length(eight):.2f} m, max error {SC.cross_track(run, eight).max() * 100:.1f} cm; "
                 f"0.6 m offset -> {e_line[500] * 1000:.2f} mm at 10 s, {e_line[1000] * 1000:.3f} mm at 20 s")


def check_free_distance(n=150):
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(n):
        m = int(rng.integers(1, 8))
        px, py = rng.uniform(-1, 4, m), rng.uniform(-3, 3, m)
        v, w = rng.uniform(0.05, 0.6), rng.uniform(-1.5, 1.5) * rng.choice([0.0, 1.0, 1.0, 1.0])
        fast = S.free_distance(px, py, v, w, 0.22, 3.0)
        kappa, s, slow = w / v, 0.0, 3.0
        while s <= 3.0:
            cx, cy = (s, 0.0) if abs(kappa) < 1e-6 else (math.sin(kappa * s) / kappa, (1 - math.cos(kappa * s)) / kappa)
            if np.any(np.hypot(px - cx, py - cy) <= 0.22):
                slow = s
                break
            s += 2e-3
        worst = max(worst, abs(fast - slow))
    R, phi = 1.5, 1.2
    closed = S.free_distance([R * math.sin(phi)], [R * (1 - math.cos(phi))], 0.4, 0.4 / R, 0.22, 20.0)
    exact = R * (phi - 2 * math.asin(0.22 / (2 * R)))
    return Check("arc free distance vs brute-force stepping and the closed form", worst < 4e-3 and abs(closed - exact) < 1e-9,
                 f"{n} random scenes, worst difference {worst * 1000:.2f} mm (brute-force step 2 mm); closed-form arc error {abs(closed - exact):.1e} m")


def check_raycast(n=200):
    def trace(world, x, y, ang, max_range):
        t, dx, dy = 0.0, math.cos(ang), math.sin(ang)
        for _ in range(5000):
            d = world.distance_to_obstacles(x + t * dx, y + t * dy)
            if d < 1e-7:
                return t
            t += d
            if t > max_range:
                return math.inf
        return t

    rng = np.random.default_rng(5)
    worst, hits, kind = 0.0, 0, 0
    for _ in range(n):
        world = SC.random_world(rng, n_circles=int(rng.integers(1, 8)))
        x, y = SC.random_free_point(world, rng, clearance=0.05)
        a = rng.uniform(-math.pi, math.pi)
        fast, slow = float(world.raycast(x, y, [a], 8.0)[0]), trace(world, x, y, a, 8.0)
        kind += math.isinf(fast) != math.isinf(slow)
        if not math.isinf(fast) and not math.isinf(slow):
            hits += 1
            worst = max(worst, abs(fast - slow))
    return Check("lidar raycast vs sphere tracing", kind == 0 and worst < 1e-4, f"{hits} hits among {n} rays, worst difference {worst:.1e} m, hit/miss disagreements {kind}")


def check_collisions(n=12):
    without = SC.collision_experiment(n, safety=False, seed=100, t_max=60.0)
    with_safety = SC.collision_experiment(n, safety=True, seed=100, t_max=60.0)
    ok = with_safety["collision"] == 0 and without["collision"] >= 3
    return Check("scan safety prevents collisions in random cluttered rooms", ok,
                 f"{n} rooms of 8 pillars, forward-only go-to-goal: without safety {without['collision']} collisions; with safety {with_safety['collision']} collisions, "
                 f"{with_safety['reached']} reached, {with_safety['stalled']} stalled (a reactive layer does not plan around obstacles)")


def check_mux_and_limiter():
    mux = CmdMux()
    mux.add_source("teleop", 10, 0.5)
    mux.add_source("nav", 1, 0.5)
    mux.update("nav", 0.3, 0.0, 0.0)
    mux.update("teleop", 0.1, 0.0, 0.4)
    steps = [mux.select(0.45)[2], mux.select(0.8)[2], mux.select(1.0)[2]]
    lim, worst, prev = VelocityLimiter(), 0.0, (0.0, 0.0)
    rng = np.random.default_rng(1)
    for _ in range(2000):
        v, w = lim.apply(rng.uniform(-2, 2), rng.uniform(-5, 5), 0.02)
        worst = max(worst, abs(v - prev[0]) / 0.02 - 0.8, abs(w - prev[1]) / 0.02 - 3.0)
        prev = (v, w)
    ok = steps == ["teleop", "teleop", None] and worst <= 1e-9
    return Check("command mux priority and watchdog; limiter respects its slew rates", ok, f"selected {steps}; slew-rate excess {max(worst, 0):.1e}")


ALL = [check_integrators, check_odometry, check_sim_lag, check_pid_steady_state, check_anti_windup, check_go_to_goal, check_reverse, check_pure_pursuit,
       check_free_distance, check_raycast, check_collisions, check_mux_and_limiter]


def run_all():
    return [c() for c in ALL]
