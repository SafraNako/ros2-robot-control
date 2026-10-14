"""Figures of the simulated runs (matplotlib is imported lazily, so the core and the ROS nodes do not need it)."""
from __future__ import annotations

import math

import numpy as np

from . import kinematics as K
from . import scenarios as SC
from .go_to_goal import GoToGoal


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_world(ax, world):
    for cx, cy, r in world.circles:
        ax.add_patch(_plt().Circle((cx, cy), r, color="0.55", zorder=1))
    for x0, y0, x1, y1 in world.segments:
        ax.plot([x0, x1], [y0, y1], color="0.2", lw=2, zorder=1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def _draw_robot(ax, pose, color, radius=0.18):
    x, y, th = pose
    ax.add_patch(_plt().Circle((x, y), radius, fill=False, color=color, lw=1.5, zorder=4))
    ax.plot([x, x + radius * math.cos(th)], [y, y + radius * math.sin(th)], color=color, lw=1.5, zorder=4)


def plot_go_to_goal(run, goal, path, title="Go to goal"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    _draw_world(ax, run.world)
    ax.plot(run.pose[:, 0], run.pose[:, 1], color="C0", lw=2, label="path driven", zorder=3)
    _draw_robot(ax, run.pose[0], "C2")
    _draw_robot(ax, run.final, "C0")
    ax.plot(*goal[:2], "r*", ms=14, zorder=5, label="goal")
    ax.annotate("", xy=(goal[0] + 0.4 * math.cos(goal[2]), goal[1] + 0.4 * math.sin(goal[2])), xytext=goal[:2], arrowprops=dict(arrowstyle="->", color="r"))
    ax.set_title(f"{title}: {'reached' if run.done else 'collided' if run.collided else 'not reached'} in {run.duration:.1f} s")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pure_pursuit(run, route, path, title="Pure pursuit"):
    plt = _plt()
    err = SC.cross_track(run, route) * 1000
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax.plot(route[:, 0], route[:, 1], "k--", lw=1.5, label="path")
    ax.plot(run.pose[:, 0], run.pose[:, 1], color="C0", lw=2, label="robot")
    _draw_robot(ax, run.pose[0], "C2")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title(title)
    bx.plot(run.t, err, color="C3")
    bx.set_xlabel("time (s)")
    bx.set_ylabel("cross-track error (mm)")
    bx.grid(alpha=0.25)
    bx.set_title(f"max {err.max():.1f} mm, finished in {run.duration:.1f} s")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def find_instructive_room(seed=100, tries=60):
    """The first random room in which the unguarded robot runs into a pillar while the guarded one does not (and the guarded one does move)."""
    for k in range(tries):
        rng = np.random.default_rng(seed + k)
        world = SC.random_world(rng)
        s, g = SC.random_free_point(world, rng), SC.random_free_point(world, rng)
        start, goal = (s[0], s[1], float(rng.uniform(-math.pi, math.pi))), (g[0], g[1], 0.0)
        kwargs = dict(world=world, dt=0.05, t_max=60.0, controller=GoToGoal(allow_reverse=False))
        bare = SC.run_go_to_goal(start, goal, safety=False, **kwargs)
        guarded = SC.run_go_to_goal(start, goal, safety=True, **kwargs)
        hit_pillar = SC.World(circles=world.circles).distance_to_obstacles(bare.final[0], bare.final[1]) < 0.18
        if bare.collided and hit_pillar and not guarded.collided and guarded.limited_steps > 0:
            return world, start, goal, bare, guarded
    raise RuntimeError("no instructive room found")


def plot_safety(path):
    plt = _plt()
    world, start, goal, bare, guarded = find_instructive_room()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, run, name, color in ((axes[0], bare, "without safety layer", "C3"), (axes[1], guarded, "with scan safety", "C0")):
        _draw_world(ax, world)
        ax.plot(run.pose[:, 0], run.pose[:, 1], color=color, lw=2, zorder=3)
        _draw_robot(ax, start, "C2")
        _draw_robot(ax, run.final, color)
        ax.plot(*goal[:2], "r*", ms=14, zorder=5)
        ax.set_title(f"{name}: " + ("COLLISION" if run.collided else "reached" if run.done else "stopped short") + f" at {run.duration:.1f} s")
    fig.suptitle("Same room, same start and goal, forward-only go-to-goal with no obstacle awareness")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pid(path):
    plt = _plt()
    plain, guarded = SC.pid_step_response(False), SC.pid_step_response(True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
    for r, label, color in ((plain, "no anti-windup", "C3"), (guarded, "conditional integration", "C0")):
        ax.plot(r["t"], r["y"], color=color, label=f"{label} (overshoot {max(r['overshoot'], 0):.2f})")
        bx.plot(r["t"], r["u"], color=color, label=label)
    ax.axhline(1.5, color="k", ls=":", label="setpoint")
    ax.set_xlim(0, 12)
    bx.set_xlim(0, 12)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("plant output")
    bx.set_xlabel("time (s)")
    bx.set_ylabel("controller output (saturates at 1)")
    ax.legend()
    ax.grid(alpha=0.25)
    bx.grid(alpha=0.25)
    fig.suptitle("PI control of a first-order plant with a saturating input")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_integrators(path):
    plt = _plt()
    ref = (0.5 / 0.8 * math.sin(2.4), 0.5 / 0.8 * (1 - math.cos(2.4)))
    ns = np.array([5, 10, 20, 50, 100, 200, 500, 1000])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for step, name, color in ((K.integrate_euler, "Euler", "C3"), (K.integrate_midpoint, "midpoint", "C1"), (K.integrate_exact, "exact arc", "C0")):
        err = []
        for n in ns:
            pose = (0.0, 0.0, 0.0)
            for _ in range(n):
                pose = step(pose, 0.5, 0.8, 3.0 / n)
            err.append(max(math.hypot(pose[0] - ref[0], pose[1] - ref[1]), 1e-17))
        ax.loglog(3.0 / ns, err, "o-", color=color, label=name)
    dt = 3.0 / ns
    ax.loglog(dt, 0.467 * dt, "k:", lw=1, label="slope 1")
    ax.loglog(dt, 0.031 * dt ** 2, "k--", lw=1, label="slope 2")
    ax.set_xlabel("step dt (s)")
    ax.set_ylabel("position error after 3 s (m)")
    ax.set_title("Dead reckoning of a constant (v, w): error vs step")
    ax.legend()
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
