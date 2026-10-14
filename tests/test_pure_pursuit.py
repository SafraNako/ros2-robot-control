import math

import numpy as np
import pytest

from robot_kontrol import pure_pursuit as pp
from robot_kontrol import scenarios as sc


def test_path_helpers():
    assert pp.path_length(pp.line_path(0, 0, 3, 4)) == pytest.approx(5.0)
    circle = pp.circle_path(1.0, -1.0, 2.0, step=0.01)
    assert np.allclose(np.hypot(circle[:, 0] - 1.0, circle[:, 1] + 1.0), 2.0)
    assert np.allclose(circle[0], circle[-1])
    assert pp.path_length(circle) == pytest.approx(2 * math.pi * 2.0, rel=1e-4)


def test_cross_track_error_uses_segments_not_just_vertices():
    line = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert pp.cross_track_error(line, 0.5, 0.2) == pytest.approx(0.2)
    assert pp.cross_track_error(line, -0.3, 0.4) == pytest.approx(0.5)     # beyond the end: distance to the end point
    assert pp.cross_track_error(line, 0.5, 0.0) == pytest.approx(0.0)


def test_lookahead_point_is_where_the_lookahead_circle_meets_the_path():
    line = pp.line_path(-1.0, 0.3, 6.0, 0.3)
    ctl = pp.PurePursuit(lookahead=1.0)
    px, py = ctl.lookahead_point(line, 0.0, 0.0)
    assert py == pytest.approx(0.3) and px == pytest.approx(math.sqrt(1 - 0.09), abs=1e-9)
    assert pp.PurePursuit(lookahead=50.0).lookahead_point(line, 0.0, 0.0) == pytest.approx(line[-1])   # past the end of the path


def test_lookahead_falls_back_to_arc_length_when_the_robot_is_far_from_the_path():
    line = pp.line_path(0, 0, 30, 0)
    ctl = pp.PurePursuit(lookahead=0.5)
    assert ctl.lookahead_point(line, 10.0, 0.6) == pytest.approx((10.5, 0.0), abs=0.06)   # 0.6 > 0.5: the circle misses the line


def test_commanded_curvature_is_two_lateral_offset_over_lookahead_squared():
    line = pp.line_path(-1.0, 0.3, 6.0, 0.3)
    ctl = pp.PurePursuit(lookahead=1.0, v_nominal=0.2, a_lat_max=100.0)
    v, w, done = ctl.command((0.0, 0.0, 0.0), line)
    assert not done and v == pytest.approx(0.2)
    assert w / v == pytest.approx(2 * 0.3 / 1.0 ** 2)            # kappa = 2 y / L^2 with y the lateral offset of the lookahead point


def test_a_point_behind_makes_the_robot_turn_on_the_spot():
    line = pp.line_path(0, 0, 6, 0)
    v, w, done = pp.PurePursuit(w_max=1.2).command((3.0, 0.0, math.pi), line)
    assert v == 0.0 and abs(w) == pytest.approx(1.2) and not done


def test_speed_is_limited_by_lateral_acceleration_in_curves():
    ctl = pp.PurePursuit(lookahead=0.5, v_nominal=1.0, a_lat_max=0.5, w_max=10.0)
    circle = pp.circle_path(0, 0, 1.0, turns=1.0)
    v, w, _ = ctl.command((1.0, 0.0, math.pi / 2), circle)       # on the path, tangent: kappa = 1 / R = 1
    assert v == pytest.approx(math.sqrt(0.5 / 1.0), rel=1e-3)    # (the path is a 0.05 m polyline, hence not 1e-6)
    assert v * v * (w / v) == pytest.approx(0.5, rel=1e-3)       # lateral acceleration v^2 kappa = a_lat_max


def test_circle_is_followed_with_zero_steady_state_error():
    # on a circle the robot's own circle coincides with the path for any lookahead; what is left is the 0.05 m polyline chord and, with the motor lag,
    # a start-up transient (the speed and turn rate lag by different time constants, so w / v is off for the first seconds)
    R = 2.0
    path = pp.circle_path(0, 0, R, turns=1.0)
    ideal = sc.run_pure_pursuit(path, (R, 0, math.pi / 2), ideal=True, controller=pp.PurePursuit(lookahead=0.6, v_nominal=0.3))
    assert ideal.done and sc.cross_track(ideal, path).max() < 5e-4
    lagged = sc.run_pure_pursuit(path, (R, 0, math.pi / 2), controller=pp.PurePursuit(lookahead=0.6, v_nominal=0.3))
    err = sc.cross_track(lagged, path)
    assert lagged.done and err.max() < 1e-2                       # measured 0.0058, at t = 2 s
    assert err[lagged.t >= 10.0].max() < 5e-4


def test_an_initial_offset_decays_on_a_circle_and_a_line():
    R = 2.0
    path = pp.circle_path(0, 0, R, turns=2.0)
    run = sc.run_pure_pursuit(path, (R + 0.5, 0, math.pi / 2), ideal=True, controller=pp.PurePursuit(lookahead=0.6, v_nominal=0.3))
    err = sc.cross_track(run, path)
    n = len(err)
    assert err[0] == pytest.approx(0.5, abs=1e-3)
    assert err[n // 4:n // 2].max() < 0.003 < 0.5 and err[n // 2:].max() < 2e-4
    line = pp.line_path(0, 0, 30, 0)
    for lookahead in (0.5, 1.0):                                 # 0.5 < the 0.6 m offset: this is the arc-length fallback
        run = sc.run_pure_pursuit(line, (0, 0.6, 0.0), ideal=True, controller=pp.PurePursuit(lookahead=lookahead, v_nominal=0.4))
        err = sc.cross_track(run, line)
        assert run.done and err[500] < 0.02 and err[1000] < 5e-4   # after 10 s and 20 s (the 0.6 m offset is gone)
        assert err[300:].max() < 0.06 < err[0]                   # and it may cross the line only slightly on the way (<= 10 % of the offset)


def test_figure_eight_crossing_does_not_make_it_skip_a_loop():
    # the path starts, crosses itself and ends at the origin; a search that looks too far along the path jumps from the crossing to the end
    path = pp.figure_eight_path()
    run = sc.run_pure_pursuit(path, (0, 0, 0.6), controller=pp.PurePursuit(lookahead=0.5, v_nominal=0.35))
    assert run.done
    assert sc.cross_track(run, path).max() < 0.06
    assert math.hypot(*(run.pose[-1, :2] - path[-1])) < 0.06
    travelled = float(np.sum(np.hypot(*np.diff(run.pose[:, :2], axis=0).T)))
    assert travelled == pytest.approx(pp.path_length(path), rel=0.03)    # both loops were driven (a skipped loop would give half)
    assert run.duration > 0.9 * pp.path_length(path) / 0.35


def test_a_double_lap_is_driven_twice():
    path = pp.circle_path(0, 0, 2.0, turns=2.0)
    run = sc.run_pure_pursuit(path, (2.0, 0, math.pi / 2), ideal=True, controller=pp.PurePursuit(lookahead=0.6, v_nominal=0.3))
    assert run.done
    assert float(np.sum(np.hypot(*np.diff(run.pose[:, :2], axis=0).T))) == pytest.approx(pp.path_length(path), rel=0.02)


def test_braking_profile_is_constant_deceleration_ending_at_the_goal_tolerance():
    line = pp.line_path(0, 0, 5, 0)
    ctl = pp.PurePursuit(v_nominal=0.4, decel=0.4, goal_tol=0.05, v_min=0.05)
    v, _, done = ctl.command((4.85, 0.0, 0.0), line)             # 0.15 m of path left, 0.10 m beyond the tolerance
    assert not done and v == pytest.approx(math.sqrt(2 * 0.4 * (0.15 - 0.05)), rel=1e-6)
    ctl.reset()
    assert ctl.command((4.0, 0.0, 0.0), line)[0] == pytest.approx(0.4)    # far from the end: the nominal speed


def test_stops_at_the_end_without_overshooting_it():
    line = pp.line_path(0, 0, 5, 0)
    run = sc.run_pure_pursuit(line, (0, 0, 0))
    assert run.done and math.hypot(run.final[0] - 5, run.final[1]) < 0.06
    assert run.pose[:, 0].max() <= 5.0 and run.v[-1] < 0.25      # braked from 0.4 m/s; the motor lag leaves some speed at the tolerance, the command then drops to 0


def test_progress_starts_at_the_closest_point_and_a_robot_at_the_end_is_done():
    line = pp.line_path(0, 0, 5, 0)
    assert pp.PurePursuit().command((5.0, 0.0, 0.0), line)[2] is True     # standing on the end of an open path
    loop = pp.circle_path(0, 0, 1.0, turns=1.0)                    # a loop ends where it starts: standing there is the start, not the end
    ctl = pp.PurePursuit()
    assert ctl.command((1.0, 0.0, math.pi / 2), loop)[2] is False and ctl.index == 0
