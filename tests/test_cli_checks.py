import numpy as np
import pytest

from robot_kontrol import checks, cli
from robot_kontrol import pure_pursuit as pp
from robot_kontrol import scenarios as sc


def png_ok(path):
    data = path.read_bytes()
    return data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 5000


def test_every_validation_check_passes():
    results = checks.run_all()
    assert len(results) == len(checks.ALL) == 12
    failed = [(c.name, c.detail) for c in results if not c.passed]
    assert not failed, failed


def test_cross_track_is_judged_against_the_branch_being_followed():
    path = pp.figure_eight_path()
    run = sc.run_pure_pursuit(path, (0, 0, 0.6), controller=pp.PurePursuit(lookahead=0.6, v_nominal=0.3))
    aware = sc.cross_track(run, path)
    naive = np.array([pp.cross_track_error(path, x, y) for x, y, _ in run.pose])
    assert np.all(aware >= naive - 1e-12)
    assert np.max(aware - naive) > 0.005                          # measured 10.7 mm: at the crossing the other branch is nearer than the one being followed


def test_cli_commands_and_exit_codes(capsys):
    assert cli.main(["kinematik"]) == 0
    out = capsys.readouterr().out
    assert "Euler" in out and "tam yay" in out
    assert cli.main(["git", "--hedef", "3", "-2", "--aci", "90"]) == 0
    assert "hedefe ulasti" in capsys.readouterr().out
    assert cli.main(["git", "--hedef", "8", "0", "--dunya", "bos", "--guvenliksiz", "--sure", "5"]) == 1      # not there within 5 s
    assert cli.main(["takip", "--yol", "dogru", "--gecikmesiz"]) == 0
    assert "bitti" in capsys.readouterr().out
    assert cli.main(["pid"]) == 0
    assert "anti-windup" in capsys.readouterr().out
    assert cli.main(["ros"]) == 0
    assert "rclpy" in capsys.readouterr().out


def test_cli_safety_comparison_reports_no_collision_with_the_guard(capsys):
    assert cli.main(["guvenlik", "--oda", "6", "--tohum", "100"]) == 0
    out = capsys.readouterr().out
    assert "guvenlikli" in out and " 0 carpisma" in out.split("guvenlikli")[1]


def test_cli_validation_reports_all_passed(capsys):
    assert cli.main(["dogrulama"]) == 0
    assert "12/12 dogrulama gecti" in capsys.readouterr().out


def test_cli_argument_errors():
    with pytest.raises(SystemExit) as e:
        cli.main(["git"])                                          # --hedef is required
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        cli.main([])


def test_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    from robot_kontrol import plots
    plots.plot_integrators(tmp_path / "a.png")
    plots.plot_pid(tmp_path / "b.png")
    route = pp.line_path(0, 0, 4, 0)
    plots.plot_pure_pursuit(sc.run_pure_pursuit(route, (0, 0.3, 0.0), ideal=True), route, tmp_path / "c.png")
    plots.plot_go_to_goal(sc.run_go_to_goal((0, 0, 0), (2, 1, 0), safety=False), (2, 1, 0), tmp_path / "d.png")
    for name in "abcd":
        assert png_ok(tmp_path / f"{name}.png")


def test_the_instructive_room_really_is_one():
    pytest.importorskip("matplotlib")
    from robot_kontrol import plots
    world, start, goal, bare, guarded = plots.find_instructive_room()
    assert bare.collided and not guarded.collided and guarded.limited_steps > 0
