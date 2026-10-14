import pytest

from robot_kontrol.mux import CmdMux


def make():
    mux = CmdMux()
    mux.add_source("teleop", 10, 0.5)
    mux.add_source("nav", 1, 0.5)
    return mux


def test_nothing_received_means_stop():
    assert make().select(0.0) == (0.0, 0.0, None)


def test_the_highest_priority_fresh_command_wins():
    mux = make()
    mux.update("nav", 0.3, 0.1, 0.0)
    assert mux.select(0.1) == (0.3, 0.1, "nav")
    mux.update("teleop", -0.2, 0.5, 0.1)
    assert mux.select(0.2) == (-0.2, 0.5, "teleop")


def test_an_expired_source_falls_back_to_the_next_and_then_to_a_stop():
    mux = make()
    mux.update("nav", 0.3, 0.1, 0.0)
    mux.update("teleop", 0.1, 0.0, 0.4)
    assert mux.select(0.8)[2] == "teleop"                         # nav is 0.8 s old (expired), teleop 0.4 s
    assert mux.select(0.95) == (0.3, 0.1, "nav") or mux.select(0.95)[2] is None
    assert mux.select(0.95)[2] is None                            # both expired -> watchdog stop
    assert mux.select(0.95)[:2] == (0.0, 0.0)


def test_timeout_boundary_and_latest_command_per_source():
    mux = make()
    mux.update("nav", 0.3, 0.0, 1.0)
    assert mux.select(1.5)[2] == "nav"                            # exactly at the timeout it is still valid
    assert mux.select(1.5 + 1e-9)[2] is None
    mux.update("nav", 0.4, 0.2, 2.0)
    assert mux.select(2.1) == (0.4, 0.2, "nav")


def test_unknown_source_is_an_error():
    with pytest.raises(KeyError):
        make().update("joystick", 0.0, 0.0, 0.0)
