"""Integration tests over real DDS: every node runs as its own process (``python -m robot_kontrol.ros.<name>``) and the tests talk to them with an
rclpy node of their own, the way any other ROS 2 program would. Skipped wherever rclpy cannot be imported (a plain pip environment, macOS)."""
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("rclpy")
pytestmark = pytest.mark.ros

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseStamped, Twist  # noqa: E402
from nav_msgs.msg import Odometry, Path as PathMsg  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402
from std_msgs.msg import Bool, Float32, String  # noqa: E402

from robot_kontrol.geometry import quaternion_to_yaw, wrap  # noqa: E402

SRC = str(Path(__file__).resolve().parents[1] / "src")
NODE_NAMES = {"sim_node": "robot_sim", "mux_node": "robot_mux", "safety_node": "robot_safety",
              "go_to_goal_node": "robot_go_to_goal", "pure_pursuit_node": "robot_pure_pursuit"}
NO_CIRCLES = [0.0, 0.0, 0.0]            # ROS cannot declare an empty numeric array: a zero-radius circle means "none"
NO_WALLS = [0.0, 0.0, 0.0, 0.0]         # and a zero-length wall means "none"


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    # an isolated DDS domain per test session, so that nothing else on the network or machine can interfere
    os.environ["ROS_DOMAIN_ID"] = str(random.randint(30, 99))
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    rclpy.init()
    yield
    rclpy.shutdown()


def _yaml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_yaml_value(x) for x in v) + "]"
    return repr(float(v)) if isinstance(v, (int, float)) else f'"{v}"'


class Harness(Node):
    """The test's own node: records what the system publishes and publishes commands, goals and paths into it."""

    def __init__(self):
        super().__init__("test_harness")
        self.last, self.log, self.pubs = {}, {}, {}
        for topic, typ in (("odom", Odometry), ("scan", LaserScan), ("collision", Bool), ("goal_reached", Bool), ("active_source", String),
                           ("cmd_vel_muxed", Twist), ("cmd_vel", Twist), ("free_distance", Float32)):
            self.log[topic] = []
            self.create_subscription(typ, topic, lambda m, t=topic: (self.last.__setitem__(t, m), self.log[t].append(m)), 10)
        for topic, typ in (("cmd_vel_teleop", Twist), ("cmd_vel_nav", Twist), ("cmd_vel_muxed", Twist), ("goal_pose", PoseStamped), ("path", PathMsg)):
            self.pubs[topic] = self.create_publisher(typ, topic, 10)

    def spin_until(self, predicate, timeout):
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
            if predicate():
                return True
        return False

    def spin_for(self, seconds):
        self.spin_until(lambda: False, seconds)

    def drive(self, topic, v, w, seconds, rate=20.0):
        """Publish the command ``(v, w)`` at ``rate`` Hz for ``seconds`` while spinning."""
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        end = time.time() + seconds
        while time.time() < end:
            self.pubs[topic].publish(msg)
            self.spin_for(1.0 / rate)

    def pose(self):
        p = self.last["odom"].pose.pose
        return p.position.x, p.position.y, quaternion_to_yaw(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)

    def collided(self):
        return any(m.data for m in self.log["collision"])

    def wait_for_nodes(self, names, timeout=30.0):
        return self.spin_until(lambda: all(n in [x for x in self.get_node_names()] for n in names), timeout)


class Stack:
    """Starts node processes with a parameter file and stops them at the end."""

    def __init__(self, tmp_path):
        self.tmp, self.procs = tmp_path, {}
        self.env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))

    def start(self, module, params=None):
        params_file = self.tmp / f"{module}.yaml"
        body = "".join(f"      {k}: {_yaml_value(v)}\n" for k, v in (params or {}).items())
        params_file.write_text(f"{NODE_NAMES[module]}:\n  ros__parameters:\n{body or '      rate: 20.0' + chr(10)}")
        log = open(self.tmp / f"{module}.log", "w")
        self.procs[module] = subprocess.Popen([sys.executable, "-m", f"robot_kontrol.ros.{module}", "--ros-args", "--params-file", str(params_file)],
                                              env=self.env, stdout=log, stderr=subprocess.STDOUT)
        return self.procs[module]

    def log(self, module):
        return (self.tmp / f"{module}.log").read_text()

    def stop(self):
        for p in self.procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in self.procs.values():
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()


@pytest.fixture
def stack(tmp_path):
    s = Stack(tmp_path)
    yield s
    s.stop()


@pytest.fixture
def harness():
    h = Harness()
    yield h
    h.destroy_node()


def bring_up(stack, harness, nodes):
    for module, params in nodes.items():
        stack.start(module, params)
    names = [NODE_NAMES[m] for m in nodes]
    assert harness.wait_for_nodes(names), "nodes did not come up:\n" + "\n".join(stack.log(m) for m in nodes)


def test_scan_and_odometry_describe_the_world(stack, harness):
    bring_up(stack, harness, {"sim_node": dict(circles=NO_CIRCLES, walls=[3.0, -5.0, 3.0, 5.0], start_pose=[0.5, 0.0, 0.0])})
    assert harness.spin_until(lambda: "scan" in harness.last and "odom" in harness.last, 15)
    scan = harness.last["scan"]
    n = len(scan.ranges)
    assert n == 360 and scan.angle_min == pytest.approx(-math.pi)
    assert scan.angle_max == pytest.approx(scan.angle_min + scan.angle_increment * (n - 1))
    ahead = scan.ranges[int(round((0.0 - scan.angle_min) / scan.angle_increment))]
    assert ahead == pytest.approx(2.5, abs=1e-6)                   # wall at x = 3, robot at x = 0.5, looking along +x
    assert math.isinf(scan.ranges[int(round((math.pi - scan.angle_min) / scan.angle_increment)) % n]) or scan.ranges[0] > 5.0   # nothing behind within range
    x, y, th = harness.pose()
    assert (x, y, th) == pytest.approx((0.5, 0.0, 0.0), abs=1e-6)
    assert harness.last["odom"].header.frame_id == "odom" and harness.last["odom"].child_frame_id == "base_link"


def test_the_base_stops_when_commands_stop(stack, harness):
    bring_up(stack, harness, {"sim_node": dict(circles=NO_CIRCLES, walls=NO_WALLS, cmd_timeout=0.5)})
    pub = harness.create_publisher(Twist, "cmd_vel", 10)
    assert harness.spin_until(lambda: pub.get_subscription_count() > 0, 15)
    msg = Twist()
    msg.linear.x = 0.3
    end = time.time() + 1.5
    while time.time() < end:
        pub.publish(msg)
        harness.spin_for(0.05)
    assert harness.last["odom"].twist.twist.linear.x > 0.25        # it is moving
    harness.spin_for(2.0)                                          # silence: the 0.5 s command timeout, then the motor lag
    assert abs(harness.last["odom"].twist.twist.linear.x) < 1e-3


def test_go_to_goal_pipeline_reaches_the_goal(stack, harness):
    bring_up(stack, harness, {"sim_node": {}, "mux_node": {}, "safety_node": {}, "go_to_goal_node": {}})
    assert harness.spin_until(lambda: "odom" in harness.last and harness.pubs["goal_pose"].get_subscription_count() > 0, 20)
    goal = PoseStamped()
    goal.pose.position.x, goal.pose.position.y = 3.0, -2.0
    goal.pose.orientation.z, goal.pose.orientation.w = math.sin(math.pi / 4), math.cos(math.pi / 4)      # facing +y
    harness.pubs["goal_pose"].publish(goal)
    reached = harness.spin_until(lambda: harness.last.get("goal_reached") is not None and harness.last["goal_reached"].data, 60)
    assert reached, f"goal not reached, pose {harness.pose()}"
    x, y, th = harness.pose()
    assert math.hypot(x - 3.0, y + 2.0) < 0.05 and abs(wrap(th - math.pi / 2)) < 0.08
    assert not harness.collided()
    assert any(m.data == "nav" for m in harness.log["active_source"])      # the mux passed the navigation command through


def test_safety_stops_before_a_wall_and_still_lets_the_robot_reverse(stack, harness):
    bring_up(stack, harness, {"sim_node": dict(circles=NO_CIRCLES, walls=[4.0, -5.0, 4.0, 5.0]), "mux_node": {}, "safety_node": {}})
    assert harness.spin_until(lambda: "odom" in harness.last and harness.pubs["cmd_vel_teleop"].get_subscription_count() > 0, 20)
    harness.drive("cmd_vel_teleop", 0.5, 0.0, 14.0)                # straight at the wall, asking for full speed
    x, _, _ = harness.pose()
    assert not harness.collided()
    assert 3.4 < x < 4.0 - 0.18                                    # close to the wall, not touching (the disc has a 0.18 m radius)
    assert abs(harness.last["odom"].twist.twist.linear.x) < 0.01
    assert max(m.twist.twist.linear.x for m in harness.log["odom"]) <= 0.61      # never faster than v_max
    assert harness.last["free_distance"].data < 0.4
    harness.drive("cmd_vel_teleop", -0.2, 0.0, 3.0)
    assert harness.pose()[0] < x - 0.3                             # reversing is not limited


def test_mux_priority_and_watchdog(stack, harness):
    bring_up(stack, harness, {"mux_node": dict(teleop_timeout=0.5, nav_timeout=0.5)})
    assert harness.spin_until(lambda: harness.pubs["cmd_vel_nav"].get_subscription_count() > 0 and "active_source" in harness.last, 20)
    harness.drive("cmd_vel_nav", 0.3, 0.0, 1.0)
    assert harness.last["active_source"].data == "nav" and harness.last["cmd_vel_muxed"].linear.x == pytest.approx(0.3)

    def both(seconds):
        msg_t, msg_n = Twist(), Twist()
        msg_t.linear.x, msg_n.linear.x = -0.1, 0.3
        end = time.time() + seconds
        while time.time() < end:
            harness.pubs["cmd_vel_teleop"].publish(msg_t)
            harness.pubs["cmd_vel_nav"].publish(msg_n)
            harness.spin_for(0.05)
    both(1.0)
    assert harness.last["active_source"].data == "teleop" and harness.last["cmd_vel_muxed"].linear.x == pytest.approx(-0.1)
    harness.drive("cmd_vel_nav", 0.3, 0.0, 1.2)                    # teleop went silent: after its timeout navigation takes over again
    assert harness.last["active_source"].data == "nav" and harness.last["cmd_vel_muxed"].linear.x == pytest.approx(0.3)
    harness.spin_for(1.0)                                          # everything silent: the watchdog stops the robot
    assert harness.last["active_source"].data == "none" and harness.last["cmd_vel_muxed"].linear.x == 0.0


def test_safety_does_not_allow_blind_forward_motion(stack, harness):
    bring_up(stack, harness, {"safety_node": {}})                  # no simulator: no scan will ever arrive
    assert harness.spin_until(lambda: harness.pubs["cmd_vel_muxed"].get_subscription_count() > 0, 20)
    harness.drive("cmd_vel_muxed", 0.3, 0.0, 1.5)
    assert harness.last["cmd_vel"].linear.x == 0.0                 # forward without a scan: refused
    harness.drive("cmd_vel_muxed", -0.2, 0.0, 1.5)
    assert harness.last["cmd_vel"].linear.x == pytest.approx(-0.2) # reversing needs no scan
    harness.drive("cmd_vel_muxed", 0.0, 0.0, 0.0)
    harness.spin_for(1.5)                                          # a stale command becomes a stop
    assert harness.last["cmd_vel"].linear.x == pytest.approx(0.0, abs=1e-9)


def test_pure_pursuit_follows_a_published_circle(stack, harness):
    R, v = 1.2, 0.4
    bring_up(stack, harness, {"sim_node": dict(circles=NO_CIRCLES, walls=NO_WALLS, start_pose=[R, 0.0, math.pi / 2]), "mux_node": {}, "safety_node": {},
                              "pure_pursuit_node": dict(lookahead=0.5, v_nominal=v)})
    assert harness.spin_until(lambda: "odom" in harness.last and harness.pubs["path"].get_subscription_count() > 0, 20)
    path = PathMsg()
    path.header.frame_id = "odom"
    n = 160
    for k in range(n + 1):
        a = 2 * math.pi * k / n
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y = R * math.cos(a), R * math.sin(a)
        path.poses.append(pose)
    harness.pubs["path"].publish(path)
    assert harness.spin_until(lambda: harness.last.get("goal_reached") is not None and harness.last["goal_reached"].data, 90), f"pose {harness.pose()}"
    radii = [math.hypot(m.pose.pose.position.x, m.pose.pose.position.y) for m in harness.log["odom"]]
    t0 = len(radii) // 5                                           # skip the start-up transient
    assert max(abs(r - R) for r in radii[t0:]) < 0.03
    assert not harness.collided()


def test_unstable_gains_are_refused_at_startup(stack):
    p = stack.start("go_to_goal_node", dict(k_beta=0.5))
    assert p.wait(timeout=30) != 0
    assert "unstable gains" in stack.log("go_to_goal_node")


def test_launch_file_brings_up_the_whole_pipeline(tmp_path, harness):
    launch_file = Path(__file__).resolve().parents[1] / "launch" / "sim_goto.launch.py"
    log = open(tmp_path / "launch.log", "w")
    proc = subprocess.Popen(["ros2", "launch", str(launch_file)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        names = list(NODE_NAMES.values())[:3] + ["robot_go_to_goal"]
        assert harness.wait_for_nodes(names, timeout=40), (tmp_path / "launch.log").read_text()
        assert harness.spin_until(lambda: "odom" in harness.last and harness.pubs["goal_pose"].get_subscription_count() > 0, 20)
        goal = PoseStamped()
        goal.pose.position.x, goal.pose.position.y, goal.pose.orientation.w = 2.0, -2.0, 1.0
        harness.pubs["goal_pose"].publish(goal)
        assert harness.spin_until(lambda: harness.last.get("goal_reached") is not None and harness.last["goal_reached"].data, 60)
        assert math.hypot(harness.pose()[0] - 2.0, harness.pose()[1] + 2.0) < 0.05
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def test_every_key_in_the_parameter_file_is_a_declared_parameter():
    # ROS silently ignores parameters a node never declares, so a typo in config/robot.yaml would do nothing: check each key against the node's source
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "robot.yaml").read_text())
    by_node = {name: module for module, name in NODE_NAMES.items()}
    assert set(config) == set(by_node)
    for node_name, section in config.items():
        source = (Path(SRC) / "robot_kontrol" / "ros" / f"{by_node[node_name]}.py").read_text()
        for key in section["ros__parameters"]:
            assert f'"{key}"' in source, f"{node_name} never declares {key!r}"
