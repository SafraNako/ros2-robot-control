"""Pure-pursuit node: ``odom`` + ``path`` (a ``nav_msgs/Path`` in the odometry frame) -> ``cmd_vel_nav``; ``goal_reached`` turns true at the end of the
path. A new path restarts the progress along it. Without a path nothing is published."""
from __future__ import annotations

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Bool

from ..pure_pursuit import PurePursuit
from ._util import make_twist, path_points, pose_of, spin


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__("robot_pure_pursuit")
        get = lambda name, default: float(self.declare_parameter(name, default).value)  # noqa: E731
        rate = get("rate", 20.0)
        self.ctl = PurePursuit(lookahead=get("lookahead", 0.5), v_nominal=get("v_nominal", 0.4), v_min=get("v_min", 0.05), w_max=get("w_max", 1.5),
                               a_lat_max=get("a_lat_max", 0.5), decel=get("decel", 0.4), goal_tol=get("goal_tol", 0.05))
        self._pose, self._path, self._done = None, None, False
        self.create_subscription(Odometry, "odom", lambda m: setattr(self, "_pose", pose_of(m.pose.pose)), 10)
        self.create_subscription(Path, "path", self._on_path, 10)
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel_nav", 10)
        self.done_pub = self.create_publisher(Bool, "goal_reached", 10)
        self.create_timer(1.0 / rate, self._tick)

    def _on_path(self, msg):
        pts = path_points(msg)
        if len(pts) < 2:
            self.get_logger().warning("ignoring a path with fewer than 2 poses")
            return
        self._path, self._done = pts, False
        self.ctl.reset()
        self.get_logger().info(f"new path with {len(pts)} points")

    def _tick(self):
        if self._pose is None or self._path is None:
            return
        v, w, self._done = self.ctl.command(self._pose, self._path)
        self.cmd_pub.publish(make_twist(v, w))
        self.done_pub.publish(Bool(data=bool(self._done)))


def main(args=None):
    spin(PurePursuitNode, args)


if __name__ == "__main__":
    main()
