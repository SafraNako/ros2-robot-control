"""Command multiplexer node: ``cmd_vel_teleop`` (priority 10) beats ``cmd_vel_nav`` (priority 1); each source expires after its timeout, and when
all have expired the node publishes a stop. It publishes ``cmd_vel_muxed`` at a fixed rate (so downstream always sees a fresh command) and the
name of the winning source on ``active_source`` (``none`` after a watchdog stop)."""
from __future__ import annotations

from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from ..mux import CmdMux
from ._util import make_twist, now_s, spin


class MuxNode(Node):
    def __init__(self):
        super().__init__("robot_mux")
        rate = float(self.declare_parameter("rate", 20.0).value)
        self.mux = CmdMux()
        self.mux.add_source("teleop", 10, float(self.declare_parameter("teleop_timeout", 0.5).value))
        self.mux.add_source("nav", 1, float(self.declare_parameter("nav_timeout", 0.5).value))
        self.create_subscription(Twist, "cmd_vel_teleop", lambda m: self._on_cmd("teleop", m), 10)
        self.create_subscription(Twist, "cmd_vel_nav", lambda m: self._on_cmd("nav", m), 10)
        self.out_pub = self.create_publisher(Twist, "cmd_vel_muxed", 10)
        self.source_pub = self.create_publisher(String, "active_source", 10)
        self.create_timer(1.0 / rate, self._tick)

    def _on_cmd(self, name, msg):
        self.mux.update(name, msg.linear.x, msg.angular.z, now_s(self))

    def _tick(self):
        v, w, source = self.mux.select(now_s(self))
        self.out_pub.publish(make_twist(v, w))
        self.source_pub.publish(String(data=source or "none"))


def main(args=None):
    spin(MuxNode, args)


if __name__ == "__main__":
    main()
