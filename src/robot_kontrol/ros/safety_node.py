"""Safety node: ``cmd_vel_muxed`` + ``scan`` -> ``cmd_vel``. Forward motion is slowed to what the scan allows (see ``robot_kontrol.safety``), the
result is clamped and slewed by the velocity limiter at a fixed rate, and ``free_distance`` reports the room ahead on the commanded arc.

Fail-safe behaviour: a command older than ``cmd_timeout`` becomes a stop, and forward motion needs a scan younger than ``scan_timeout`` (reversing
and turning on the spot do not, as they are never limited). A purely reactive layer: it prevents contact, it does not plan around anything."""
from __future__ import annotations

from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

from ..limits import VelocityLimiter
from ..safety import ScanSafety, scan_points
from ._util import make_twist, now_s, spin


class SafetyNode(Node):
    def __init__(self):
        super().__init__("robot_safety")
        get = lambda name, default: float(self.declare_parameter(name, default).value)  # noqa: E731
        self.rate = get("rate", 20.0)
        self.cmd_timeout, self.scan_timeout = get("cmd_timeout", 0.5), get("scan_timeout", 0.5)
        self.safety = ScanSafety(radius=get("radius", 0.22), decel=get("decel", 0.8), margin=get("margin", 0.10), horizon=get("horizon", 3.0))
        self.limiter = VelocityLimiter(v_max=get("v_max", 0.6), w_max=get("w_max", 1.5), a_max=get("a_max", 0.8), alpha_max=get("alpha_max", 3.0), v_min=get("v_min", -0.3))
        self._cmd, self._cmd_t = (0.0, 0.0), -1e18
        self._points, self._scan_t = None, -1e18
        self._last = None
        self.create_subscription(Twist, "cmd_vel_muxed", self._on_cmd, 10)
        self.create_subscription(LaserScan, "scan", self._on_scan, 10)
        self.out_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.free_pub = self.create_publisher(Float32, "free_distance", 10)
        self.create_timer(1.0 / self.rate, self._tick)

    def _on_cmd(self, msg):
        self._cmd, self._cmd_t = (msg.linear.x, msg.angular.z), now_s(self)

    def _on_scan(self, msg):
        self._points = scan_points(msg.angle_min, msg.angle_increment, msg.ranges, msg.range_min, msg.range_max)
        self._scan_t = now_s(self)

    def _tick(self):
        now = now_s(self)
        dt = 1.0 / self.rate if self._last is None else min(max(now - self._last, 1e-3), 0.2)
        self._last = now
        v, w = self._cmd if now - self._cmd_t <= self.cmd_timeout else (0.0, 0.0)
        free = self.safety.horizon
        if v > 0:
            if self._points is None or now - self._scan_t > self.scan_timeout:
                v, w = 0.0, 0.0                         # moving forward blind is not allowed
            else:
                v, w, free, _ = self.safety.limit(v, w, *self._points)
        v, w = self.limiter.apply(v, w, dt)
        self.out_pub.publish(make_twist(v, w))
        self.free_pub.publish(Float32(data=float(free)))


def main(args=None):
    spin(SafetyNode, args)


if __name__ == "__main__":
    main()
