"""The simulator as a ROS 2 node: subscribes ``cmd_vel``, publishes ``odom`` (wheel odometry), ``scan`` (lidar) and ``collision``.

The odometry is dead reckoning from the simulated wheel travel with no noise, so it equals the true pose; the node does no collision physics
(it only reports contact). No ``tf`` is published.
"""
from __future__ import annotations

import math

import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from ..geometry import yaw_to_quaternion
from ..kinematics import DiffDrive
from ..kinematics import Odometry as WheelOdometry
from ..sim import DEMO_CIRCLES, DEMO_WALLS, DiffDriveSim, world_from_flat
from ._util import now_s, spin

class SimNode(Node):
    def __init__(self):
        super().__init__("robot_sim")
        get = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.rate = float(get("rate", 50.0))
        self.scan_rate = float(get("scan_rate", 10.0))
        self.cmd_timeout = float(get("cmd_timeout", 0.5))
        start = [float(a) for a in get("start_pose", [0.0, 0.0, 0.0])]
        self.world = world_from_flat(get("circles", DEMO_CIRCLES), get("walls", DEMO_WALLS))
        self.sim = DiffDriveSim(*start, tau_v=float(get("tau_v", 0.15)), tau_w=float(get("tau_w", 0.10)), radius=float(get("robot_radius", 0.18)))
        self.beams = int(get("scan_beams", 360))
        self.max_range = float(get("scan_max_range", 6.0))
        self.drive = DiffDrive(float(get("wheel_radius", 0.05)), float(get("track", 0.30)))
        self.odometry = WheelOdometry(self.drive.track, start)
        self.frame_id, self.child_frame_id = str(get("odom_frame", "odom")), str(get("base_frame", "base_link"))
        self._cmd = (0.0, 0.0)
        self._cmd_t = -math.inf
        self._last = None
        self.create_subscription(Twist, "cmd_vel", self._on_cmd, 10)
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.scan_pub = self.create_publisher(LaserScan, "scan", 10)
        self.collision_pub = self.create_publisher(Bool, "collision", 10)
        self.create_timer(1.0 / self.rate, self._step)
        self.create_timer(1.0 / self.scan_rate, self._publish_scan)
        self.get_logger().info(f"sim ready: {len(self.world.circles)} circles, {len(self.world.segments)} walls, start {start}")

    def _on_cmd(self, msg):
        self._cmd = (msg.linear.x, msg.angular.z)
        self._cmd_t = now_s(self)

    def _step(self):
        now = now_s(self)
        dt = 1.0 / self.rate if self._last is None else min(max(now - self._last, 1e-4), 0.1)
        self._last = now
        v_cmd, w_cmd = self._cmd if now - self._cmd_t <= self.cmd_timeout else (0.0, 0.0)
        self.sim.step(v_cmd, w_cmd, dt)
        dl, dr = self.drive.wheel_travel(*self.sim.mean_twist, dt)
        self.odometry.update(dl, dr)
        self._publish_odom(now)
        self.collision_pub.publish(Bool(data=bool(self.sim.collided(self.world))))

    def _publish_odom(self, now):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id, msg.child_frame_id = self.frame_id, self.child_frame_id
        x, y, th = self.odometry.pose
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        q = yaw_to_quaternion(th)
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = q
        msg.twist.twist.linear.x, msg.twist.twist.angular.z = self.sim.v, self.sim.w
        self.odom_pub.publish(msg)

    def _publish_scan(self):
        a0, inc, ranges = self.sim.scan(self.world, self.beams, 2 * math.pi, self.max_range)
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.child_frame_id
        msg.angle_min, msg.angle_increment, msg.angle_max = float(a0), float(inc), float(a0 + inc * (self.beams - 1))
        msg.range_min, msg.range_max = 0.05, self.max_range
        msg.ranges = [float(r) for r in np.asarray(ranges)]
        self.scan_pub.publish(msg)


def main(args=None):
    spin(SimNode, args)


if __name__ == "__main__":
    main()
