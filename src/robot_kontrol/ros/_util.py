"""Small helpers shared by the nodes: message conversions and the spin boilerplate."""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Twist

from ..geometry import quaternion_to_yaw


def now_s(node):
    return node.get_clock().now().nanoseconds * 1e-9


def make_twist(v, w):
    msg = Twist()
    msg.linear.x, msg.angular.z = float(v), float(w)
    return msg


def pose_of(pose_msg):
    """``(x, y, theta)`` from a ``geometry_msgs/Pose``."""
    q = pose_msg.orientation
    return (pose_msg.position.x, pose_msg.position.y, quaternion_to_yaw(q.x, q.y, q.z, q.w))


def path_points(path_msg):
    """An ``(n, 2)`` array from a ``nav_msgs/Path``."""
    return np.array([[p.pose.position.x, p.pose.position.y] for p in path_msg.poses], float).reshape(-1, 2)


def spin(node_factory, args=None):
    """Create the node after ``rclpy.init`` and spin until interrupted; a failure in the constructor is raised, not swallowed."""
    rclpy.init(args=args)
    node = None
    try:
        node = node_factory()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
