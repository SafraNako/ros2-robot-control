"""Go-to-goal node: ``odom`` + ``goal_pose`` (a ``PoseStamped``, taken in the odometry frame) -> ``cmd_vel_nav``; ``goal_reached`` says whether the
last goal has been reached. Without a goal nothing is published, so the mux watchdog stops the robot. Unstable gains are refused at start-up."""
from __future__ import annotations

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool

from ..go_to_goal import GoToGoal
from ._util import make_twist, pose_of, spin


class GoToGoalNode(Node):
    def __init__(self):
        super().__init__("robot_go_to_goal")
        get = lambda name, default: float(self.declare_parameter(name, default).value)  # noqa: E731
        rate = get("rate", 20.0)
        self.ctl = GoToGoal(k_rho=get("k_rho", 0.6), k_alpha=get("k_alpha", 1.8), k_beta=get("k_beta", -0.5), k_final=get("k_final", 1.5),
                            v_max=get("v_max", 0.5), w_max=get("w_max", 1.5), pos_tol=get("pos_tol", 0.03), yaw_tol=get("yaw_tol", 0.05),
                            allow_reverse=bool(self.declare_parameter("allow_reverse", True).value))
        self._pose, self._goal, self._done = None, None, False
        self.create_subscription(Odometry, "odom", lambda m: setattr(self, "_pose", pose_of(m.pose.pose)), 10)
        self.create_subscription(PoseStamped, "goal_pose", self._on_goal, 10)
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel_nav", 10)
        self.done_pub = self.create_publisher(Bool, "goal_reached", 10)
        self.create_timer(1.0 / rate, self._tick)

    def _on_goal(self, msg):
        self._goal, self._done = pose_of(msg.pose), False
        self.get_logger().info(f"new goal {self._goal}")

    def _tick(self):
        if self._pose is None or self._goal is None:
            return
        v, w, self._done = self.ctl.command(self._pose, self._goal)
        self.cmd_pub.publish(make_twist(v, w))
        self.done_pub.publish(Bool(data=bool(self._done)))


def main(args=None):
    spin(GoToGoalNode, args)


if __name__ == "__main__":
    main()
