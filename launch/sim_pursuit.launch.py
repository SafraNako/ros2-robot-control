"""The simulated robot with pure-pursuit navigation: sim -> (mux) -> safety, parameters from config/robot.yaml.

    ros2 launch launch/sim_pursuit.launch.py

The nodes are started as plain Python processes (no colcon build needed); the package source is put on PYTHONPATH.
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import ExecuteProcess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def node(module):
    pythonpath = os.path.join(ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    return ExecuteProcess(
        cmd=[sys.executable, "-m", "robot_kontrol.ros." + module, "--ros-args", "--params-file", os.path.join(ROOT, "config", "robot.yaml")],
        additional_env={"PYTHONPATH": pythonpath},
        output="screen",
    )


def generate_launch_description():
    return LaunchDescription([node(m) for m in ("sim_node", "mux_node", "safety_node", "pure_pursuit_node")])
