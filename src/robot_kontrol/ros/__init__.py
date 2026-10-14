"""ROS 2 nodes (rclpy) around the pure-Python core. Importing this package needs a sourced ROS 2 environment; the core (``robot_kontrol``) does not.

Pipeline and topics (all relative, remap with ``--ros-args -r``)::

    go_to_goal_node / pure_pursuit_node --cmd_vel_nav--> mux_node --cmd_vel_muxed--> safety_node --cmd_vel--> sim_node
    teleop (yours) ---------cmd_vel_teleop------------->                              scan, odom <-- sim_node

Run a node with ``python3 -m robot_kontrol.ros.<name>`` (``sim_node``, ``mux_node``, ``safety_node``, ``go_to_goal_node``, ``pure_pursuit_node``).
"""
