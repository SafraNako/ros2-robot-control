#!/usr/bin/env bash
# Run the whole test suite, including the ROS 2 integration tests (real nodes talking over real DDS), in the official ROS 2 Jazzy container.
# Works on any machine with Docker (on macOS: Colima or Docker Desktop). Extra arguments are passed to pytest, e.g.  tools/ros_container_test.sh -k "mux or safety"
set -euo pipefail
cd "$(dirname "$0")/.."
args=$(printf ' %q' "$@")
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD":/work -w /work ros:jazzy-ros-core \
    bash -lc "source /opt/ros/jazzy/setup.bash && python3 -m pytest tests -v -p no:cacheprovider${args}"
