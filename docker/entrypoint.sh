#!/bin/bash
set -e

# Source ROS 2 Humble base installation
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source "/opt/ros/humble/setup.bash"
fi

# Source workspace if built
if [ -f "/home/ros/ros2_ws/install/setup.bash" ]; then
    source "/home/ros/ros2_ws/install/setup.bash"
fi

exec "$@"

