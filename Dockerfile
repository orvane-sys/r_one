FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TURTLEBOT3_MODEL=waffle

# Install essential system dependencies and ROS 2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-tf2-geometry-msgs \
    ros-humble-message-filters \
    ros-humble-visualization-msgs \
    ros-humble-vision-msgs \
    ros-humble-ament-cmake-python \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-xacro \
    ros-humble-turtlebot3-gazebo \
    ros-humble-turtlebot3-teleop \
    && rm -rf /var/lib/apt/lists/*

# Install strictly pinned Python packages (numpy < 2 required for cv_bridge/OpenCV compatibility)
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    ultralytics \
    lapx

# Pre-cache YOLOv8 nano model weights into container to prevent download delays on startup
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create ROS 2 workspace
WORKDIR /home/ros/ros2_ws
RUN mkdir -p src/r_one

# Copy package source
COPY . src/r_one/

# Build package inside container
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install --packages-select r_one"

# Configure entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "r_one", "spatial_scene.launch.py"]

