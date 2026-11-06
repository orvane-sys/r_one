FROM osrf/ros:humble-desktop-full

# PyTorch wheel index. CPU is the default so the image works on any machine.
# For an NVIDIA GPU build pass: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TURTLEBOT3_MODEL=waffle
# spatial_vision looks for model weights here before trying to download them
ENV R_ONE_MODEL_DIR=/opt/r_one/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    python3-pip \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/ros/ros2_ws

# ROS dependencies come from package.xml, so a missing declaration fails the build
# here instead of failing at runtime. Only package.xml is copied at this point so this
# layer stays cached while the source changes.
COPY package.xml src/r_one/package.xml
RUN apt-get update \
    && rosdep update --rosdistro humble \
    && rosdep install --from-paths src --ignore-src -y --rosdistro humble \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (numpy must stay < 2 for cv_bridge, see requirements.txt)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir "torch>=2.2,<2.9" torchvision --index-url ${TORCH_INDEX_URL} \
    && pip3 install --no-cache-dir -r /tmp/requirements.txt

# Cache the YOLO weights inside the image so startup needs no network
RUN mkdir -p ${R_ONE_MODEL_DIR} \
    && cd ${R_ONE_MODEL_DIR} \
    && python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Fail the build if the perception stack cannot import together
RUN /bin/bash -c "source /opt/ros/humble/setup.bash \
    && python3 -c \"import numpy, cv2; from cv_bridge import CvBridge; from ultralytics import YOLO; \
assert numpy.__version__.startswith('1.'), numpy.__version__; print('perception imports OK, numpy', numpy.__version__)\""

# Copy package source and build
COPY . src/r_one/
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install --packages-select r_one"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "r_one", "spatial_scene.launch.py"]
