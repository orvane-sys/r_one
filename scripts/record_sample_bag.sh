#!/usr/bin/env bash
# ==============================================================================
# Record a 15-30 second sample demonstration rosbag for zero-simulation testing
# ==============================================================================

set -e

BAG_NAME=${1:-"sample_spatial_run"}
OUTPUT_DIR="sample_bags"

mkdir -p "${OUTPUT_DIR}"
FULL_PATH="${OUTPUT_DIR}/${BAG_NAME}"

echo "=========================================================="
echo " Recording sample perception rosbag to: ${FULL_PATH}"
echo " Topics: Camera, CameraInfo, LiDAR, Depth Image, TF"
echo " Press Ctrl+C when finished recording (15-20s recommended)."
echo "=========================================================="

# Gazebo stamps messages with simulation time, so the bag has to record on that clock too.
# Without this, playback publishes a /clock that does not match the message stamps.
SIM_TIME_FLAG=()
if ros2 topic list | grep -qx '/clock'; then
  SIM_TIME_FLAG=(--use-sim-time)
fi

# This writes a directory holding metadata.yaml and sample_spatial_run_0.mcap
ros2 bag record -s mcap "${SIM_TIME_FLAG[@]}" -o "${FULL_PATH}" \
  /camera/image_raw \
  /camera/camera_info \
  /scan \
  /camera/depth/image_raw \
  /tf \
  /tf_static

