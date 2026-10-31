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

ros2 bag record -s mcap -o "${FULL_PATH}" \
  /camera/image_raw \
  /camera/camera_info \
  /scan \
  /camera/depth/image_raw \
  /tf \
  /tf_static

