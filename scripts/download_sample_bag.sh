#!/usr/bin/env bash
# ==============================================================================
# Download sample demonstration rosbag from GitHub Releases
# ==============================================================================

set -e

DEST_DIR="sample_bags"
MCAP_FILE="${DEST_DIR}/sample_spatial_run.mcap"
RELEASE_URL="https://github.com/orvane-sys/r_one/releases/download/v0.1.0/sample_spatial_run.mcap"

mkdir -p "${DEST_DIR}"

if [ -f "${MCAP_FILE}" ]; then
    echo "Sample bag already exists at: ${MCAP_FILE}"
    exit 0
fi

echo "Downloading sample demonstration rosbag..."
if command -v curl >/dev/null 2>&1; then
    curl -L "${RELEASE_URL}" -o "${MCAP_FILE}" || echo "Download note: If release asset is not yet published, record locally using ./scripts/record_sample_bag.sh"
elif command -v wget >/dev/null 2>&1; then
    wget -O "${MCAP_FILE}" "${RELEASE_URL}" || echo "Download note: If release asset is not yet published, record locally using ./scripts/record_sample_bag.sh"
else
    echo "Neither curl nor wget found. Please install curl or wget."
    exit 1
fi

echo "Ready. Run: ros2 launch r_one spatial_bag_playback.launch.py bag_path:=${MCAP_FILE}"

