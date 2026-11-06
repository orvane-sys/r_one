#!/usr/bin/env bash
# ==============================================================================
# Download the sample demonstration rosbag from GitHub Releases.
#
# Optional environment variables:
#   SAMPLE_BAG_SHA256=<hex digest>  verify the download against this digest. If unset,
#                                   the script looks for "<asset>.sha256" next to the asset.
#   SAMPLE_BAG_URL=<url>            download from somewhere other than the GitHub release.
# ==============================================================================

set -euo pipefail

DEST_DIR="sample_bags"
BAG_NAME="sample_spatial_run"
BAG_DIR="${DEST_DIR}/${BAG_NAME}"
RELEASE_URL="${SAMPLE_BAG_URL:-https://github.com/orvane-sys/r_one/releases/download/v0.1.0/${BAG_NAME}.tar.gz}"

# A bag is a directory. An MCAP file starts with the magic bytes 0x89 'M' 'C' 'A' 'P'.
is_bag() {
    [ -s "$1/metadata.yaml" ] && [ -s "$1/${BAG_NAME}_0.mcap" ] \
        && [ "$(head -c 5 "$1/${BAG_NAME}_0.mcap" | tail -c 4)" = "MCAP" ]
}

fail_with_hint() {
    echo "ERROR: $1" >&2
    echo "You can record your own bag instead: ./scripts/record_sample_bag.sh" >&2
    exit 1
}

mkdir -p "${DEST_DIR}"

if [ -e "${BAG_DIR}" ]; then
    if is_bag "${BAG_DIR}"; then
        echo "Sample bag already exists at: ${BAG_DIR}"
        exit 0
    fi
    echo "Existing ${BAG_DIR} is not a valid bag, downloading again."
    rm -rf "${BAG_DIR}"
fi

if ! command -v curl >/dev/null 2>&1; then
    fail_with_hint "curl is required. Install it with: sudo apt install curl"
fi

TMP_FILE="$(mktemp "${DEST_DIR}/.download.XXXXXX")"
TMP_DIR="$(mktemp -d "${DEST_DIR}/.extract.XXXXXX")"
trap 'rm -rf "${TMP_FILE}" "${TMP_DIR}"' EXIT

echo "Downloading sample demonstration rosbag..."
# -f makes HTTP errors (such as 404 for an unpublished asset) fail instead of saving the error page
if ! curl -fL --retry 3 "${RELEASE_URL}" -o "${TMP_FILE}"; then
    fail_with_hint "could not download ${RELEASE_URL} (the release asset may not be published yet)."
fi

EXPECTED_SHA="${SAMPLE_BAG_SHA256:-}"
if [ -z "${EXPECTED_SHA}" ]; then
    EXPECTED_SHA="$(curl -fsL "${RELEASE_URL}.sha256" 2>/dev/null | awk '{print $1}' || true)"
fi
if [ -n "${EXPECTED_SHA}" ]; then
    ACTUAL_SHA="$(sha256sum "${TMP_FILE}" | awk '{print $1}')"
    if [ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]; then
        fail_with_hint "checksum mismatch (expected ${EXPECTED_SHA}, got ${ACTUAL_SHA})."
    fi
    echo "Checksum verified."
else
    echo "WARNING: no checksum available, skipping verification." >&2
fi

if ! tar -xzf "${TMP_FILE}" -C "${TMP_DIR}" --no-same-owner; then
    fail_with_hint "could not extract the downloaded archive."
fi

if ! is_bag "${TMP_DIR}/${BAG_NAME}"; then
    fail_with_hint "downloaded archive does not contain a valid ${BAG_NAME} bag."
fi

mv "${TMP_DIR}/${BAG_NAME}" "${BAG_DIR}"

echo "Ready. Run: ros2 launch r_one spatial_bag_playback.launch.py bag_path:=${BAG_DIR}"
