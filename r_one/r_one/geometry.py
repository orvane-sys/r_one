"""Camera intrinsics, range lookup and TF helpers used to place detections in the world."""

import math
from typing import Optional

from geometry_msgs.msg import Point, PointStamped
import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
import tf2_geometry_msgs  # noqa: F401 - registers geometry_msgs types with TF2
import tf2_ros


def intrinsics_are_valid(k) -> bool:
    """Check a CameraInfo K matrix. Uncalibrated cameras often publish all zeros."""
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    if not all(math.isfinite(value) for value in (fx, cx, fy, cy)):
        return False
    return fx > 0.0 and fy > 0.0


def extract_lidar_distance(angle_rad: float, scan: LaserScan) -> Optional[float]:
    """Return the median valid range of the five beams around a bearing."""
    target_yaw = math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    if scan.angle_min >= 0.0 and target_yaw < 0.0:
        target_yaw += 2.0 * math.pi

    if target_yaw < scan.angle_min or target_yaw > scan.angle_max:
        return None

    idx = int(round((target_yaw - scan.angle_min) / scan.angle_increment))

    window_ranges = []
    for offset in range(-2, 3):
        cur_idx = idx + offset
        if 0 <= cur_idx < len(scan.ranges):
            r = scan.ranges[cur_idx]
            if (
                scan.range_min < r < scan.range_max
                and not math.isinf(r)
                and not math.isnan(r)
            ):
                window_ranges.append(r)

    if not window_ranges:
        return None

    window_ranges.sort()
    return window_ranges[len(window_ranges) // 2]


def extract_depth_image_distance(
    u: float,
    v: float,
    depth_image: np.ndarray,
    half_width: float = 2.0,
    half_height: float = 2.0,
) -> Optional[float]:
    """Return the nearest-quartile depth of a patch around a pixel."""
    # Use the nearest quartile of the patch: the centre of a box can fall between a
    # person's legs, so a reading there may hit the background.
    h, w = depth_image.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None

    half_u = max(2, int(half_width))
    half_v = max(2, int(half_height))
    u_min, u_max = max(0, ui - half_u), min(w, ui + half_u + 1)
    v_min, v_max = max(0, vi - half_v), min(h, vi + half_v + 1)
    window = depth_image[v_min:v_max, u_min:u_max]

    valid_vals = window[
        (window > 0.1) & (window < 10.0) & ~np.isnan(window) & ~np.isinf(window)
    ]
    if len(valid_vals) == 0:
        return None
    return float(np.percentile(valid_vals, 25))


def transform_point(
    tf_buffer,
    target_frame: str,
    local_x: float,
    local_y: float,
    local_z: float,
    source_frame: str,
    stamp,
    logger,
) -> Optional[Point]:
    """Move a point into target_frame, falling back to the latest transform."""
    pt = PointStamped()
    pt.header.frame_id = source_frame
    pt.header.stamp = stamp
    pt.point.x = local_x
    pt.point.y = local_y
    pt.point.z = local_z

    try:
        transformed = tf_buffer.transform(
            pt, target_frame, timeout=Duration(seconds=0.05)
        )
        return transformed.point
    except (
        tf2_ros.LookupException,
        tf2_ros.ExtrapolationException,
        tf2_ros.ConnectivityException,
    ):
        try:
            pt.header.stamp = Time().to_msg()
            transformed = tf_buffer.transform(
                pt, target_frame, timeout=Duration(seconds=0.05)
            )
            return transformed.point
        except Exception as e:
            logger.debug(f"TF transformation error: {e}")
            return None
