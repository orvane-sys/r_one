"""Camera intrinsics, range lookup and TF helpers used to place detections in the world."""

import math
from typing import Optional, Tuple

from geometry_msgs.msg import Point, PointStamped, Transform
import numpy as np
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan
import tf2_geometry_msgs  # noqa: F401 - registers geometry_msgs types with TF2
import tf2_ros

# Lidar beams inside a box are split into groups wherever their range jumps by more than
# this. A person's two legs at the same range stay one group; the wall behind them does not.
CLUSTER_GAP_M = 0.3
# A group needs this many beams. At 1 m one degree is 1.7 cm, so a leg spans several beams
# and a single stray beam is ignored.
CLUSTER_MIN_POINTS = 2
# Depth readings outside this range are treated as invalid
DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 10.0


def intrinsics_are_valid(k) -> bool:
    """Check a CameraInfo K matrix. Uncalibrated cameras often publish all zeros."""
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    if not all(math.isfinite(value) for value in (fx, cx, fy, cy)):
        return False
    return fx > 0.0 and fy > 0.0


def scan_points(scan: LaserScan) -> np.ndarray:
    """Return the scan's valid beams as an N×3 array of points in the scan's frame."""
    ranges = np.asarray(scan.ranges, dtype=np.float64)
    angles = scan.angle_min + scan.angle_increment * np.arange(len(ranges))
    valid = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
    ranges, angles = ranges[valid], angles[valid]
    return np.column_stack(
        (ranges * np.cos(angles), ranges * np.sin(angles), np.zeros(len(ranges)))
    )


def transform_to_matrix(transform: Transform) -> np.ndarray:
    """Turn a geometry_msgs Transform into a 4×4 matrix."""
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
    t = transform.translation
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4×4 transform to an N×3 array of points."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def points_in_box(
    points_cam: np.ndarray,
    box: Tuple[float, float, float, float],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Keep the camera-frame points in front of the camera whose pixel falls inside box."""
    points = points_cam[points_cam[:, 2] > 0.0]
    u = fx * points[:, 0] / points[:, 2] + cx
    v = fy * points[:, 1] / points[:, 2] + cy
    x1, y1, x2, y2 = box
    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return points[inside]


def measured_size(
    box: Tuple[float, ...],
    range_m: float,
    fx: float,
    fy: float,
) -> Tuple[float, ...]:
    """
    Return an object's (width, depth, height) in metres from its pixel box and range.

    The camera cannot see an object's depth, so depth is assumed equal to its width.
    Returns () when the box or the range is not positive and finite.
    """
    if len(box) != 4:
        return ()
    x1, y1, x2, y2 = box
    values = (x2 - x1, y2 - y1, range_m, fx, fy)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        return ()
    width = float((x2 - x1) * range_m / fx)
    height = float((y2 - y1) * range_m / fy)
    return (width, width, height)


def nearest_cluster(
    points: np.ndarray,
    gap_m: float = CLUSTER_GAP_M,
    min_points: int = CLUSTER_MIN_POINTS,
) -> Optional[np.ndarray]:
    """
    Return the centroid of the nearest group of points, or None.

    The points are sorted by their distance from the origin and split wherever that
    distance jumps by more than gap_m. Groups with fewer than min_points points are skipped.
    """
    if len(points) == 0:
        return None
    ranges = np.linalg.norm(points, axis=1)
    order = np.argsort(ranges)
    points, ranges = points[order], ranges[order]
    breaks = np.flatnonzero(np.diff(ranges) > gap_m) + 1
    for group in np.split(points, breaks):
        if len(group) >= min_points:
            return group.mean(axis=0)
    return None


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
        (window > DEPTH_MIN_M) & (window < DEPTH_MAX_M) & ~np.isnan(window) & ~np.isinf(window)
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
    counters: Optional[dict] = None,
) -> Optional[Point]:
    """
    Move a point into target_frame using the transform at the measurement's stamp.

    Returns None when that transform is not available within 0.1 s. A point placed with a
    transform from another time would land in the wrong spot while the robot moves. When
    counters is given, its 'tf_failures' entry counts the failed lookups.
    """
    pt = PointStamped()
    pt.header.frame_id = source_frame
    pt.header.stamp = stamp
    pt.point.x = local_x
    pt.point.y = local_y
    pt.point.z = local_z

    try:
        transformed = tf_buffer.transform(
            pt, target_frame, timeout=Duration(seconds=0.1)
        )
        return transformed.point
    except tf2_ros.TransformException as e:
        if counters is not None:
            counters['tf_failures'] += 1
        logger.debug(f"TF transformation error: {e}")
        return None
