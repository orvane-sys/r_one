"""Build the ROS messages that spatial_vision publishes: costmap cloud, 3D detections, markers."""

import struct
from typing import Dict, Iterable, List, Sequence, Tuple

from geometry_msgs.msg import Point
from r_one import tracking
from r_one_msgs.msg import Track as TrackMsg, TrackArray
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray

# Optional import for vision_msgs to maintain compatibility on minimal systems
try:
    from vision_msgs.msg import (
        Detection3D,
        Detection3DArray,
        ObjectHypothesisWithPose,
    )

    VISION_MSGS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    VISION_MSGS_AVAILABLE = False
    Detection3D = None
    Detection3DArray = None
    ObjectHypothesisWithPose = None

# append_markers draws a track in these namespaces, with the track ID as the marker ID.
TRACK_MARKER_NAMESPACES = (
    'detected_objects', 'object_labels', 'velocity_vectors', 'predicted_trajectories',
)
WORLD_MODEL_NAMESPACE = 'world_model_memory'

# Box used when an object's size is unknown, drawn from 0.25 m below its point upwards
DEFAULT_SIZE = (0.45, 0.45, 0.50)
DEFAULT_CENTRE_OFFSET_M = 0.25


def box_centre_and_size(pt: Point, size: Sequence[float] = ()):
    """
    Return the box centre height and (x, y, z) size for an object at pt.

    A measured size is centred on the point. An unknown size gets the default box.
    """
    if len(size) == 3:
        return pt.z, tuple(float(value) for value in size)
    return pt.z + DEFAULT_CENTRE_OFFSET_M, DEFAULT_SIZE


def build_pointcloud2(
    points: List[Tuple[float, float, float]],
    frame_id: str,
    stamp,
) -> PointCloud2:
    cloud = PointCloud2()
    cloud.header.frame_id = frame_id
    cloud.header.stamp = stamp
    cloud.height = 1
    cloud.width = len(points)
    cloud.is_dense = True
    cloud.is_bigendian = False

    cloud.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud.point_step = 12
    cloud.row_step = 12 * len(points)

    buffer = bytearray()
    for x, y, z in points:
        buffer.extend(struct.pack('fff', float(x), float(y), float(z)))
    cloud.data = bytes(buffer)
    return cloud


def append_detection_3d(
    detection_array,
    frame_id: str,
    track_id: int,
    obj_name: str,
    score: float,
    pt: Point,
    stamp,
    size: Sequence[float] = (),
):
    centre_z, (size_x, size_y, size_z) = box_centre_and_size(pt, size)
    det = Detection3D()
    det.header.frame_id = frame_id
    det.header.stamp = stamp
    det.id = str(track_id) if track_id != -1 else ""

    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = obj_name
    hyp.hypothesis.score = score
    hyp.pose.pose.position.x = pt.x
    hyp.pose.pose.position.y = pt.y
    hyp.pose.pose.position.z = centre_z
    hyp.pose.pose.orientation.w = 1.0
    det.results.append(hyp)

    det.bbox.center.position.x = pt.x
    det.bbox.center.position.y = pt.y
    det.bbox.center.position.z = centre_z
    det.bbox.center.orientation.w = 1.0
    det.bbox.size.x = size_x
    det.bbox.size.y = size_y
    det.bbox.size.z = size_z

    detection_array.detections.append(det)


def append_markers(
    marker_array: MarkerArray,
    frame_id: str,
    base_marker_id: int,
    track_id: int,
    obj_name: str,
    state: str,
    pt: Point,
    velocity: float,
    vel_x: float,
    vel_y: float,
    stamp,
    size: Sequence[float] = (),
):
    base_id = base_marker_id
    centre_z, (size_x, size_y, size_z) = box_centre_and_size(pt, size)

    cube = Marker()
    cube.header.frame_id = frame_id
    cube.header.stamp = stamp
    cube.ns = 'detected_objects'
    cube.id = base_id
    cube.type = Marker.CUBE
    cube.action = Marker.ADD
    cube.pose.position.x = pt.x
    cube.pose.position.y = pt.y
    cube.pose.position.z = centre_z
    cube.pose.orientation.w = 1.0
    cube.scale.x = size_x
    cube.scale.y = size_y
    cube.scale.z = size_z
    cube.lifetime = Duration(seconds=0.6).to_msg()

    if state == 'DYNAMIC':
        cube.color.r = 1.0
        cube.color.g = 0.3
        cube.color.b = 0.1
        cube.color.a = 0.65
    else:
        cube.color.r = 0.1
        cube.color.g = 0.85
        cube.color.b = 0.2
        cube.color.a = 0.65
    marker_array.markers.append(cube)

    text = Marker()
    text.header.frame_id = frame_id
    text.header.stamp = stamp
    text.ns = 'object_labels'
    text.id = base_id
    text.type = Marker.TEXT_VIEW_FACING
    text.action = Marker.ADD
    text.pose.position.x = pt.x
    text.pose.position.y = pt.y
    # Just above the box
    text.pose.position.z = centre_z + size_z / 2.0 + 0.15
    text.scale.z = 0.18
    text.color.r = 1.0
    text.color.g = 1.0
    text.color.b = 1.0
    text.color.a = 1.0
    text.text = f"[{obj_name.upper()}] ID:{track_id}\n{velocity:.2f} m/s ({state})"
    text.lifetime = Duration(seconds=0.6).to_msg()
    marker_array.markers.append(text)

    if state == 'DYNAMIC' and velocity > 0.05:
        arrow = Marker()
        arrow.header.frame_id = frame_id
        arrow.header.stamp = stamp
        arrow.ns = 'velocity_vectors'
        arrow.id = base_id
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.scale.x = 0.04
        arrow.scale.y = 0.08
        arrow.scale.z = 0.08
        arrow.color.r = 1.0
        arrow.color.g = 0.9
        arrow.color.b = 0.0
        arrow.color.a = 0.9
        arrow.lifetime = Duration(seconds=0.6).to_msg()

        p_start = Point(x=pt.x, y=pt.y, z=pt.z + 0.25)
        p_end = Point(
            x=pt.x + vel_x * 0.7, y=pt.y + vel_y * 0.7, z=pt.z + 0.25
        )
        arrow.points = [p_start, p_end]
        marker_array.markers.append(arrow)

        # 4. Lookahead Trajectory Path (Line Strip)
        traj = Marker()
        traj.header.frame_id = frame_id
        traj.header.stamp = stamp
        traj.ns = 'predicted_trajectories'
        traj.id = base_id
        traj.type = Marker.LINE_STRIP
        traj.action = Marker.ADD
        traj.scale.x = 0.04
        traj.color.r = 1.0
        traj.color.g = 0.8
        traj.color.b = 0.0
        traj.color.a = 0.8
        traj.lifetime = Duration(seconds=0.6).to_msg()
        traj.points.append(Point(x=pt.x, y=pt.y, z=pt.z + 0.25))
        for t_ahead in [0.5, 1.0, 1.5, 2.0]:
            traj.points.append(Point(
                x=pt.x + vel_x * t_ahead,
                y=pt.y + vel_y * t_ahead,
                z=pt.z + 0.25
            ))
        marker_array.markers.append(traj)


def append_world_model_markers(
    marker_array: MarkerArray,
    frame_id: str,
    world_model: Dict[str, Tuple[float, float, float]],
    stamp,
):
    for idx, (name_key, (wx, wy, wz)) in enumerate(world_model.items()):
        wm_marker = Marker()
        wm_marker.header.frame_id = frame_id
        wm_marker.header.stamp = stamp
        wm_marker.ns = WORLD_MODEL_NAMESPACE
        wm_marker.id = idx
        wm_marker.type = Marker.CYLINDER
        wm_marker.action = Marker.ADD
        wm_marker.pose.position.x = wx
        wm_marker.pose.position.y = wy
        wm_marker.pose.position.z = wz + 0.15
        wm_marker.pose.orientation.w = 1.0
        wm_marker.scale.x = 0.35
        wm_marker.scale.y = 0.35
        wm_marker.scale.z = 0.30
        wm_marker.color.r = 0.2
        wm_marker.color.g = 0.6
        wm_marker.color.b = 1.0
        wm_marker.color.a = 0.45
        wm_marker.lifetime = Duration(seconds=2.0).to_msg()
        marker_array.markers.append(wm_marker)


def append_delete_markers(
    marker_array: MarkerArray,
    frame_id: str,
    namespace: str,
    marker_ids: Iterable[int],
    stamp,
):
    """Remove the given markers of one namespace from RViz."""
    for marker_id in marker_ids:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.DELETE
        marker_array.markers.append(marker)


_STATES = {
    tracking.TENTATIVE: TrackMsg.STATE_TENTATIVE,
    tracking.CONFIRMED: TrackMsg.STATE_CONFIRMED,
    tracking.COASTING: TrackMsg.STATE_COASTING,
}
_MOTIONS = {
    tracking.UNKNOWN: TrackMsg.MOTION_UNKNOWN,
    tracking.STATIC: TrackMsg.MOTION_STATIC,
    tracking.DYNAMIC: TrackMsg.MOTION_DYNAMIC,
}


def build_track_array(
    tracker: 'tracking.Tracker',
    tracks: Iterable['tracking.Track'],
    frame_id: str,
    stamp,
    time: float,
) -> TrackArray:
    """Describe the given tracks, with the tracker's lifecycle state, as a TrackArray."""
    array = TrackArray()
    array.header.frame_id = frame_id
    array.header.stamp = stamp
    for track in tracks:
        msg = TrackMsg()
        msg.id = int(track.id)
        msg.class_name = str(track.class_name)
        msg.score = float(track.measurement.score)
        msg.state = _STATES[tracker.lifecycle(track)]
        msg.motion = _MOTIONS[track.motion]
        x, y, z = track.position
        msg.position.x, msg.position.y, msg.position.z = x, y, z
        msg.velocity.x, msg.velocity.y = track.velocity
        if len(track.measurement.size) == 3:
            size_x, size_y, size_z = (float(value) for value in track.measurement.size)
            msg.size.x, msg.size.y, msg.size.z = size_x, size_y, size_z
        msg.covariance = [float(value) for value in track.filter.P.flatten()]
        msg.age_s = float(time - track.created)
        msg.time_since_update_s = float(time - track.last_update)
        array.tracks.append(msg)
    return array
