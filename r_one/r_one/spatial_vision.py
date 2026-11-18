#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from dataclasses import replace
import math
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

try:
    import cv2
    from cv_bridge import CvBridge

    CV_BRIDGE_AVAILABLE = True
except Exception:
    cv2 = None
    CvBridge = None
    CV_BRIDGE_AVAILABLE = False
from geometry_msgs.msg import Point
import message_filters
import numpy as np
from r_one import geometry, messages
from r_one.detectors import ultralytics_detector
from r_one.detectors.ultralytics_detector import resolve_model_path, UltralyticsDetector
from r_one.frame_slot import FrameSlot
from r_one.geometry import intrinsics_are_valid
from r_one.messages import Detection3DArray, VISION_MSGS_AVAILABLE
from r_one.parameters import PARAMETER_RULES, validate_parameter
from r_one.run_log import detection_record, RunLog, track_record
from r_one.tracking import (
    COASTING, CONFIRMED, DYNAMIC, Measurement, TENTATIVE, Track, Tracker, TrackerConfig,
)
from r_one.world_model import WorldModel
from r_one_msgs.msg import TrackArray
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
try:
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
except ImportError:
    ExternalShutdownException = KeyboardInterrupt
    from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RosTime
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2
from std_srvs.srv import Trigger
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

# Optional import for diagnostic_msgs
try:
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

    DIAGNOSTICS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    DIAGNOSTICS_AVAILABLE = False
    DiagnosticArray = None
    DiagnosticStatus = None
    KeyValue = None

# Optional import for rclpy.lifecycle
try:
    from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

    LIFECYCLE_AVAILABLE = True
except ImportError:
    LIFECYCLE_AVAILABLE = False
    from rclpy.node import Node as LifecycleNode

    class TransitionCallbackReturn:
        SUCCESS = 0
        FAILURE = 1
        ERROR = 2


DEBUG_WINDOW = "Robot's 3D Spatial Vision"
# Diagnostics report frames finished per second over this window, and WARN below the minimum
# (the p95 rate gate, docs/timing_2c.md)
THROUGHPUT_WINDOW_S = 5.0
MIN_THROUGHPUT_HZ = 2.0


def debug_window_supported() -> bool:
    """Check for OpenCV and a display, because a window without a display aborts the process."""
    has_display = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    return cv2 is not None and bool(has_display)


def predicted_path_points(track: Track) -> List[Tuple[float, float, float]]:
    """Return costmap points along a moving track's path, 0.5 to 2 s ahead."""
    x, y, z = track.position
    vel_x, vel_y = track.velocity
    points = []
    for t_ahead in (0.5, 1.0, 1.5, 2.0):
        proj_x = x + vel_x * t_ahead
        proj_y = y + vel_y * t_ahead
        points.append((proj_x, proj_y, z))
        for a_idx in range(8):
            ang = a_idx * (2.0 * math.pi / 8.0)
            points.append((
                proj_x + 0.25 * math.cos(ang),
                proj_y + 0.25 * math.sin(ang),
                z,
            ))
    return points


class SpatialVisionNode(LifecycleNode):
    def __init__(self):
        super().__init__('spatial_vision')
        if CV_BRIDGE_AVAILABLE:
            self.bridge = CvBridge()
        else:
            self.bridge = None
            self.get_logger().error(
                "OpenCV/CvBridge failed to load (NumPy 2.x ABI incompatibility detected). "
                "Fix by running: pip3 install 'numpy<2'"
            )
        self.cb_group = ReentrantCallbackGroup()
        self.state_lock = threading.RLock()

        # --- Parameters ---
        self.declare_parameter('target_frame', 'odom')
        self.declare_parameter('depth_source', 'lidar')  # 'lidar' or 'depth_image'
        self.declare_parameter('depth_topic', 'camera/depth/image_raw')
        self.declare_parameter('speed_threshold', 0.15)
        self.declare_parameter('sync_slop', 0.08)
        self.declare_parameter('process_every_n_frames', 5)
        self.declare_parameter('show_debug_window', False)
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('img_size', 320)
        self.declare_parameter('autostart', True)
        self.declare_parameter('run_log_path', '')  # Empty turns the run log off

        self.target_frame = (
            self.get_parameter('target_frame').get_parameter_value().string_value
        )
        self.depth_source = (
            self.get_parameter('depth_source').get_parameter_value().string_value
        )
        self.depth_topic = (
            self.get_parameter('depth_topic').get_parameter_value().string_value
        )
        self.speed_threshold = (
            self.get_parameter('speed_threshold')
            .get_parameter_value()
            .double_value
        )
        self.sync_slop = (
            self.get_parameter('sync_slop').get_parameter_value().double_value
        )
        self.process_every_n = (
            self.get_parameter('process_every_n_frames')
            .get_parameter_value()
            .integer_value
        )
        self.show_debug_window = (
            self.get_parameter('show_debug_window')
            .get_parameter_value()
            .bool_value
        )
        yolo_path = resolve_model_path(
            self.get_parameter('yolo_model').get_parameter_value().string_value
        )
        self.img_size = (
            self.get_parameter('img_size').get_parameter_value().integer_value
        )
        self.autostart = (
            self.get_parameter('autostart').get_parameter_value().bool_value
        )

        for name in PARAMETER_RULES:
            error = validate_parameter(name, self.get_parameter(name).value)
            if error is not None:
                raise ValueError(error)

        self.run_log = RunLog(
            self.get_parameter('run_log_path').get_parameter_value().string_value
        )

        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info(
            f"Loading YOLO model ({yolo_path}) | Depth source: '{self.depth_source}'..."
        )
        if ultralytics_detector.YOLO_AVAILABLE:
            self.detector = UltralyticsDetector(yolo_path, self.img_size)
        else:
            self.detector = None
            self.get_logger().error(
                "YOLO failed to load (NumPy 2.x ABI incompatibility detected). "
                "Fix by running: pip3 install 'numpy<2'"
            )

        # --- TF2 Transform Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Camera Intrinsics Subscriber ---
        self.sub_info = self.create_subscription(
            CameraInfo,
            'camera/camera_info',
            self.info_callback,
            10,
            callback_group=self.cb_group,
        )
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        # --- Deterministic QoS Profiles ---
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        output_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        costmap_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        # --- Synchronized Subscriptions ---
        self.sub_img = message_filters.Subscriber(
            self,
            Image,
            'camera/image_raw',
            qos_profile=sensor_qos,
            callback_group=self.cb_group,
        )
        # Count every message each synced topic delivers, before the synchronizer pairs them
        self.received = {'images_received': 0, 'range_msgs_received': 0}
        self.received_lock = threading.Lock()
        self.sub_img.registerCallback(self.count_received, 'images_received')

        if self.depth_source == 'depth_image':
            self.sub_depth = message_filters.Subscriber(
                self,
                Image,
                self.depth_topic,
                qos_profile=sensor_qos,
                callback_group=self.cb_group,
            )
            self.sub_depth.registerCallback(self.count_received, 'range_msgs_received')
            self.time_sync = message_filters.ApproximateTimeSynchronizer(
                [self.sub_img, self.sub_depth],
                queue_size=10,
                slop=self.sync_slop,
            )
            self.time_sync.registerCallback(self.sync_depth_callback)
            self.get_logger().info(
                f"Synchronizing RGB with Depth Image: '{self.depth_topic}'"
            )
        else:
            self.sub_scan = message_filters.Subscriber(
                self,
                LaserScan,
                'scan',
                qos_profile=sensor_qos,
                callback_group=self.cb_group,
            )
            self.sub_scan.registerCallback(self.count_received, 'range_msgs_received')
            self.time_sync = message_filters.ApproximateTimeSynchronizer(
                [self.sub_img, self.sub_scan],
                queue_size=10,
                slop=self.sync_slop,
            )
            self.time_sync.registerCallback(self.sync_lidar_callback)
            self.get_logger().info("Synchronizing RGB with 2D LiDAR: 'scan'")

        # --- Publishers ---
        self.marker_pub = self.create_publisher(
            MarkerArray, 'perception/markers', output_qos
        )

        self.costmap_obstacle_pub = self.create_publisher(
            PointCloud2, 'perception/dynamic_costmap_obstacles', costmap_qos
        )

        # Every live track with its lifecycle state, published every processed frame
        self.track_pub = self.create_publisher(
            TrackArray, 'perception/tracks', output_qos
        )

        if VISION_MSGS_AVAILABLE:
            self.detection_3d_pub = self.create_publisher(
                Detection3DArray, 'perception/detections_3d', output_qos
            )
        else:
            self.detection_3d_pub = None
            self.get_logger().warn(
                "vision_msgs is not installed. For 3D detections on perception/detections_3d, "
                "install via: sudo apt install ros-humble-vision-msgs"
            )

        # --- Operator Management Services ---
        self.srv_clear_world = self.create_service(
            Trigger,
            '~/clear_world_model',
            self.handle_clear_world_model,
            callback_group=self.cb_group,
        )
        self.srv_reset_tracker = self.create_service(
            Trigger,
            '~/reset_tracker',
            self.handle_reset_tracker,
            callback_group=self.cb_group,
        )

        # --- Diagnostics Telemetry ---
        self.last_inference_latency: float = 0.0
        self.cycle_times: List[float] = []
        self.last_sync_delta_ms: float = 0.0
        # Wall-clock times of finished frames over the last THROUGHPUT_WINDOW_S
        self.finished_frames = deque()
        self.first_finished_frame: Optional[float] = None
        self.last_range_gap_ms: float = 0.0
        if DIAGNOSTICS_AVAILABLE:
            self.diag_pub = self.create_publisher(
                DiagnosticArray, '/diagnostics', 10
            )
            self.diag_timer = self.create_timer(
                1.0, self.publish_diagnostics, callback_group=self.cb_group
            )
        else:
            self.diag_pub = None
            self.diag_timer = None

        # --- Tracker & Memory ---
        self.tracker = Tracker(TrackerConfig(speed_threshold_mps=self.speed_threshold))
        self.world_model = WorldModel()
        # World-model markers drawn last frame, so entries that leave get deleted.
        self.world_marker_count = 0

        self.frame_counter = 0
        self.dropped_frames = 0
        self.tf_counts = {'tf_failures': 0}
        # Scan-to-camera matrices, keyed by (scan frame, camera frame)
        self.laser_camera_cache = {}

        # --- Autostart Lifecycle Support ---
        if LIFECYCLE_AVAILABLE and self.autostart:
            try:
                self.trigger_configure()
                self.trigger_activate()
            except Exception as e:
                self.get_logger().debug(f"Lifecycle autostart transition: {e}")

        # Inference runs on its own thread, so callbacks, TF and timers never wait for it
        self.debug_window_open = False
        self.frame_slot = FrameSlot()
        self.worker = threading.Thread(
            target=self.frame_worker, name='spatial_vision_worker', daemon=True
        )
        self.worker.start()

        self.get_logger().info(
            f"Spatial Vision initialized with target_frame: '{self.target_frame}'."
        )

    def info_callback(self, msg: CameraInfo):
        if self.fx is None or self.cx is None:
            if not intrinsics_are_valid(msg.k):
                self.get_logger().warn(
                    "Ignoring camera_info with an invalid K matrix. Calibrate the "
                    "camera: nothing is detected until a valid K arrives.",
                    throttle_duration_sec=10.0,
                )
                return
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.get_logger().info(
                f"Camera intrinsics cached: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )

    def count_received(self, msg, key: str):
        # Subscription callbacks can run on several executor threads at once
        with self.received_lock:
            self.received[key] += 1

    def sync_lidar_callback(self, img_msg: Image, scan_msg: LaserScan):
        # The boxes come from the image, so place them with the image's stamp. The scan can
        # be up to sync_slop older or newer.
        self.submit_frame(
            img_msg=img_msg,
            stamp=img_msg.header.stamp,
            sensor_data=scan_msg,
            is_depth_image=False,
            range_stamp=scan_msg.header.stamp,
        )

    def sync_depth_callback(self, img_msg: Image, depth_msg: Image):
        if self.bridge is None:
            return

        try:
            cv_depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough'
            )
            depth_array = np.array(cv_depth, dtype=np.float32)
        except Exception as exc:
            self.get_logger().error(
                f"Skipping frame, could not read the depth image: {exc}",
                throttle_duration_sec=5.0,
            )
            return

        if depth_msg.encoding in ('16UC1', 'mono16'):
            depth_array = depth_array / 1000.0  # mm to meters

        # Ranging reads the depth at the box's pixels, so both images must be the same size
        if depth_array.shape[:2] != (img_msg.height, img_msg.width):
            self.get_logger().error(
                f"Skipping frame: the depth image is {depth_array.shape[1]}x"
                f"{depth_array.shape[0]} but the colour image is "
                f"{img_msg.width}x{img_msg.height}. Publish a depth image registered "
                "to the colour camera.",
                throttle_duration_sec=5.0,
            )
            return

        # The boxes come from the colour image, so place them with its stamp, as in lidar mode
        self.submit_frame(
            img_msg=img_msg,
            stamp=img_msg.header.stamp,
            sensor_data=depth_array,
            is_depth_image=True,
            depth_frame_id=depth_msg.header.frame_id,
            range_stamp=depth_msg.header.stamp,
        )

    def submit_frame(self, **frame):
        """Hand a synced frame set to the worker, keeping only every n-th and the newest."""
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n != 0:
            return
        if self.frame_slot.put(frame):
            self.dropped_frames += 1

    def frame_worker(self):
        """Warm up the detector, then process the newest frame until the slot closes."""
        self.warm_up_detector()
        try:
            while not self.frame_slot.closed:
                frame = self.frame_slot.take(timeout=0.5)
                if frame is None:
                    continue
                try:
                    self.sync_debug_window()
                    self.process_perception_frame(**frame)
                except Exception as exc:
                    # The worker must survive, or the node would stay up and stop perceiving
                    self.get_logger().error(
                        f"Perception frame failed: {exc!r}", throttle_duration_sec=5.0
                    )
        finally:
            if self.debug_window_open:
                self.debug_window_open = False
                cv2.destroyWindow(DEBUG_WINDOW)
                cv2.waitKey(1)

    def sync_debug_window(self):
        """
        Open or close the debug window to match show_debug_window.

        Only the worker thread calls this and draws the window, so every OpenCV window
        call happens on one thread.
        """
        if self.show_debug_window == self.debug_window_open:
            return
        if self.debug_window_open:
            self.debug_window_open = False
            cv2.destroyWindow(DEBUG_WINDOW)
            cv2.waitKey(1)
        elif debug_window_supported():
            cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
            self.debug_window_open = True
        else:
            self.show_debug_window = False
            self.get_logger().warn(
                "show_debug_window turned off: OpenCV or a display is missing."
            )

    def warm_up_detector(self):
        """Run the detector once, so the first real frame does not pay for model setup."""
        if self.detector is None:
            return
        start = time.perf_counter()
        try:
            self.detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        except Exception as exc:
            self.get_logger().warn(f"Detector warm-up failed: {exc!r}")
            return
        self.get_logger().info(
            f"Detector warmed up in {time.perf_counter() - start:.1f} s"
        )

    def laser_to_camera(self, scan_frame: str, camera_frame: str) -> Optional[np.ndarray]:
        """
        Return the 4×4 matrix that moves scan points into the camera frame, or None.

        The lidar and the camera are both fixed to the robot, so this transform does not
        change with time. It is looked up once, from the latest TF, and cached.
        """
        key = (scan_frame, camera_frame)
        matrix = self.laser_camera_cache.get(key)
        if matrix is not None:
            return matrix
        try:
            transform = self.tf_buffer.lookup_transform(camera_frame, scan_frame, RosTime())
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(
                f"No transform from '{scan_frame}' to '{camera_frame}' yet, so lidar "
                f"detections are not ranged: {exc}",
                throttle_duration_sec=5.0,
            )
            return None
        matrix = geometry.transform_to_matrix(transform.transform)
        self.laser_camera_cache[key] = matrix
        self.get_logger().info(f"Cached the '{scan_frame}' to '{camera_frame}' transform")
        return matrix

    def stop_worker(self):
        """Stop the perception worker and wait for its current frame to finish."""
        self.frame_slot.close()
        if self.worker.is_alive() and self.worker is not threading.current_thread():
            self.worker.join(timeout=10.0)

    def destroy_node(self):
        self.stop_worker()
        return super().destroy_node()

    def process_perception_frame(
        self,
        img_msg: Image,
        stamp,
        sensor_data,
        is_depth_image: bool,
        depth_frame_id: Optional[str] = None,
        range_stamp=None,
    ):
        if self.fx is None or self.cx is None:
            return

        if self.detector is None or self.bridge is None:
            return

        start_time = time.perf_counter()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            detections = self.detector.detect(cv_image)
        except Exception as exc:
            self.get_logger().error(
                f"Skipping frame, detection failed: {exc}",
                throttle_duration_sec=5.0,
            )
            return

        stamp_sec = stamp.sec + stamp.nanosec * 1e-9
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if stamp_sec > 0.0:
            self.last_sync_delta_ms = abs(now_sec - stamp_sec) * 1000.0
            current_time = stamp_sec
        else:
            current_time = now_sec

        if VISION_MSGS_AVAILABLE and self.detection_3d_pub is not None:
            detection_3d_array = Detection3DArray()
            detection_3d_array.header.frame_id = self.target_frame
            detection_3d_array.header.stamp = stamp
        else:
            detection_3d_array = None

        # Move the scan into the camera once per frame, so each box picks the beams that
        # project into it
        points_cam = None
        if not is_depth_image:
            camera_frame = img_msg.header.frame_id or 'camera_optical_link'
            scan_frame = sensor_data.header.frame_id or 'laser_frame'
            to_camera = self.laser_to_camera(scan_frame, camera_frame)
            if to_camera is not None:
                points_cam = geometry.transform_points(
                    to_camera, geometry.scan_points(sensor_data)
                )

        measurements: List[Measurement] = []
        ranged: List[bool] = []
        for detection in detections:
            ranged.append(False)
            x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
            u = float((x1 + x2) / 2)
            v = float((y1 + y2) / 2)

            if is_depth_image:
                # Sample the middle third of the box
                distance = geometry.extract_depth_image_distance(
                    u,
                    v,
                    sensor_data,
                    half_width=float(x2 - x1) / 6.0,
                    half_height=float(y2 - y1) / 6.0,
                )
                if distance is None:
                    continue
                local_x = (u - self.cx) * distance / self.fx
                local_y = (v - self.cy) * distance / self.fy
                local_z = distance
                source_frame = (
                    depth_frame_id
                    if depth_frame_id
                    else 'camera_optical_link'
                )
            else:
                if points_cam is None:
                    continue
                centroid = geometry.nearest_cluster(geometry.points_in_box(
                    points_cam, (x1, y1, x2, y2), self.fx, self.fy, self.cx, self.cy
                ))
                if centroid is None:
                    continue
                local_x, local_y, local_z = (float(value) for value in centroid)
                source_frame = camera_frame

            global_pt = geometry.transform_point(
                self.tf_buffer,
                self.target_frame,
                local_x,
                local_y,
                local_z,
                source_frame,
                stamp,
                self.get_logger(),
                self.tf_counts,
            )
            if global_pt is None:
                continue

            ranged[-1] = True
            box = (float(x1), float(y1), float(x2), float(y2))
            measurements.append(Measurement(
                global_pt.x,
                global_pt.y,
                global_pt.z,
                detection.class_name,
                detection.score,
                box,
                geometry.measured_size(box, local_z, self.fx, self.fy),
            ))

        with self.state_lock:
            tracks = self.tracker.step(current_time, measurements)
            for track in tracks:
                self.world_model.observe(track, current_time)

            marker_array = MarkerArray()
            for track in tracks:
                x, y, z = track.position
                vel_x, vel_y = track.velocity
                point = Point(x=x, y=y, z=z)
                messages.append_markers(
                    marker_array,
                    self.target_frame,
                    track.id,
                    track.id,
                    track.class_name,
                    track.motion,
                    point,
                    track.speed,
                    vel_x,
                    vel_y,
                    stamp,
                    size=track.measurement.size,
                )
                if detection_3d_array is not None:
                    messages.append_detection_3d(
                        detection_3d_array,
                        self.target_frame,
                        track.id,
                        track.class_name,
                        track.measurement.score,
                        point,
                        stamp,
                        size=track.measurement.size,
                    )
                self.get_logger().debug(
                    f"[{track.motion}] ID:{track.id} {track.class_name.upper()} | "
                    f"Global X:{x:.2f}m Y:{y:.2f}m | Speed: {track.speed:.2f} m/s"
                )

            # Moving tracks feed the costmap while they coast through a short occlusion too.
            dynamic_costmap_points: List[Tuple[float, float, float]] = []
            for track in tracks + self.tracker.coasting_dynamic():
                if track.motion == DYNAMIC:
                    dynamic_costmap_points.extend(predicted_path_points(track))

            for namespace in messages.TRACK_MARKER_NAMESPACES:
                messages.append_delete_markers(
                    marker_array, self.target_frame, namespace,
                    self.tracker.ended_ids(), stamp,
                )
            messages.append_world_model_markers(
                marker_array, self.target_frame, self.world_model, stamp
            )
            messages.append_delete_markers(
                marker_array, self.target_frame, messages.WORLD_MODEL_NAMESPACE,
                range(len(self.world_model), self.world_marker_count), stamp,
            )
            self.world_marker_count = len(self.world_model)

            if marker_array.markers:
                self.marker_pub.publish(marker_array)

            # The cloud, 3D detections and tracks go out every frame, empty when there is
            # nothing, so consumers drop what has gone. Nav2's ObstacleLayer still keeps its
            # marks until a clearing source raytraces through them (docs/outputs_2e.md).
            self.costmap_obstacle_pub.publish(messages.build_pointcloud2(
                dynamic_costmap_points, self.target_frame, stamp
            ))
            if detection_3d_array is not None:
                self.detection_3d_pub.publish(detection_3d_array)
            self.track_pub.publish(messages.build_track_array(
                self.tracker, self.tracker.tracks, self.target_frame, stamp, current_time
            ))

            if self.debug_window_open:
                cv2.imshow(DEBUG_WINDOW, self.detector.plot())
                cv2.waitKey(1)

            cycle_dt = time.perf_counter() - start_time
            self.last_inference_latency = cycle_dt * 1000.0
            self.cycle_times.append(cycle_dt)
            if len(self.cycle_times) > 30:
                self.cycle_times.pop(0)
            finished = time.monotonic()
            self.finished_frames.append(finished)
            if self.first_finished_frame is None:
                self.first_finished_frame = finished
            if range_stamp is not None:
                range_sec = range_stamp.sec + range_stamp.nanosec * 1e-9
                self.last_range_gap_ms = abs(stamp_sec - range_sec) * 1000.0

            self.run_log.write_frame(
                current_time, cycle_dt, [track_record(track) for track in tracks],
                self.world_model,
                [detection_record(d, r) for d, r in zip(detections, ranged)],
                {
                    **self.received,
                    'synced_frames': self.frame_counter,
                    'dropped_frames': self.dropped_frames,
                    **self.tf_counts,
                },
            )

    def parameters_callback(self, params):
        # Check every value first so a multi-parameter set applies fully or not at all.
        for param in params:
            error = validate_parameter(param.name, param.value)
            if error is not None:
                return SetParametersResult(successful=False, reason=error)

        for param in params:
            if param.name == 'speed_threshold':
                self.speed_threshold = float(param.value)
                self.tracker.config = replace(
                    self.tracker.config, speed_threshold_mps=self.speed_threshold
                )
                self.get_logger().info(
                    f"Dynamic reconfigure: speed_threshold set to {self.speed_threshold:.2f} m/s"
                )
            elif param.name == 'process_every_n_frames':
                self.process_every_n = int(param.value)
                self.get_logger().info(
                    f"Dynamic reconfigure: process_every_n_frames set to {self.process_every_n}"
                )
            elif param.name == 'sync_slop':
                self.sync_slop = float(param.value)
                if hasattr(self, 'time_sync') and self.time_sync is not None:
                    self.time_sync.slop = Duration(seconds=self.sync_slop)
                self.get_logger().info(
                    f"Dynamic reconfigure: sync_slop set to {self.sync_slop:.3f} s"
                )
            elif param.name == 'show_debug_window':
                self.show_debug_window = bool(param.value)
                self.get_logger().info(
                    f"Dynamic reconfigure: show_debug_window set to {self.show_debug_window}"
                )
        return SetParametersResult(successful=True)

    def publish_diagnostics(self):
        if not DIAGNOSTICS_AVAILABLE or self.diag_pub is None:
            return

        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        status = DiagnosticStatus()
        status.name = f"{self.get_name()}: Perception Pipeline"
        status.hardware_id = "intel_cpu_spatial_vision"

        with self.state_lock:
            now = time.monotonic()
            while self.finished_frames and self.finished_frames[0] < now - THROUGHPUT_WINDOW_S:
                self.finished_frames.popleft()
            throughput_hz = len(self.finished_frames) / THROUGHPUT_WINDOW_S
            # The rate needs a full window of history before it can be judged
            rate_known = (
                self.first_finished_frame is not None
                and now - self.first_finished_frame >= THROUGHPUT_WINDOW_S
            )
            if not self.cycle_times:
                fps = 0.0
                status.level = DiagnosticStatus.OK
                status.message = "Awaiting sensor stream"
            else:
                fps = len(self.cycle_times) / sum(self.cycle_times)
                if self.last_inference_latency > 100.0:
                    status.level = DiagnosticStatus.WARN
                    status.message = (
                        f"High latency: {self.last_inference_latency:.1f}ms "
                        f"({throughput_hz:.1f} Hz)"
                    )
                elif rate_known and throughput_hz < MIN_THROUGHPUT_HZ:
                    status.level = DiagnosticStatus.WARN
                    status.message = (
                        f"Low rate: {throughput_hz:.1f} Hz over the last "
                        f"{THROUGHPUT_WINDOW_S:.0f} s"
                    )
                else:
                    status.level = DiagnosticStatus.OK
                    status.message = "Nominal operation"

            states = [self.tracker.lifecycle(track) for track in self.tracker.tracks]
            dynamic = sum(track.motion == DYNAMIC for track in self.tracker.tracks)

            status.values = [
                KeyValue(key="fps", value=f"{fps:.2f}"),
                KeyValue(
                    key="latency_ms", value=f"{self.last_inference_latency:.2f}"
                ),
                KeyValue(
                    key="active_tracks", value=str(len(self.tracker.tracks))
                ),
                KeyValue(
                    key="world_model_objects", value=str(len(self.world_model))
                ),
                KeyValue(key="depth_source", value=self.depth_source),
                KeyValue(
                    key="speed_threshold", value=f"{self.speed_threshold:.2f}"
                ),
                KeyValue(
                    key="sync_delta_ms", value=f"{self.last_sync_delta_ms:.2f}"
                ),
                KeyValue(key="throughput_hz", value=f"{throughput_hz:.2f}"),
                KeyValue(key="rgb_range_gap_ms", value=f"{self.last_range_gap_ms:.2f}"),
                KeyValue(key="tentative_tracks", value=str(states.count(TENTATIVE))),
                KeyValue(key="confirmed_tracks", value=str(states.count(CONFIRMED))),
                KeyValue(key="coasting_tracks", value=str(states.count(COASTING))),
                KeyValue(key="dynamic_tracks", value=str(dynamic)),
                KeyValue(key="dropped_frames", value=str(self.dropped_frames)),
                KeyValue(key="tf_failures", value=str(self.tf_counts['tf_failures'])),
            ]
            with self.received_lock:
                status.values.extend(
                    KeyValue(key=key, value=str(count))
                    for key, count in self.received.items()
                )

        diag_array.status.append(status)
        self.diag_pub.publish(diag_array)

    def handle_clear_world_model(self, request, response):
        with self.state_lock:
            self.world_model.clear()
            delete_array = MarkerArray()
            del_marker = Marker()
            del_marker.action = Marker.DELETEALL
            delete_array.markers.append(del_marker)
            self.marker_pub.publish(delete_array)
            self.get_logger().info(
                "World model cleared and RViz markers wiped via service."
            )
            response.success = True
            response.message = (
                "World model cleared and RViz markers wiped successfully"
            )
            return response

    def handle_reset_tracker(self, request, response):
        with self.state_lock:
            self.tracker.reset()
            self.get_logger().info(
                "Perception tracker state reset via service."
            )
            response.success = True
            response.message = "Tracker state reset successfully"
            return response

    def on_configure(self, state):
        self.get_logger().info("Lifecycle: on_configure called.")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info("Lifecycle: on_activate called.")
        if hasattr(super(), 'on_activate'):
            return super().on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info("Lifecycle: on_deactivate called.")
        if hasattr(super(), 'on_deactivate'):
            return super().on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info("Lifecycle: on_cleanup called.")
        with self.state_lock:
            self.world_model.clear()
            self.tracker.reset()
        if hasattr(super(), 'on_cleanup'):
            return super().on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self.get_logger().info("Lifecycle: on_shutdown called.")
        self.stop_worker()
        self.run_log.close()
        if hasattr(super(), 'on_shutdown'):
            return super().on_shutdown(state)
        return TransitionCallbackReturn.SUCCESS


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SpatialVisionNode()
    except ValueError as exc:
        print(f"spatial_vision: invalid parameter: {exc}", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Stop the worker first: it owns the debug window, and this runs on another thread
        node.destroy_node()
        if cv2 is not None:
            cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
