#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    from cv_bridge import CvBridge

    CV_BRIDGE_AVAILABLE = True
except Exception:
    cv2 = None
    CvBridge = None
    CV_BRIDGE_AVAILABLE = False
import message_filters
import numpy as np
from r_one import geometry, messages
from r_one.detectors import ultralytics_detector
from r_one.detectors.ultralytics_detector import resolve_model_path, UltralyticsDetector
from r_one.geometry import intrinsics_are_valid
from r_one.messages import Detection3DArray, VISION_MSGS_AVAILABLE
from r_one.parameters import PARAMETER_RULES, validate_parameter
from r_one.run_log import RunLog
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
try:
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
except ImportError:
    ExternalShutdownException = KeyboardInterrupt
    from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
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


def debug_window_supported() -> bool:
    """Check for OpenCV and a display, because a window without a display aborts the process."""
    has_display = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    return cv2 is not None and bool(has_display)


class KalmanTracker2D:
    def __init__(self, init_x: float, init_y: float):
        self.x = np.array([init_x, init_y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([0.1, 0.1, 1.0, 1.0])
        self.Q = np.diag([0.01, 0.01, 0.05, 0.05])
        self.R = np.diag([0.04, 0.04])
        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )
        self.missed_frames = 0

    def predict(self, dt: float) -> Tuple[float, float, float, float]:
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        self.missed_frames += 1
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])

    def update(
        self, meas_x: float, meas_y: float
    ) -> Tuple[float, float, float, float]:
        z = np.array([meas_x, meas_y], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        eye4 = np.eye(4, dtype=np.float64)
        self.P = (eye4 - K @ self.H) @ self.P
        self.missed_frames = 0
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])


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
        self.declare_parameter('ema_alpha', 0.35)
        self.declare_parameter('tracking_filter', 'ema')  # 'ema' or 'kalman'
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
        self.ema_alpha = (
            self.get_parameter('ema_alpha').get_parameter_value().double_value
        )
        self.tracking_filter = (
            self.get_parameter('tracking_filter')
            .get_parameter_value()
            .string_value.lower()
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
            f"Loading YOLO model ({yolo_path}) | Depth source: '{self.depth_source}' | "
            f"Tracker: '{self.tracking_filter}'..."
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

        if self.depth_source == 'depth_image':
            self.sub_depth = message_filters.Subscriber(
                self,
                Image,
                self.depth_topic,
                qos_profile=sensor_qos,
                callback_group=self.cb_group,
            )
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

        # --- Kinematics & Memory Buffers ---
        self.object_history: Dict[int, Tuple[float, ...]] = {}
        self.kalman_trackers: Dict[int, KalmanTracker2D] = {}
        self.world_model: Dict[str, Tuple[float, float, float]] = {}

        self.frame_counter = 0

        # --- Autostart Lifecycle Support ---
        if LIFECYCLE_AVAILABLE and self.autostart:
            try:
                self.trigger_configure()
                self.trigger_activate()
            except Exception as e:
                self.get_logger().debug(f"Lifecycle autostart transition: {e}")

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

    def sync_lidar_callback(self, img_msg: Image, scan_msg: LaserScan):
        self.process_perception_frame(
            img_msg=img_msg,
            stamp=scan_msg.header.stamp,
            sensor_data=scan_msg,
            is_depth_image=False,
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

        if depth_msg.encoding == '16UC1':
            depth_array = depth_array / 1000.0  # mm to meters

        self.process_perception_frame(
            img_msg=img_msg,
            stamp=depth_msg.header.stamp,
            sensor_data=depth_array,
            is_depth_image=True,
            depth_frame_id=depth_msg.header.frame_id,
        )

    def process_perception_frame(
        self,
        img_msg: Image,
        stamp,
        sensor_data,
        is_depth_image: bool,
        depth_frame_id: Optional[str] = None,
    ):
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n != 0:
            return

        if self.fx is None or self.cx is None:
            return

        if self.detector is None or self.bridge is None:
            return

        with self.state_lock:
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

            marker_array = MarkerArray()
            dynamic_costmap_points: List[Tuple[float, float, float]] = []
            detected_ids = set()
            logged_tracks = []

            if VISION_MSGS_AVAILABLE and self.detection_3d_pub is not None:
                detection_3d_array = Detection3DArray()
                detection_3d_array.header.frame_id = self.target_frame
                detection_3d_array.header.stamp = stamp
            else:
                detection_3d_array = None

            if detections:
                for box_idx, detection in enumerate(detections):
                    obj_name = detection.class_name
                    score = detection.score
                    track_id = detection.track_id

                    base_marker_id = (
                        track_id if track_id != -1 else (10000 + box_idx)
                    )

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
                        fy = self.fy if self.fy is not None else self.fx
                        cy = (
                            self.cy
                            if self.cy is not None
                            else (cv_image.shape[0] / 2.0)
                        )
                        local_x = (u - self.cx) * distance / self.fx
                        local_y = (v - cy) * distance / fy
                        local_z = distance
                        source_frame = (
                            depth_frame_id
                            if depth_frame_id
                            else 'camera_optical_link'
                        )
                    else:
                        angle_rad = math.atan((self.cx - u) / self.fx)
                        distance = geometry.extract_lidar_distance(
                            angle_rad, sensor_data
                        )
                        if distance is None:
                            continue
                        local_x = distance * math.cos(angle_rad)
                        local_y = distance * math.sin(angle_rad)
                        local_z = 0.0
                        source_frame = (
                            sensor_data.header.frame_id
                            if sensor_data.header.frame_id
                            else 'laser_frame'
                        )

                    global_pt = geometry.transform_point(
                        self.tf_buffer,
                        self.target_frame,
                        local_x,
                        local_y,
                        local_z,
                        source_frame,
                        stamp,
                        self.get_logger(),
                    )
                    if global_pt is None:
                        continue

                    velocity = 0.0
                    vel_x = 0.0
                    vel_y = 0.0

                    if track_id != -1:
                        detected_ids.add(track_id)
                        if self.tracking_filter == 'kalman':
                            if track_id not in self.kalman_trackers:
                                self.kalman_trackers[track_id] = KalmanTracker2D(
                                    global_pt.x, global_pt.y
                                )
                            else:
                                prev_t = (
                                    self.object_history[track_id][3]
                                    if track_id in self.object_history
                                    else current_time - 0.08
                                )
                                dt = max(0.01, min(current_time - prev_t, 0.5))
                                self.kalman_trackers[track_id].predict(dt)
                                _, _, vel_x, vel_y = self.kalman_trackers[
                                    track_id
                                ].update(global_pt.x, global_pt.y)
                                velocity = math.sqrt(vel_x**2 + vel_y**2)

                            self.object_history[track_id] = (
                                global_pt.x,
                                global_pt.y,
                                global_pt.z,
                                current_time,
                                vel_x,
                                vel_y,
                            )
                        else:
                            # EMA Velocity Smoothing
                            if track_id in self.object_history:
                                hist_entry = self.object_history[track_id]
                                prev_x = hist_entry[0]
                                prev_y = hist_entry[1]
                                prev_time = hist_entry[3]
                                prev_vx = (
                                    hist_entry[4]
                                    if len(hist_entry) > 4
                                    else 0.0
                                )
                                prev_vy = (
                                    hist_entry[5]
                                    if len(hist_entry) > 5
                                    else 0.0
                                )
                                has_prev_vel = len(hist_entry) > 4

                                dt = current_time - prev_time
                                if dt > 0.05:
                                    raw_vx = (global_pt.x - prev_x) / dt
                                    raw_vy = (global_pt.y - prev_y) / dt

                                    if has_prev_vel:
                                        vel_x = (
                                            self.ema_alpha * raw_vx
                                            + (1.0 - self.ema_alpha) * prev_vx
                                        )
                                        vel_y = (
                                            self.ema_alpha * raw_vy
                                            + (1.0 - self.ema_alpha) * prev_vy
                                        )
                                    else:
                                        vel_x = raw_vx
                                        vel_y = raw_vy

                                    velocity = math.sqrt(vel_x**2 + vel_y**2)
                                elif has_prev_vel:
                                    vel_x = prev_vx
                                    vel_y = prev_vy
                                    velocity = math.sqrt(vel_x**2 + vel_y**2)

                            self.object_history[track_id] = (
                                global_pt.x,
                                global_pt.y,
                                global_pt.z,
                                current_time,
                                vel_x,
                                vel_y,
                            )

                    state = (
                        'DYNAMIC'
                        if velocity > self.speed_threshold
                        else 'STATIC'
                    )

                    if state == 'STATIC' and track_id != -1:
                        self.world_model[f'{obj_name}_{track_id}'] = (
                            global_pt.x,
                            global_pt.y,
                            global_pt.z,
                        )
                    elif state == 'DYNAMIC' and track_id != -1:
                        self.world_model.pop(f'{obj_name}_{track_id}', None)

                        # Dynamic Costmap Injection
                        for t_ahead in [0.5, 1.0, 1.5, 2.0]:
                            proj_x = global_pt.x + vel_x * t_ahead
                            proj_y = global_pt.y + vel_y * t_ahead
                            proj_z = global_pt.z
                            dynamic_costmap_points.append(
                                (proj_x, proj_y, proj_z)
                            )
                            for a_idx in range(8):
                                ang = a_idx * (2.0 * math.pi / 8.0)
                                dynamic_costmap_points.append((
                                    proj_x + 0.25 * math.cos(ang),
                                    proj_y + 0.25 * math.sin(ang),
                                    proj_z,
                                ))

                    logged_tracks.append({
                        'id': track_id,
                        'class': obj_name,
                        'score': score,
                        'x': global_pt.x,
                        'y': global_pt.y,
                        'z': global_pt.z,
                        'vx': vel_x,
                        'vy': vel_y,
                        'state': state,
                        'box': [float(x1), float(y1), float(x2), float(y2)],
                    })

                    # Append visualization markers
                    messages.append_markers(
                        marker_array,
                        self.target_frame,
                        base_marker_id,
                        track_id,
                        obj_name,
                        state,
                        global_pt,
                        velocity,
                        vel_x,
                        vel_y,
                        stamp,
                    )

                    # Append standard 3D detections
                    if detection_3d_array is not None:
                        messages.append_detection_3d(
                            detection_3d_array,
                            self.target_frame,
                            track_id,
                            obj_name,
                            score,
                            global_pt,
                            stamp,
                        )

                    self.get_logger().info(
                        f"[{state}] ID:{track_id} {obj_name.upper()} | "
                        f"Global X:{global_pt.x:.2f}m Y:{global_pt.y:.2f}m | "
                        f"Speed: {velocity:.2f} m/s"
                    )

            # Kalman Occlusion Coasting for tracks temporarily lost
            if self.tracking_filter == 'kalman':
                for tid, tracker in list(self.kalman_trackers.items()):
                    if tid not in detected_ids:
                        if tracker.missed_frames <= 3:
                            dt = 0.08
                            cx, cy, cvx, cvy = tracker.predict(dt)
                            c_vel = math.sqrt(cvx**2 + cvy**2)
                            if c_vel > self.speed_threshold:
                                for t_ahead in [0.5, 1.0, 1.5, 2.0]:
                                    proj_x = cx + cvx * t_ahead
                                    proj_y = cy + cvy * t_ahead
                                    dynamic_costmap_points.append(
                                        (proj_x, proj_y, 0.0)
                                    )
                        else:
                            del self.kalman_trackers[tid]
                            self.object_history.pop(tid, None)

            if self.frame_counter % 20 == 0:
                messages.append_world_model_markers(
                    marker_array, self.target_frame, self.world_model, stamp
                )

            if marker_array.markers:
                self.marker_pub.publish(marker_array)

            if dynamic_costmap_points:
                cloud_msg = messages.build_pointcloud2(
                    dynamic_costmap_points, self.target_frame, stamp
                )
                self.costmap_obstacle_pub.publish(cloud_msg)

            if (
                detection_3d_array is not None
                and len(detection_3d_array.detections) > 0
            ):
                self.detection_3d_pub.publish(detection_3d_array)

            if self.show_debug_window and not debug_window_supported():
                self.show_debug_window = False
                self.get_logger().warn(
                    "show_debug_window turned off: OpenCV or a display is missing."
                )

            if self.show_debug_window:
                annotated_frame = self.detector.plot()
                cv2.namedWindow("Robot's 3D Spatial Vision", cv2.WINDOW_NORMAL)
                cv2.imshow("Robot's 3D Spatial Vision", annotated_frame)
                cv2.waitKey(1)

            cycle_dt = time.perf_counter() - start_time
            self.last_inference_latency = cycle_dt * 1000.0
            self.cycle_times.append(cycle_dt)
            if len(self.cycle_times) > 30:
                self.cycle_times.pop(0)

            self.run_log.write_frame(current_time, cycle_dt, logged_tracks, self.world_model)

    def parameters_callback(self, params):
        # Check every value first so a multi-parameter set applies fully or not at all.
        for param in params:
            error = validate_parameter(param.name, param.value)
            if error is not None:
                return SetParametersResult(successful=False, reason=error)

        for param in params:
            if param.name == 'speed_threshold':
                self.speed_threshold = float(param.value)
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
            elif param.name == 'ema_alpha':
                self.ema_alpha = float(param.value)
                self.get_logger().info(
                    f"Dynamic reconfigure: ema_alpha set to {self.ema_alpha:.2f}"
                )
            elif param.name == 'tracking_filter':
                self.tracking_filter = str(param.value).lower()
                self.get_logger().info(
                    f"Dynamic reconfigure: tracking_filter set to {self.tracking_filter}"
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
            if not self.cycle_times:
                fps = 0.0
                status.level = DiagnosticStatus.OK
                status.message = "Awaiting sensor stream"
            else:
                fps = len(self.cycle_times) / sum(self.cycle_times)
                if self.last_inference_latency > 100.0 or (0.0 < fps < 10.0):
                    status.level = DiagnosticStatus.WARN
                    status.message = (
                        f"High latency: {self.last_inference_latency:.1f}ms "
                        f"({fps:.1f} FPS)"
                    )
                else:
                    status.level = DiagnosticStatus.OK
                    status.message = "Nominal operation"

            status.values = [
                KeyValue(key="fps", value=f"{fps:.2f}"),
                KeyValue(
                    key="latency_ms", value=f"{self.last_inference_latency:.2f}"
                ),
                KeyValue(
                    key="active_tracks", value=str(len(self.object_history))
                ),
                KeyValue(
                    key="world_model_objects", value=str(len(self.world_model))
                ),
                KeyValue(key="depth_source", value=self.depth_source),
                KeyValue(key="tracking_filter", value=self.tracking_filter),
                KeyValue(
                    key="speed_threshold", value=f"{self.speed_threshold:.2f}"
                ),
                KeyValue(key="ema_alpha", value=f"{self.ema_alpha:.2f}"),
                KeyValue(
                    key="sync_delta_ms", value=f"{self.last_sync_delta_ms:.2f}"
                ),
            ]

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
            self.object_history.clear()
            self.kalman_trackers.clear()
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
            self.object_history.clear()
            self.kalman_trackers.clear()
        if hasattr(super(), 'on_cleanup'):
            return super().on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self.get_logger().info("Lifecycle: on_shutdown called.")
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
        if cv2 is not None:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
