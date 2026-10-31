#!/usr/bin/env python3
from __future__ import annotations

import math
import struct
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
from geometry_msgs.msg import Point, PointStamped
import message_filters
import numpy as np
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
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2, PointField
from std_srvs.srv import Trigger
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
import tf2_geometry_msgs  # noqa: F401 - registers geometry_msgs methods for TF2
_ = tf2_geometry_msgs

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    YOLO_AVAILABLE = False

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
        yolo_path = (
            self.get_parameter('yolo_model').get_parameter_value().string_value
        )
        self.img_size = (
            self.get_parameter('img_size').get_parameter_value().integer_value
        )
        self.autostart = (
            self.get_parameter('autostart').get_parameter_value().bool_value
        )

        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info(
            f"Loading YOLO model ({yolo_path}) | Depth source: '{self.depth_source}' | "
            f"Tracker: '{self.tracking_filter}'..."
        )
        if YOLO_AVAILABLE:
            self.model = YOLO(yolo_path)
        else:
            self.model = None
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
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.get_logger().info(
                f"Camera intrinsics cached: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )

    def extract_lidar_distance(
        self, angle_rad: float, scan: LaserScan
    ) -> Optional[float]:
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
        self, u: float, v: float, depth_image: np.ndarray
    ) -> Optional[float]:
        h, w = depth_image.shape[:2]
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            return None

        u_min, u_max = max(0, ui - 2), min(w, ui + 3)
        v_min, v_max = max(0, vi - 2), min(h, vi + 3)
        window = depth_image[v_min:v_max, u_min:u_max]

        valid_vals = window[
            (window > 0.1) & (window < 10.0) & ~np.isnan(window) & ~np.isinf(window)
        ]
        if len(valid_vals) == 0:
            return None
        return float(np.median(valid_vals))

    def transform_point_to_target(
        self,
        local_x: float,
        local_y: float,
        local_z: float,
        source_frame: str,
        stamp,
    ) -> Optional[Point]:
        pt = PointStamped()
        pt.header.frame_id = source_frame
        pt.header.stamp = stamp
        pt.point.x = local_x
        pt.point.y = local_y
        pt.point.z = local_z

        try:
            transformed = self.tf_buffer.transform(
                pt, self.target_frame, timeout=Duration(seconds=0.05)
            )
            return transformed.point
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ):
            try:
                pt.header.stamp = rclpy.time.Time().to_msg()
                transformed = self.tf_buffer.transform(
                    pt, self.target_frame, timeout=Duration(seconds=0.05)
                )
                return transformed.point
            except Exception as e:
                self.get_logger().debug(f"TF transformation error: {e}")
                return None

    def sync_lidar_callback(self, img_msg: Image, scan_msg: LaserScan):
        self.process_perception_frame(
            img_msg=img_msg,
            stamp=scan_msg.header.stamp,
            sensor_data=scan_msg,
            is_depth_image=False,
        )

    def sync_depth_callback(self, img_msg: Image, depth_msg: Image):
        cv_depth = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='passthrough'
        )
        depth_array = np.array(cv_depth, dtype=np.float32)
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

        if self.model is None or self.bridge is None:
            return

        with self.state_lock:
            start_time = time.perf_counter()
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            results = self.model.track(
                cv_image,
                persist=True,
                tracker='bytetrack.yaml',
                imgsz=self.img_size,
                verbose=False,
            )

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

            if VISION_MSGS_AVAILABLE and self.detection_3d_pub is not None:
                detection_3d_array = Detection3DArray()
                detection_3d_array.header.frame_id = self.target_frame
                detection_3d_array.header.stamp = stamp
            else:
                detection_3d_array = None

            if len(results[0].boxes) > 0:
                for box_idx, box in enumerate(results[0].boxes):
                    class_id = int(box.cls[0])
                    obj_name = self.model.names[class_id]
                    score = float(box.conf[0])

                    track_id = -1
                    if box.id is not None:
                        track_id = int(box.id[0])

                    base_marker_id = (
                        track_id if track_id != -1 else (10000 + box_idx)
                    )

                    x1, y1, x2, y2 = box.xyxy[0]
                    u = float((x1 + x2) / 2)
                    v = float((y1 + y2) / 2)

                    if is_depth_image:
                        distance = self.extract_depth_image_distance(
                            u, v, sensor_data
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
                        distance = self.extract_lidar_distance(
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

                    global_pt = self.transform_point_to_target(
                        local_x,
                        local_y,
                        local_z,
                        source_frame,
                        stamp,
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

                    # Append visualization markers
                    self.append_markers(
                        marker_array,
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
                        self.append_detection_3d(
                            detection_3d_array,
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
                self.append_world_model_markers(marker_array, stamp)

            if marker_array.markers:
                self.marker_pub.publish(marker_array)

            if dynamic_costmap_points:
                cloud_msg = self.build_pointcloud2(
                    dynamic_costmap_points, self.target_frame, stamp
                )
                self.costmap_obstacle_pub.publish(cloud_msg)

            if (
                detection_3d_array is not None
                and len(detection_3d_array.detections) > 0
            ):
                self.detection_3d_pub.publish(detection_3d_array)

            if self.show_debug_window:
                annotated_frame = results[0].plot()
                cv2.namedWindow("Robot's 3D Spatial Vision", cv2.WINDOW_NORMAL)
                cv2.imshow("Robot's 3D Spatial Vision", annotated_frame)
                cv2.waitKey(1)

            cycle_dt = time.perf_counter() - start_time
            self.last_inference_latency = cycle_dt * 1000.0
            self.cycle_times.append(cycle_dt)
            if len(self.cycle_times) > 30:
                self.cycle_times.pop(0)

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'speed_threshold':
                if param.value < 0.0:
                    return SetParametersResult(
                        successful=False,
                        reason="speed_threshold must be non-negative",
                    )
                self.speed_threshold = float(param.value)
                self.get_logger().info(
                    f"Dynamic reconfigure: speed_threshold set to {self.speed_threshold:.2f} m/s"
                )
            elif param.name == 'process_every_n_frames':
                if param.value < 1:
                    return SetParametersResult(
                        successful=False,
                        reason="process_every_n_frames must be >= 1",
                    )
                self.process_every_n = int(param.value)
                self.get_logger().info(
                    f"Dynamic reconfigure: process_every_n_frames set to {self.process_every_n}"
                )
            elif param.name == 'sync_slop':
                if param.value <= 0.0:
                    return SetParametersResult(
                        successful=False,
                        reason="sync_slop must be > 0.0",
                    )
                self.sync_slop = float(param.value)
                if hasattr(self, 'time_sync') and self.time_sync is not None:
                    self.time_sync.slop = Duration(seconds=self.sync_slop)
                self.get_logger().info(
                    f"Dynamic reconfigure: sync_slop set to {self.sync_slop:.3f} s"
                )
            elif param.name == 'ema_alpha':
                if not (0.0 < param.value <= 1.0):
                    return SetParametersResult(
                        successful=False,
                        reason="ema_alpha must be in range (0.0, 1.0]",
                    )
                self.ema_alpha = float(param.value)
                self.get_logger().info(
                    f"Dynamic reconfigure: ema_alpha set to {self.ema_alpha:.2f}"
                )
            elif param.name == 'tracking_filter':
                val = str(param.value).lower()
                if val not in ['ema', 'kalman']:
                    return SetParametersResult(
                        successful=False,
                        reason="tracking_filter must be 'ema' or 'kalman'",
                    )
                self.tracking_filter = val
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
        if hasattr(super(), 'on_shutdown'):
            return super().on_shutdown(state)
        return TransitionCallbackReturn.SUCCESS

    def build_pointcloud2(
        self,
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
        self,
        detection_array: Detection3DArray,
        track_id: int,
        obj_name: str,
        score: float,
        pt: Point,
        stamp,
    ):
        det = Detection3D()
        det.header.frame_id = self.target_frame
        det.header.stamp = stamp
        det.id = str(track_id) if track_id != -1 else ""

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = obj_name
        hyp.hypothesis.score = score
        hyp.pose.pose.position.x = pt.x
        hyp.pose.pose.position.y = pt.y
        hyp.pose.pose.position.z = pt.z + 0.25
        hyp.pose.pose.orientation.w = 1.0
        det.results.append(hyp)

        det.bbox.center.position.x = pt.x
        det.bbox.center.position.y = pt.y
        det.bbox.center.position.z = pt.z + 0.25
        det.bbox.center.orientation.w = 1.0
        det.bbox.size.x = 0.45
        det.bbox.size.y = 0.45
        det.bbox.size.z = 0.50

        detection_array.detections.append(det)

    def append_markers(
        self,
        marker_array: MarkerArray,
        base_marker_id: int,
        track_id: int,
        obj_name: str,
        state: str,
        pt: Point,
        velocity: float,
        vel_x: float,
        vel_y: float,
        stamp,
    ):
        base_id = base_marker_id

        cube = Marker()
        cube.header.frame_id = self.target_frame
        cube.header.stamp = stamp
        cube.ns = 'detected_objects'
        cube.id = base_id
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose.position.x = pt.x
        cube.pose.position.y = pt.y
        cube.pose.position.z = pt.z + 0.25
        cube.pose.orientation.w = 1.0
        cube.scale.x = 0.45
        cube.scale.y = 0.45
        cube.scale.z = 0.50
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
        text.header.frame_id = self.target_frame
        text.header.stamp = stamp
        text.ns = 'object_labels'
        text.id = base_id
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = pt.x
        text.pose.position.y = pt.y
        text.pose.position.z = pt.z + 0.65
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
            arrow.header.frame_id = self.target_frame
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
            traj.header.frame_id = self.target_frame
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

    def append_world_model_markers(self, marker_array: MarkerArray, stamp):
        for idx, (name_key, (wx, wy, wz)) in enumerate(
            self.world_model.items()
        ):
            wm_marker = Marker()
            wm_marker.header.frame_id = self.target_frame
            wm_marker.header.stamp = stamp
            wm_marker.ns = 'world_model_memory'
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


def main(args=None):
    rclpy.init(args=args)
    node = SpatialVisionNode()
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
