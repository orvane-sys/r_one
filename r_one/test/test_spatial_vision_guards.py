"""Regression tests for the failure modes that used to kill or stall spatial_vision."""

from collections import deque
import os
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Quaternion, Transform, Vector3
import numpy as np
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image, LaserScan
import tf2_ros

from r_one import geometry, spatial_vision
from r_one.detectors import ultralytics_detector
from r_one.detectors.base import Detection
from r_one.frame_slot import FrameSlot
from r_one.spatial_vision import (
    DEBUG_WINDOW,
    debug_window_supported,
    intrinsics_are_valid,
    SpatialVisionNode,
    validate_parameter,
)
from r_one.tracking import Measurement, Tracker, TrackerConfig
from r_one.world_model import WorldModel
from r_one_msgs.msg import Track as TrackMsg


def camera_info(k):
    msg = CameraInfo()
    msg.k = [float(value) for value in k]
    return msg


def fake_node(**overrides):
    """Build the attributes that the node methods under test read from self."""
    logger = mock.Mock()
    clock = mock.Mock()
    clock.now.return_value.nanoseconds = 0
    clock.now.return_value.to_msg.return_value = Time()
    attrs = dict(
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        bridge=mock.Mock(),
        detector=mock.Mock(),
        state_lock=threading.RLock(),
        frame_counter=0,
        dropped_frames=0,
        frame_slot=FrameSlot(),
        tf_counts={'tf_failures': 0},
        received={'images_received': 0, 'range_msgs_received': 0},
        received_lock=threading.Lock(),
        process_every_n=1,
        img_size=320,
        speed_threshold=0.15,
        sync_slop=0.08,
        show_debug_window=False,
        debug_window_open=False,
        sync_debug_window=mock.Mock(),
        time_sync=None,
        get_logger=lambda: logger,
        get_clock=lambda: clock,
        get_name=lambda: 'spatial_vision',
        depth_source='depth_image',
        diag_pub=mock.Mock(),
        process_perception_frame=mock.Mock(),
        submit_frame=mock.Mock(),
        run_log=mock.Mock(),
        tracker=Tracker(TrackerConfig(speed_threshold_mps=0.15)),
        world_model=WorldModel(),
        world_marker_count=0,
        target_frame='odom',
        tf_buffer=mock.Mock(),
        marker_pub=mock.Mock(),
        costmap_obstacle_pub=mock.Mock(),
        track_pub=mock.Mock(),
        detection_3d_pub=None,
        cycle_times=[],
        last_sync_delta_ms=0.0,
        last_inference_latency=0.0,
        finished_frames=deque(),
        first_finished_frame=None,
        last_range_gap_ms=0.0,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class TestIntrinsics(unittest.TestCase):
    def test_all_zero_k_is_rejected(self):
        self.assertFalse(intrinsics_are_valid([0.0] * 9))

    def test_non_finite_or_negative_focal_length_is_rejected(self):
        self.assertFalse(intrinsics_are_valid([float('nan'), 0, 320, 0, 500, 240, 0, 0, 1]))
        self.assertFalse(intrinsics_are_valid([500, 0, 320, 0, -1, 240, 0, 0, 1]))
        self.assertFalse(intrinsics_are_valid([500, 0, float('inf'), 0, 500, 240, 0, 0, 1]))

    def test_calibrated_k_is_accepted(self):
        self.assertTrue(intrinsics_are_valid([500, 0, 320, 0, 500, 240, 0, 0, 1]))

    def test_zero_k_is_not_cached_and_warns(self):
        node = fake_node()
        SpatialVisionNode.info_callback(node, camera_info([0] * 9))
        self.assertIsNone(node.fx)
        node.get_logger().warn.assert_called_once()

    def test_valid_k_after_zero_k_is_cached(self):
        node = fake_node()
        SpatialVisionNode.info_callback(node, camera_info([0] * 9))
        SpatialVisionNode.info_callback(
            node, camera_info([500, 0, 320, 0, 501, 240, 0, 0, 1])
        )
        self.assertEqual((node.fx, node.fy, node.cx, node.cy), (500.0, 501.0, 320.0, 240.0))


class TestParameterValidation(unittest.TestCase):
    def test_invalid_values_are_reported(self):
        for name, value in (
            ('process_every_n_frames', 0),
            ('img_size', 0),
            ('speed_threshold', -0.1),
            ('speed_threshold', float('nan')),
            ('sync_slop', 0.0),
            ('depth_source', 'sonar'),
        ):
            with self.subTest(name=name, value=value):
                self.assertIsNotNone(validate_parameter(name, value))

    def test_valid_values_pass(self):
        for name, value in (
            ('process_every_n_frames', 1),
            ('img_size', 320),
            ('speed_threshold', 0.0),
            ('sync_slop', 0.08),
            ('depth_source', 'depth_image'),
            ('target_frame', 'anything'),
        ):
            with self.subTest(name=name, value=value):
                self.assertIsNone(validate_parameter(name, value))

    def test_rejected_multi_parameter_set_applies_nothing(self):
        node = fake_node()
        result = SpatialVisionNode.parameters_callback(node, [
            Parameter('speed_threshold', value=0.9),
            Parameter('process_every_n_frames', value=0),
        ])
        self.assertFalse(result.successful)
        self.assertEqual(node.speed_threshold, 0.15)
        self.assertEqual(node.tracker.config.speed_threshold_mps, 0.15)
        self.assertEqual(node.process_every_n, 1)

    def test_valid_multi_parameter_set_applies_everything(self):
        node = fake_node()
        result = SpatialVisionNode.parameters_callback(node, [
            Parameter('speed_threshold', value=0.9),
            Parameter('process_every_n_frames', value=3),
        ])
        self.assertTrue(result.successful)
        self.assertEqual((node.speed_threshold, node.process_every_n), (0.9, 3))
        self.assertEqual(node.tracker.config.speed_threshold_mps, 0.9)


class TestNodeStartup(unittest.TestCase):
    def assert_startup_rejected(self, ros_argument):
        rclpy.init(args=['--ros-args', '-p', ros_argument])
        try:
            with mock.patch.object(ultralytics_detector, 'YOLO', None), \
                    mock.patch.object(ultralytics_detector, 'YOLO_AVAILABLE', False):
                with self.assertRaises(ValueError):
                    SpatialVisionNode()
        finally:
            rclpy.shutdown()

    def test_zero_frame_interval_is_rejected(self):
        self.assert_startup_rejected('process_every_n_frames:=0')

    def test_unknown_depth_source_is_rejected(self):
        self.assert_startup_rejected('depth_source:=sonar')

    def test_removed_tracking_filter_is_ignored(self):
        # tracking_filter was removed in 2d; old launch commands that still pass it must start.
        rclpy.init(args=['--ros-args', '-p', 'tracking_filter:=particle'])
        try:
            with mock.patch.object(ultralytics_detector, 'YOLO', None), \
                    mock.patch.object(ultralytics_detector, 'YOLO_AVAILABLE', False):
                node = SpatialVisionNode()
            self.assertFalse(node.has_parameter('tracking_filter'))
            self.assertTrue(node.worker.is_alive())
            node.destroy_node()
            self.assertFalse(node.worker.is_alive())
        finally:
            rclpy.shutdown()


class TestFrameErrorHandling(unittest.TestCase):
    def frame(self, node):
        SpatialVisionNode.process_perception_frame(
            node,
            img_msg=object(),
            stamp=SimpleNamespace(sec=1, nanosec=0),
            sensor_data=np.zeros((4, 4), dtype=np.float32),
            is_depth_image=True,
        )

    def test_inference_error_skips_the_frame(self):
        node = fake_node(fx=500.0, cx=320.0)
        node.bridge.imgmsg_to_cv2.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
        node.detector.detect.side_effect = RuntimeError('CUDA out of memory')
        self.frame(node)
        node.get_logger().error.assert_called_once()

    def test_conversion_error_skips_the_frame(self):
        node = fake_node(fx=500.0, cx=320.0)
        node.bridge.imgmsg_to_cv2.side_effect = ValueError('bad encoding')
        self.frame(node)
        node.get_logger().error.assert_called_once()
        node.detector.detect.assert_not_called()


class TestFrameTracking(unittest.TestCase):
    def test_a_detection_is_published_once_its_track_is_confirmed(self):
        node = fake_node(fx=500.0, fy=500.0, cx=50.0, cy=50.0)
        node.bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        node.detector.detect.return_value = [Detection(40.0, 40.0, 60.0, 60.0, 'person', 0.9)]
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        with mock.patch.object(
            geometry, 'transform_point', return_value=Point(x=1.0, y=2.0, z=0.0)
        ):
            for sec in (1, 2):
                SpatialVisionNode.process_perception_frame(
                    node,
                    img_msg=object(),
                    stamp=Time(sec=sec, nanosec=0),
                    sensor_data=depth,
                    is_depth_image=True,
                )

        logged = [call.args[2] for call in node.run_log.write_frame.call_args_list]
        self.assertEqual(logged[0], [])
        self.assertEqual(len(logged[1]), 1)
        self.assertEqual((logged[1][0]['id'], logged[1][0]['class']), (1, 'person'))
        self.assertAlmostEqual(logged[1][0]['x'], 1.0)
        node.marker_pub.publish.assert_called_once()
        node.get_logger().info.assert_not_called()

        detections, counters = node.run_log.write_frame.call_args_list[1].args[4:6]
        self.assertEqual(len(detections), 1)
        self.assertTrue(detections[0]['ranged'])
        # The subscribers and submit_frame do the counting; this test bypasses them
        self.assertEqual(counters, {
            'images_received': 0, 'range_msgs_received': 0, 'synced_frames': 0,
            'dropped_frames': 0, 'tf_failures': 0,
        })

    def test_a_detection_without_a_transform_is_logged_as_not_ranged(self):
        node = fake_node(fx=500.0, fy=500.0, cx=50.0, cy=50.0)
        node.bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        node.detector.detect.return_value = [Detection(40.0, 40.0, 60.0, 60.0, 'person', 0.9)]
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        with mock.patch.object(geometry, 'transform_point', return_value=None):
            SpatialVisionNode.process_perception_frame(
                node,
                img_msg=object(),
                stamp=Time(sec=1, nanosec=0),
                sensor_data=depth,
                is_depth_image=True,
            )

        detections = node.run_log.write_frame.call_args.args[4]
        self.assertEqual(detections, [
            {'class': 'person', 'score': 0.9, 'box': [40.0, 40.0, 60.0, 60.0], 'ranged': False}
        ])


class TestOutputsEveryFrame(unittest.TestCase):
    def run_frame(self, node, detections, sec=1):
        node.bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        node.detector.detect.return_value = detections
        with mock.patch.object(
            geometry, 'transform_point', return_value=Point(x=1.0, y=2.0, z=0.0)
        ):
            SpatialVisionNode.process_perception_frame(
                node,
                img_msg=object(),
                stamp=Time(sec=sec, nanosec=0),
                sensor_data=np.full((100, 100), 2.0, dtype=np.float32),
                is_depth_image=True,
            )

    def test_an_empty_frame_publishes_empty_outputs(self):
        node = fake_node(fx=500.0, fy=500.0, cx=50.0, cy=50.0)
        self.run_frame(node, [])
        cloud = node.costmap_obstacle_pub.publish.call_args.args[0]
        self.assertEqual((cloud.width, len(cloud.data)), (0, 0))
        self.assertEqual(cloud.header.frame_id, 'odom')
        tracks = node.track_pub.publish.call_args.args[0]
        self.assertEqual(tracks.tracks, [])
        self.assertEqual((tracks.header.frame_id, tracks.header.stamp.sec), ('odom', 1))

    @unittest.skipUnless(spatial_vision.VISION_MSGS_AVAILABLE, 'vision_msgs is not installed')
    def test_an_empty_frame_publishes_empty_3d_detections(self):
        node = fake_node(fx=500.0, fy=500.0, cx=50.0, cy=50.0, detection_3d_pub=mock.Mock())
        self.run_frame(node, [])
        detections = node.detection_3d_pub.publish.call_args.args[0]
        self.assertEqual(len(detections.detections), 0)

    def test_a_walker_is_published_as_one_track_with_its_size(self):
        node = fake_node(fx=500.0, fy=500.0, cx=50.0, cy=50.0)
        person = Detection(40.0, 20.0, 60.0, 80.0, 'person', 0.9)
        self.run_frame(node, [person], sec=1)
        self.run_frame(node, [person], sec=2)
        self.assertEqual(node.track_pub.publish.call_count, 2)
        tentative = node.track_pub.publish.call_args_list[0].args[0].tracks
        self.assertEqual([track.state for track in tentative], [TrackMsg.STATE_TENTATIVE])
        (track,) = node.track_pub.publish.call_args.args[0].tracks
        self.assertEqual((track.id, track.state), (1, TrackMsg.STATE_CONFIRMED))
        self.assertAlmostEqual(track.position.x, 1.0)
        # 20 x 60 px at 2 m with fx = fy = 500
        self.assertAlmostEqual(track.size.x, 0.08)
        self.assertAlmostEqual(track.size.z, 0.24)
        logged = node.run_log.write_frame.call_args.args[2]
        np.testing.assert_allclose(logged[0]['size'], [0.08, 0.08, 0.24])

    def test_nothing_is_published_before_the_intrinsics_arrive(self):
        node = fake_node()
        self.run_frame(node, [Detection(40.0, 20.0, 60.0, 80.0, 'person', 0.9)])
        node.track_pub.publish.assert_not_called()
        node.costmap_obstacle_pub.publish.assert_not_called()
        node.marker_pub.publish.assert_not_called()
        node.run_log.write_frame.assert_not_called()


class TestDepthRanging(unittest.TestCase):
    def setUp(self):
        # A person 2 m away, seen as two legs with the 6 m background between them
        self.image = np.full((100, 100), 6.0, dtype=np.float32)
        self.image[:, 38:44] = 2.0
        self.image[:, 57:63] = 2.0

    def test_box_centre_between_the_legs_ranges_the_person(self):
        distance = geometry.extract_depth_image_distance(
            50.0, 50.0, self.image, half_width=10.0, half_height=20.0
        )
        self.assertAlmostEqual(distance, 2.0)

    def test_invalid_depth_is_ignored(self):
        self.image[:, :] = np.inf
        self.assertIsNone(
            geometry.extract_depth_image_distance(50.0, 50.0, self.image)
        )


class TestDebugWindow(unittest.TestCase):
    def test_needs_a_display(self):
        with mock.patch.dict(os.environ, clear=True):
            self.assertFalse(debug_window_supported())

    def test_needs_opencv(self):
        with mock.patch.dict(os.environ, {'DISPLAY': ':0'}), \
                mock.patch.object(spatial_vision, 'cv2', None):
            self.assertFalse(debug_window_supported())

    def test_works_with_both(self):
        with mock.patch.dict(os.environ, {'DISPLAY': ':0'}), \
                mock.patch.object(spatial_vision, 'cv2', object()):
            self.assertTrue(debug_window_supported())


class TestDebugWindowThread(unittest.TestCase):
    """Every OpenCV window call must come from the worker thread (M8)."""

    def setUp(self):
        self.cv2 = mock.Mock()
        self.threads = []
        for name in ('namedWindow', 'imshow', 'waitKey', 'destroyWindow'):
            getattr(self.cv2, name).side_effect = (
                lambda *args: self.threads.append(threading.current_thread())
            )
        patches = (
            mock.patch.object(spatial_vision, 'cv2', self.cv2),
            mock.patch.dict(os.environ, {'DISPLAY': ':0'}),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def window_node(self, **overrides):
        node = fake_node(**{'show_debug_window': True, **overrides})
        node.sync_debug_window = lambda: SpatialVisionNode.sync_debug_window(node)
        return node

    def test_the_window_is_created_once_across_two_frames_and_closed_on_stop(self):
        node = self.window_node(fx=500.0, fy=500.0, cx=50.0, cy=50.0)
        node.bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        node.detector.detect.return_value = []
        node.warm_up_detector = mock.Mock()
        node.process_perception_frame = (
            lambda **frame: SpatialVisionNode.process_perception_frame(node, **frame)
        )
        worker = threading.Thread(target=SpatialVisionNode.frame_worker, args=(node,))
        worker.start()
        try:
            node.frame_slot.put(depth_frame(1))
            self.assertTrue(wait_for(lambda: self.cv2.imshow.call_count == 1))
            node.frame_slot.put(depth_frame(2))
            self.assertTrue(wait_for(lambda: self.cv2.imshow.call_count == 2))
        finally:
            node.frame_slot.close()
            worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.cv2.namedWindow.assert_called_once_with(DEBUG_WINDOW, self.cv2.WINDOW_NORMAL)
        self.cv2.imshow.assert_called_with(DEBUG_WINDOW, node.detector.plot.return_value)
        self.cv2.destroyWindow.assert_called_once_with(DEBUG_WINDOW)
        self.assertFalse(node.debug_window_open)
        self.assertEqual(set(self.threads), {worker})
        node.get_logger().error.assert_not_called()

    def test_turning_the_window_off_closes_it(self):
        node = self.window_node(show_debug_window=False, debug_window_open=True)
        node.sync_debug_window()
        self.cv2.destroyWindow.assert_called_once_with(DEBUG_WINDOW)
        self.assertFalse(node.debug_window_open)
        self.cv2.namedWindow.assert_not_called()

    def test_without_a_display_the_window_is_turned_off_with_a_warning(self):
        node = self.window_node()
        with mock.patch.dict(os.environ, clear=True):
            node.sync_debug_window()
        self.assertFalse(node.show_debug_window)
        self.assertFalse(node.debug_window_open)
        self.cv2.namedWindow.assert_not_called()
        node.get_logger().warn.assert_called_once()

    def test_a_frame_draws_nothing_while_the_window_is_closed(self):
        node = self.window_node(fx=500.0, fy=500.0, cx=50.0, cy=50.0)
        node.bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        node.detector.detect.return_value = []
        SpatialVisionNode.process_perception_frame(node, **depth_frame(1))
        self.assertEqual(self.threads, [])
        node.run_log.write_frame.assert_called_once()


class TestDepthCallback(unittest.TestCase):
    def test_missing_bridge_is_ignored(self):
        node = fake_node(bridge=None)
        SpatialVisionNode.sync_depth_callback(node, object(), object())
        node.submit_frame.assert_not_called()
        node.get_logger().error.assert_not_called()

    def test_bad_depth_image_skips_the_frame(self):
        node = fake_node()
        node.bridge.imgmsg_to_cv2.side_effect = ValueError('bad encoding')
        SpatialVisionNode.sync_depth_callback(node, object(), object())
        node.submit_frame.assert_not_called()
        node.get_logger().error.assert_called_once()

    def depth_frame_from(self, encoding, depth, image_size=(2, 2)):
        """Run the callback on one colour and depth pair, returning the submitted frame."""
        node = fake_node()
        node.bridge.imgmsg_to_cv2.return_value = depth
        image = SimpleNamespace(
            width=image_size[0], height=image_size[1],
            header=SimpleNamespace(stamp='image stamp'),
        )
        depth_msg = SimpleNamespace(
            encoding=encoding, header=SimpleNamespace(stamp='depth stamp', frame_id='depth')
        )
        SpatialVisionNode.sync_depth_callback(node, image, depth_msg)
        return node, image

    def test_good_depth_image_is_submitted_in_meters_with_the_image_stamp(self):
        node, image = self.depth_frame_from('16UC1', np.full((2, 2), 1500, dtype=np.uint16))
        frame = node.submit_frame.call_args.kwargs
        self.assertEqual(
            (frame['img_msg'], frame['stamp'], frame['depth_frame_id']),
            (image, 'image stamp', 'depth'),
        )
        self.assertEqual(frame['range_stamp'], 'depth stamp')
        self.assertTrue(frame['is_depth_image'])
        np.testing.assert_allclose(frame['sensor_data'], 1.5)

    def test_mono16_depth_is_in_millimetres_too(self):
        node, _ = self.depth_frame_from('mono16', np.full((2, 2), 2500, dtype=np.uint16))
        np.testing.assert_allclose(node.submit_frame.call_args.kwargs['sensor_data'], 2.5)

    def test_float_depth_is_already_in_meters(self):
        node, _ = self.depth_frame_from('32FC1', np.full((2, 2), 2.5, dtype=np.float32))
        np.testing.assert_allclose(node.submit_frame.call_args.kwargs['sensor_data'], 2.5)

    def test_a_depth_image_of_another_size_skips_the_frame(self):
        node, _ = self.depth_frame_from(
            '32FC1', np.full((2, 3), 2.0, dtype=np.float32), image_size=(2, 2)
        )
        node.submit_frame.assert_not_called()
        node.get_logger().error.assert_called_once()


class TestLidarCallback(unittest.TestCase):
    def test_the_frame_is_stamped_with_the_image_not_the_scan(self):
        node = fake_node()
        image = SimpleNamespace(header=SimpleNamespace(stamp='image stamp'))
        scan = SimpleNamespace(header=SimpleNamespace(stamp='scan stamp'))
        SpatialVisionNode.sync_lidar_callback(node, image, scan)
        frame = node.submit_frame.call_args.kwargs
        self.assertEqual((frame['img_msg'], frame['stamp']), (image, 'image stamp'))
        self.assertIs(frame['sensor_data'], scan)
        self.assertFalse(frame['is_depth_image'])
        self.assertEqual(frame['range_stamp'], 'scan stamp')


class TestLidarRanging(unittest.TestCase):
    """The lidar branch of process_perception_frame, with the robot's real extrinsics."""

    # laser_frame to camera_optical_link, from the sample bag's /tf_static
    LASER_TO_OPTICAL = Transform(
        translation=Vector3(x=0.0, y=-0.095, z=-0.205),
        rotation=Quaternion(x=0.5, y=-0.5, z=0.5, w=0.5),
    )

    def setUp(self):
        self.node = fake_node(
            fx=528.43, fy=528.43, cx=320.5, cy=240.5, laser_camera_cache={},
            laser_to_camera=lambda *args: SpatialVisionNode.laser_to_camera(self.node, *args),
        )
        self.node.bridge.imgmsg_to_cv2.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        self.node.detector.detect.return_value = [
            Detection(280.0, 0.0, 360.0, 480.0, 'person', 0.9)
        ]
        self.node.tf_buffer.lookup_transform.return_value = SimpleNamespace(
            transform=self.LASER_TO_OPTICAL
        )
        # A wall 4 m away, and an object 1.0 m straight ahead of the camera
        self.scan = LaserScan()
        self.scan.header.frame_id = 'laser_frame'
        self.scan.angle_min = -3.14
        self.scan.angle_increment = 0.01749
        self.scan.range_min = 0.3
        self.scan.range_max = 12.0
        ranges = [4.0] * 360
        ranges[176:184] = [1.205] * 8
        self.scan.ranges = ranges
        self.image = Image()
        self.image.header.frame_id = 'camera_optical_link'

    def run_frame(self, sec):
        SpatialVisionNode.process_perception_frame(
            self.node,
            img_msg=self.image,
            stamp=Time(sec=sec),
            sensor_data=self.scan,
            is_depth_image=False,
        )

    def test_the_nearest_beams_in_the_box_are_placed_from_the_camera_frame(self):
        with mock.patch.object(
            geometry, 'transform_point', return_value=Point(x=1.0)
        ) as transform_point:
            self.run_frame(1)
            self.run_frame(2)
        args = transform_point.call_args.args
        local_x, local_z, source_frame, stamp = args[2], args[4], args[5], args[6]
        self.assertLess(abs(local_z - 1.0), 0.05)
        self.assertLess(abs(local_x), 0.05)
        self.assertEqual((source_frame, stamp), ('camera_optical_link', Time(sec=2)))
        # The extrinsics are static, so they are looked up once, from the latest TF
        self.node.tf_buffer.lookup_transform.assert_called_once()
        target, source, when = self.node.tf_buffer.lookup_transform.call_args.args
        self.assertEqual((target, source, when.nanoseconds), (
            'camera_optical_link', 'laser_frame', 0,
        ))
        detections = self.node.run_log.write_frame.call_args.args[4]
        self.assertTrue(detections[0]['ranged'])

    def test_without_the_extrinsics_nothing_is_ranged_and_the_next_frame_retries(self):
        self.node.tf_buffer.lookup_transform.side_effect = tf2_ros.LookupException('no frame')
        with mock.patch.object(geometry, 'transform_point') as transform_point:
            self.run_frame(1)
            self.run_frame(2)
        transform_point.assert_not_called()
        self.assertEqual(self.node.tf_buffer.lookup_transform.call_count, 2)
        self.node.get_logger().warn.assert_called()
        detections = self.node.run_log.write_frame.call_args.args[4]
        self.assertFalse(detections[0]['ranged'])


class TestReceivedCounts(unittest.TestCase):
    def counts_after(self, depth_source, range_msg):
        """Feed a real node two images and one range message that do not pair."""
        rclpy.init(args=['--ros-args', '-p', f'depth_source:={depth_source}'])
        try:
            with mock.patch.object(ultralytics_detector, 'YOLO', None), \
                    mock.patch.object(ultralytics_detector, 'YOLO_AVAILABLE', False):
                node = SpatialVisionNode()
            range_sub = node.sub_depth if depth_source == 'depth_image' else node.sub_scan
            for sec in (1, 2):
                image = Image()
                image.header.stamp = Time(sec=sec)
                node.sub_img.signalMessage(image)
            range_msg.header.stamp = Time(sec=10)
            range_sub.signalMessage(range_msg)
            counts = (dict(node.received), node.frame_counter)
            node.destroy_node()
            return counts
        finally:
            rclpy.shutdown()

    def test_depth_mode_counts_each_topic_before_pairing(self):
        self.assertEqual(
            self.counts_after('depth_image', Image()),
            ({'images_received': 2, 'range_msgs_received': 1}, 0),
        )

    def test_lidar_mode_counts_each_topic_before_pairing(self):
        self.assertEqual(
            self.counts_after('lidar', LaserScan()),
            ({'images_received': 2, 'range_msgs_received': 1}, 0),
        )


def depth_frame(sec):
    return dict(
        img_msg=object(),
        stamp=Time(sec=sec, nanosec=0),
        sensor_data=np.full((100, 100), 2.0, dtype=np.float32),
        is_depth_image=True,
    )


def wait_for(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.005)
    return condition()


class TestSubmitFrame(unittest.TestCase):
    def test_every_nth_frame_reaches_the_slot(self):
        node = fake_node(process_every_n=2)
        frames = [depth_frame(sec) for sec in range(1, 5)]
        for frame in frames:
            SpatialVisionNode.submit_frame(node, **frame)
        # Frames 2 and 4 pass the every-n check, and 4 replaces 2 before the worker takes it
        self.assertEqual(node.frame_counter, 4)
        self.assertEqual(node.dropped_frames, 1)
        self.assertIs(node.frame_slot.take(timeout=0.1)['stamp'], frames[3]['stamp'])
        self.assertIsNone(node.frame_slot.take(timeout=0.0))


class TestFrameWorker(unittest.TestCase):
    def test_warm_up_runs_the_detector_once_on_a_black_frame(self):
        node = fake_node()
        SpatialVisionNode.warm_up_detector(node)
        image = node.detector.detect.call_args.args[0]
        self.assertEqual((image.shape, image.dtype, image.max()), ((480, 640, 3), np.uint8, 0))

    def test_a_failed_warm_up_is_logged_and_does_not_raise(self):
        node = fake_node()
        node.detector.detect.side_effect = RuntimeError('no model')
        SpatialVisionNode.warm_up_detector(node)
        node.get_logger().warn.assert_called_once()

    def test_worker_survives_a_failed_frame_and_stops_on_close(self):
        processed = []

        def process(**frame):
            processed.append(frame['n'])
            if frame['n'] == 1:
                raise RuntimeError('publisher gone')

        node = fake_node(warm_up_detector=mock.Mock(), process_perception_frame=process)
        worker = threading.Thread(target=SpatialVisionNode.frame_worker, args=(node,))
        worker.start()
        try:
            node.frame_slot.put({'n': 1})
            self.assertTrue(wait_for(lambda: processed == [1]))
            node.frame_slot.put({'n': 2})
            self.assertTrue(wait_for(lambda: processed == [1, 2]))
        finally:
            node.frame_slot.close()
            worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        node.warm_up_detector.assert_called_once()
        node.get_logger().error.assert_called_once()


class TestSlowDetector(unittest.TestCase):
    def test_callbacks_and_diagnostics_do_not_wait_for_the_detector(self):
        started, release = threading.Event(), threading.Event()

        def slow_detect(image):
            started.set()
            release.wait(5.0)
            return []

        node = fake_node(fx=500.0, fy=500.0, cx=50.0, cy=50.0)
        node.bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        node.detector.detect.side_effect = slow_detect
        worker = threading.Thread(
            target=SpatialVisionNode.process_perception_frame, args=(node,),
            kwargs=depth_frame(1),
        )
        worker.start()
        try:
            self.assertTrue(started.wait(2.0))
            # Blocking would take the detector's full 5 s; the diagnostics timer runs at 1 Hz
            for call in (
                lambda: SpatialVisionNode.submit_frame(node, **depth_frame(2)),
                lambda: SpatialVisionNode.publish_diagnostics(node),
            ):
                start = time.monotonic()
                call()
                self.assertLess(time.monotonic() - start, 0.5)
            self.assertTrue(worker.is_alive())
        finally:
            release.set()
            worker.join(timeout=5.0)
        node.diag_pub.publish.assert_called_once()
        self.assertIsNotNone(node.frame_slot.take(timeout=0.0))
        node.run_log.write_frame.assert_called_once()


class TestDiagnostics(unittest.TestCase):
    OLD_KEYS = (
        'fps', 'latency_ms', 'active_tracks', 'world_model_objects', 'depth_source',
        'speed_threshold', 'sync_delta_ms',
    )

    def diagnostics(self, node, now):
        with mock.patch.object(spatial_vision.time, 'monotonic', return_value=now):
            SpatialVisionNode.publish_diagnostics(node)
        status = node.diag_pub.publish.call_args.args[0].status[0]
        return status, {value.key: value.value for value in status.values}

    def node_with_frames(self, times):
        node = fake_node()
        node.cycle_times = [0.03]
        node.last_inference_latency = 30.0
        node.finished_frames = deque(times)
        node.first_finished_frame = times[0]
        return node

    def test_throughput_counts_frames_and_falls_to_zero(self):
        node = self.node_with_frames([100.0 + 0.2 * index for index in range(50)])
        status, values = self.diagnostics(node, 110.0)
        self.assertEqual(values['throughput_hz'], '5.00')
        self.assertEqual(status.level, spatial_vision.DiagnosticStatus.OK)
        status, values = self.diagnostics(node, 115.1)
        self.assertEqual(values['throughput_hz'], '0.00')
        self.assertEqual(status.level, spatial_vision.DiagnosticStatus.WARN)

    def test_a_low_rate_warns_once_the_window_is_full(self):
        node = self.node_with_frames([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        status, values = self.diagnostics(node, 105.5)
        self.assertEqual(values['throughput_hz'], '1.00')
        self.assertEqual(status.level, spatial_vision.DiagnosticStatus.WARN)

        node = self.node_with_frames([100.0, 101.0])
        status, _ = self.diagnostics(node, 101.5)
        self.assertEqual(status.level, spatial_vision.DiagnosticStatus.OK)

    def test_the_range_gap_is_measured_between_the_stamps(self):
        node = fake_node(fx=500.0, fy=500.0, cx=50.0, cy=50.0)
        node.bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        node.detector.detect.return_value = []
        SpatialVisionNode.process_perception_frame(
            node, range_stamp=Time(sec=0, nanosec=970_000_000), **depth_frame(1)
        )
        self.assertEqual(len(node.finished_frames), 1)
        _, values = self.diagnostics(node, node.finished_frames[0])
        self.assertEqual(values['rgb_range_gap_ms'], '30.00')

    def test_tracks_are_counted_by_state(self):
        node = fake_node()
        for step in range(10):
            t = step * 0.2
            seen = [Measurement(0.0, t, 0.0, 'person', 0.9)]
            if step < 2:
                seen.append(Measurement(5.0, 5.0, 0.0, 'chair', 0.6))
            if step == 9:
                seen.append(Measurement(-5.0, -5.0, 0.0, 'person', 0.6))
            node.tracker.step(t, seen)
        _, values = self.diagnostics(node, 0.0)
        self.assertEqual(
            [values[key] for key in (
                'active_tracks', 'tentative_tracks', 'confirmed_tracks', 'coasting_tracks',
                'dynamic_tracks',
            )],
            ['3', '1', '1', '1', '1'],
        )

    def test_old_keys_stay_and_counters_are_reported(self):
        node = fake_node(dropped_frames=4, tf_counts={'tf_failures': 2})
        node.received.update(images_received=10, range_msgs_received=9)
        _, values = self.diagnostics(node, 0.0)
        for key in self.OLD_KEYS:
            self.assertIn(key, values)
        self.assertEqual(
            [values[key] for key in (
                'dropped_frames', 'tf_failures', 'images_received', 'range_msgs_received',
            )],
            ['4', '2', '10', '9'],
        )


if __name__ == '__main__':
    unittest.main()
