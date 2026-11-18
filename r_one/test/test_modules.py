"""Unit tests for the modules split out of spatial_vision, run without starting ROS."""

import dataclasses
import math
import struct
from types import SimpleNamespace
import unittest
from unittest import mock

from geometry_msgs.msg import Point, Quaternion, Transform, Vector3
import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
import tf2_ros
from visualization_msgs.msg import MarkerArray

from r_one import geometry, messages
from r_one.detectors import Detection, ultralytics_detector
from r_one.tracking import Measurement, Tracker
from r_one_msgs.msg import Track as TrackMsg

STAMP = Time().to_msg()


def fake_box(xyxy, class_id, score, track_id=None):
    return SimpleNamespace(
        xyxy=[xyxy],
        cls=[class_id],
        conf=[score],
        id=None if track_id is None else [track_id],
    )


def fake_yolo(boxes):
    """Stand in for ultralytics.YOLO: one model whose predict() returns these boxes."""
    result = mock.Mock(boxes=boxes)
    model = mock.Mock(names={0: 'person', 56: 'chair'})
    model.predict.return_value = [result]
    return mock.Mock(return_value=model)


def scan(ranges, angle_min=-0.5, angle_increment=0.1):
    msg = LaserScan()
    msg.angle_min = angle_min
    msg.angle_increment = angle_increment
    msg.angle_max = angle_min + angle_increment * (len(ranges) - 1)
    msg.range_min = 0.1
    msg.range_max = 10.0
    msg.ranges = [float(value) for value in ranges]
    return msg


class TestDetection(unittest.TestCase):
    def test_track_id_defaults_to_untracked(self):
        detection = Detection(1.0, 2.0, 3.0, 4.0, 'person', 0.9)
        self.assertEqual(detection.track_id, -1)

    def test_is_immutable(self):
        detection = Detection(1.0, 2.0, 3.0, 4.0, 'person', 0.9, 5)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            detection.track_id = 6


class TestUltralyticsDetector(unittest.TestCase):
    def make(self, boxes):
        factory = fake_yolo(boxes)
        with mock.patch.object(ultralytics_detector, 'YOLO', factory):
            detector = ultralytics_detector.UltralyticsDetector('weights.pt', 320)
        factory.assert_called_once_with('weights.pt')
        return detector

    def test_boxes_become_detections(self):
        detector = self.make([
            fake_box([10.0, 20.0, 30.0, 60.0], 0, 0.8, track_id=7),
            fake_box([1.0, 2.0, 3.0, 4.0], 56, 0.5),
        ])
        detections = detector.detect('image')
        self.assertEqual(detections, [
            Detection(10.0, 20.0, 30.0, 60.0, 'person', 0.8, 7),
            Detection(1.0, 2.0, 3.0, 4.0, 'chair', 0.5, -1),
        ])

    def test_detect_runs_plain_detection_on_one_model(self):
        detector = self.make([])
        detector.detect('first')
        detector.detect('second')
        self.assertEqual(detector.model.predict.call_count, 2)
        detector.model.predict.assert_called_with('second', imgsz=320, verbose=False)
        detector.model.track.assert_not_called()

    def test_plot_draws_the_last_frame(self):
        detector = self.make([])
        self.assertIsNone(detector.plot())
        detector.detect('image')
        self.assertIs(detector.plot(), detector.model.predict.return_value[0].plot.return_value)


# The robot's laser_frame to camera_optical_link transform, as the sample bag's /tf_static
# gives it: the lidar sits 0.205 m behind the camera and 0.095 m above it.
LASER_TO_OPTICAL = Transform(
    translation=Vector3(x=0.0, y=-0.095, z=-0.205),
    rotation=Quaternion(x=0.5, y=-0.5, z=0.5, w=0.5),
)
# The sample bag's camera intrinsics and scan layout
FX = FY = 528.43
CX, CY = 320.5, 240.5


def synthetic_scan(circles, background=4.0):
    """Return a 360-beam scan, in the laser frame, of circles (x, y, radius) before a wall."""
    msg = LaserScan()
    msg.angle_min = -3.14
    msg.angle_increment = 0.01749
    msg.angle_max = msg.angle_min + msg.angle_increment * 359
    msg.range_min = 0.3
    msg.range_max = 12.0
    ranges = []
    for i in range(360):
        angle = msg.angle_min + i * msg.angle_increment
        dx, dy = math.cos(angle), math.sin(angle)
        best = background
        for x, y, radius in circles:
            along = dx * x + dy * y
            disc = along * along - (x * x + y * y) + radius * radius
            if disc >= 0.0 and along - math.sqrt(disc) > 0.0:
                best = min(best, along - math.sqrt(disc))
        ranges.append(best)
    msg.ranges = ranges
    return msg


def camera_points(scan_msg):
    return geometry.transform_points(
        geometry.transform_to_matrix(LASER_TO_OPTICAL), geometry.scan_points(scan_msg)
    )


def centroid_in_box(scan_msg, box):
    return geometry.nearest_cluster(
        geometry.points_in_box(camera_points(scan_msg), box, FX, FY, CX, CY)
    )


class TestScanPoints(unittest.TestCase):
    def test_valid_beams_become_points_in_the_scan_frame(self):
        points = geometry.scan_points(scan([2.0, 3.0], angle_min=0.0, angle_increment=math.pi / 2))
        np.testing.assert_allclose(points, [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], atol=1e-9)

    def test_invalid_beams_are_dropped(self):
        ranges = [float('inf'), float('nan'), 0.05, 10.0, 11.0, 2.0]
        points = geometry.scan_points(scan(ranges, angle_min=0.0, angle_increment=0.0))
        np.testing.assert_allclose(points, [[2.0, 0.0, 0.0]])


class TestTransformToMatrix(unittest.TestCase):
    def test_the_robots_extrinsics_put_the_lidar_behind_and_above_the_camera(self):
        matrix = geometry.transform_to_matrix(LASER_TO_OPTICAL)
        # Laser forward, left and up become optical +z, -x and -y
        np.testing.assert_allclose(
            geometry.transform_points(matrix, np.array([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            ])),
            [
                [0.0, -0.095, -0.205], [0.0, -0.095, 0.795],
                [-1.0, -0.095, -0.205], [0.0, -1.095, -0.205],
            ],
            atol=1e-9,
        )


class TestPointsInBox(unittest.TestCase):
    def test_keeps_points_that_project_into_the_box(self):
        points = np.array([
            [0.0, 0.0, 2.0],  # The image centre
            [1.0, 0.0, 2.0],  # Far to the right
            [0.0, 0.0, -2.0],  # Behind the camera, though it would project to the centre
        ])
        kept = geometry.points_in_box(points, (300.0, 220.0, 340.0, 260.0), FX, FY, CX, CY)
        np.testing.assert_allclose(kept, [[0.0, 0.0, 2.0]])

    def test_no_points_gives_an_empty_array(self):
        kept = geometry.points_in_box(np.zeros((0, 3)), (0.0, 0.0, 640.0, 480.0), FX, FY, CX, CY)
        self.assertEqual(kept.shape, (0, 3))


class TestMeasuredSize(unittest.TestCase):
    def test_box_and_range_give_width_depth_height(self):
        size = geometry.measured_size((100.0, 50.0, 200.0, 250.0), 2.0, 400.0, 400.0)
        np.testing.assert_allclose(size, (0.5, 0.5, 1.0))

    def test_unknown_size_is_empty(self):
        for box, range_m in (
            ((100.0, 50.0, 200.0, 250.0), 0.0),
            ((100.0, 50.0, 200.0, 250.0), -1.0),
            ((100.0, 50.0, 200.0, 250.0), float('nan')),
            ((100.0, 50.0, 100.0, 250.0), 2.0),
            ((100.0, 250.0, 200.0, 50.0), 2.0),
            ((), 2.0),
        ):
            self.assertEqual(geometry.measured_size(box, range_m, 400.0, 400.0), ())


class TestNearestCluster(unittest.TestCase):
    def test_legs_with_a_gap_range_the_person_not_the_wall(self):
        # Two legs 1.0 m in front of the camera and 0.2 m apart, a wall at 4 m behind and
        # between them. A median of the beams at the box centre returns 4 m here.
        legs = synthetic_scan([(1.205, 0.1, 0.06), (1.205, -0.1, 0.06)])
        centroid = centroid_in_box(legs, (230.0, 0.0, 410.0, 480.0))
        self.assertLess(abs(centroid[2] - 1.0), 0.1)
        self.assertLess(abs(centroid[0]), 0.05)

    def test_an_object_off_to_the_side_gets_its_own_beams(self):
        # A post 1.5 m ahead of the camera and 0.6 m to its left. The lidar sees it at a
        # smaller bearing than the camera does, because the lidar sits 0.205 m further back.
        post = synthetic_scan([(1.705, 0.6, 0.05)])
        u = FX * -0.6 / 1.5 + CX
        centroid = centroid_in_box(post, (u - 25.0, 0.0, u + 25.0, 480.0))
        self.assertLess(abs(centroid[2] - 1.5), 0.1)
        self.assertLess(abs(centroid[0] + 0.6), 0.05)

    def test_an_empty_box_has_no_range(self):
        self.assertIsNone(geometry.nearest_cluster(np.zeros((0, 3))))

    def test_a_box_above_the_scan_plane_has_no_range(self):
        # The scan plane projects just above the image centre; this box is near the top
        self.assertIsNone(centroid_in_box(synthetic_scan([]), (0.0, 0.0, 640.0, 100.0)))

    def test_a_single_stray_beam_has_no_range(self):
        self.assertIsNone(geometry.nearest_cluster(np.array([[0.0, 0.0, 2.0]])))

    def test_a_stray_beam_in_front_is_skipped(self):
        points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.05, 0.0, 3.05]])
        np.testing.assert_allclose(geometry.nearest_cluster(points), [0.025, 0.0, 3.025])


class TestTransformPoint(unittest.TestCase):
    def test_looks_up_the_transform_at_the_stamp(self):
        calls = []

        def transform(point_stamped, target_frame, timeout):
            calls.append((point_stamped.header.stamp.sec, target_frame, timeout))
            return SimpleNamespace(point=Point(x=point_stamped.point.x + 1.0))

        counts = {'tf_failures': 0}
        point = geometry.transform_point(
            mock.Mock(transform=transform), 'odom', 2.0, 0.0, 0.0, 'laser_frame',
            Time(seconds=5).to_msg(), mock.Mock(), counts,
        )
        self.assertEqual(calls, [(5, 'odom', Duration(seconds=0.1))])
        self.assertAlmostEqual(point.x, 3.0)
        self.assertEqual(counts, {'tf_failures': 0})

    def test_a_failed_lookup_gives_none_without_trying_the_latest_transform(self):
        buffer = mock.Mock()
        buffer.transform.side_effect = tf2_ros.ExtrapolationException('too old')
        logger = mock.Mock()
        counts = {'tf_failures': 0}
        point = geometry.transform_point(
            buffer, 'odom', 2.0, 0.0, 0.0, 'laser_frame', Time(seconds=5).to_msg(), logger,
            counts,
        )
        self.assertIsNone(point)
        buffer.transform.assert_called_once()
        logger.debug.assert_called_once()
        self.assertEqual(counts, {'tf_failures': 1})

    def test_returns_none_when_no_transform_exists(self):
        buffer = mock.Mock()
        buffer.transform.side_effect = tf2_ros.LookupException('no frame')
        logger = mock.Mock()
        counts = {'tf_failures': 0}
        point = geometry.transform_point(
            buffer, 'odom', 2.0, 0.0, 0.0, 'laser_frame', Time().to_msg(), logger, counts
        )
        self.assertIsNone(point)
        logger.debug.assert_called_once()
        self.assertEqual(counts, {'tf_failures': 1})


class TestMessages(unittest.TestCase):
    def test_point_cloud_packs_every_point(self):
        cloud = messages.build_pointcloud2([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], 'odom', STAMP)
        self.assertEqual(cloud.header.frame_id, 'odom')
        self.assertEqual(cloud.width, 2)
        self.assertEqual(struct.unpack('6f', bytes(cloud.data)), (1, 2, 3, 4, 5, 6))

    def test_static_object_gets_a_box_and_a_label(self):
        markers = MarkerArray()
        messages.append_markers(
            markers, 'odom', 3, 3, 'person', 'STATIC', Point(), 0.0, 0.0, 0.0, STAMP
        )
        self.assertEqual(
            [marker.ns for marker in markers.markers], ['detected_objects', 'object_labels']
        )
        self.assertTrue(all(marker.header.frame_id == 'odom' for marker in markers.markers))

    def test_unknown_size_keeps_the_default_box(self):
        markers = MarkerArray()
        messages.append_markers(
            markers, 'odom', 3, 3, 'person', 'STATIC', Point(z=1.0), 0.0, 0.0, 0.0, STAMP
        )
        cube, text = markers.markers
        self.assertEqual((cube.scale.x, cube.scale.y, cube.scale.z), (0.45, 0.45, 0.50))
        self.assertAlmostEqual(cube.pose.position.z, 1.25)
        self.assertAlmostEqual(text.pose.position.z, 1.65)

    def test_measured_size_is_centred_on_the_point(self):
        markers = MarkerArray()
        messages.append_markers(
            markers, 'odom', 3, 3, 'person', 'STATIC', Point(z=1.0), 0.0, 0.0, 0.0, STAMP,
            size=(0.5, 0.5, 1.7),
        )
        cube, text = markers.markers
        np.testing.assert_allclose((cube.scale.x, cube.scale.y, cube.scale.z), (0.5, 0.5, 1.7))
        self.assertAlmostEqual(cube.pose.position.z, 1.0)
        self.assertAlmostEqual(text.pose.position.z, 1.0 + 0.85 + 0.15)

    def test_moving_object_also_gets_its_predicted_path(self):
        markers = MarkerArray()
        messages.append_markers(
            markers, 'odom', 3, 3, 'person', 'DYNAMIC', Point(), 0.75, 0.0, 0.75, STAMP
        )
        self.assertEqual(len(markers.markers), 4)
        self.assertAlmostEqual(markers.markers[3].points[-1].y, 1.5)

    def test_world_model_gets_one_marker_per_object(self):
        markers = MarkerArray()
        world_model = {'person_2': (4.5, -1.9, 0.0), 'chair_5': (1.0, 1.0, 0.0)}
        messages.append_world_model_markers(markers, 'odom', world_model, STAMP)
        self.assertEqual(len(markers.markers), 2)
        self.assertAlmostEqual(markers.markers[0].pose.position.x, 4.5)

    @unittest.skipUnless(messages.VISION_MSGS_AVAILABLE, 'vision_msgs is not installed')
    def test_detection_3d_carries_the_track(self):
        array = messages.Detection3DArray()
        messages.append_detection_3d(array, 'odom', 7, 'person', 0.8, Point(x=1.0), STAMP)
        detection = array.detections[0]
        self.assertEqual((detection.id, detection.header.frame_id), ('7', 'odom'))
        self.assertEqual(detection.results[0].hypothesis.class_id, 'person')
        self.assertAlmostEqual(detection.bbox.center.position.x, 1.0)
        self.assertAlmostEqual(detection.bbox.center.position.z, 0.25)
        self.assertAlmostEqual(detection.bbox.size.z, 0.50)

    @unittest.skipUnless(messages.VISION_MSGS_AVAILABLE, 'vision_msgs is not installed')
    def test_detection_3d_uses_the_measured_size(self):
        array = messages.Detection3DArray()
        messages.append_detection_3d(
            array, 'odom', 7, 'person', 0.8, Point(x=1.0, z=0.9), STAMP, size=(0.5, 0.5, 1.7)
        )
        bbox = array.detections[0].bbox
        self.assertAlmostEqual(bbox.center.position.z, 0.9)
        np.testing.assert_allclose((bbox.size.x, bbox.size.y, bbox.size.z), (0.5, 0.5, 1.7))


class TestTrackArray(unittest.TestCase):
    def two_track_scene(self):
        """Track a walker matched every frame and a chair last seen at 0.2 s."""
        tracker = Tracker()
        for step in range(10):
            t = step * 0.2
            seen = [Measurement(0.0, t, 0.3, 'person', 0.9, size=(0.5, 0.5, 1.7))]
            if step < 2:
                seen.append(Measurement(5.0, 5.0, 0.0, 'chair', 0.6))
            tracker.step(t, seen)
        return tracker

    def test_every_field_is_filled(self):
        tracker = self.two_track_scene()
        array = messages.build_track_array(tracker, tracker.tracks, 'odom', STAMP, 1.8)
        self.assertEqual(array.header.frame_id, 'odom')
        walker, chair = array.tracks

        self.assertEqual((walker.id, walker.class_name), (1, 'person'))
        self.assertAlmostEqual(walker.score, 0.9)
        self.assertEqual(walker.state, TrackMsg.STATE_CONFIRMED)
        self.assertEqual(walker.motion, TrackMsg.MOTION_DYNAMIC)
        self.assertAlmostEqual(walker.position.y, 1.8, delta=0.1)
        self.assertAlmostEqual(walker.position.z, 0.3)
        self.assertAlmostEqual(walker.velocity.y, 1.0, delta=0.15)
        self.assertEqual((walker.size.x, walker.size.y, walker.size.z), (0.5, 0.5, 1.7))
        self.assertAlmostEqual(walker.age_s, 1.8)
        self.assertAlmostEqual(walker.time_since_update_s, 0.0)

        self.assertEqual((chair.id, chair.class_name), (2, 'chair'))
        self.assertEqual(chair.state, TrackMsg.STATE_COASTING)
        self.assertEqual(chair.motion, TrackMsg.MOTION_UNKNOWN)
        self.assertEqual((chair.size.x, chair.size.y, chair.size.z), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(chair.age_s, 1.8)
        self.assertAlmostEqual(chair.time_since_update_s, 1.6)

    def test_covariance_is_row_major(self):
        tracker = self.two_track_scene()
        # A real covariance is symmetric, which would hide a transposed copy
        tracker.tracks[0].filter.P = np.arange(16.0).reshape(4, 4)
        array = messages.build_track_array(tracker, tracker.tracks, 'odom', STAMP, 1.8)
        covariance = list(array.tracks[0].covariance)
        self.assertEqual(len(covariance), 16)
        self.assertEqual(covariance[2], 2.0)  # P[0, 2]
        self.assertEqual(covariance[8], 8.0)  # P[2, 0]
        self.assertEqual(covariance, [float(value) for value in range(16)])

    def test_tentative_track_is_reported(self):
        tracker = Tracker()
        tracker.step(0.0, [Measurement(1.0, 1.0, 0.0, 'person', 0.9)])
        array = messages.build_track_array(tracker, tracker.tracks, 'odom', STAMP, 0.0)
        self.assertEqual([track.state for track in array.tracks], [TrackMsg.STATE_TENTATIVE])

    def test_no_tracks_gives_an_empty_list_with_a_header(self):
        tracker = Tracker()
        array = messages.build_track_array(tracker, tracker.tracks, 'odom', STAMP, 0.0)
        self.assertEqual(array.tracks, [])
        self.assertEqual(array.header.frame_id, 'odom')
        self.assertEqual(array.header.stamp, STAMP)


if __name__ == '__main__':
    unittest.main()
