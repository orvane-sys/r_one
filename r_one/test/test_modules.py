"""Unit tests for the modules split out of spatial_vision, run without starting ROS."""

import dataclasses
import struct
from types import SimpleNamespace
import unittest
from unittest import mock

from geometry_msgs.msg import Point
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
import tf2_ros
from visualization_msgs.msg import MarkerArray

from r_one import geometry, messages
from r_one.detectors import Detection, ultralytics_detector

STAMP = Time().to_msg()


def fake_box(xyxy, class_id, score, track_id=None):
    return SimpleNamespace(
        xyxy=[xyxy],
        cls=[class_id],
        conf=[score],
        id=None if track_id is None else [track_id],
    )


def fake_yolo(boxes):
    """Stand in for ultralytics.YOLO: one model whose track() returns these boxes."""
    result = mock.Mock(boxes=boxes)
    model = mock.Mock(names={0: 'person', 56: 'chair'})
    model.track.return_value = [result]
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

    def test_tracking_state_persists_on_one_model(self):
        detector = self.make([])
        detector.detect('first')
        detector.detect('second')
        self.assertEqual(detector.model.track.call_count, 2)
        detector.model.track.assert_called_with(
            'second', persist=True, tracker='bytetrack.yaml', imgsz=320, verbose=False
        )

    def test_plot_draws_the_last_frame(self):
        detector = self.make([])
        self.assertIsNone(detector.plot())
        detector.detect('image')
        self.assertIs(detector.plot(), detector.model.track.return_value[0].plot.return_value)


class TestLidarDistance(unittest.TestCase):
    def test_median_of_the_beams_around_the_bearing(self):
        ranges = [5.0] * 11
        ranges[4:7] = [2.0, 2.1, 2.2]
        self.assertAlmostEqual(geometry.extract_lidar_distance(0.0, scan(ranges)), 2.2)

    def test_invalid_beams_are_ignored(self):
        ranges = [float('inf')] * 11
        ranges[5] = 3.0
        self.assertAlmostEqual(geometry.extract_lidar_distance(0.0, scan(ranges)), 3.0)

    def test_bearing_outside_the_scan_has_no_range(self):
        self.assertIsNone(geometry.extract_lidar_distance(1.0, scan([2.0] * 11)))


class TestTransformPoint(unittest.TestCase):
    def test_falls_back_to_the_latest_transform(self):
        stamps = []

        def transform(point_stamped, target_frame, timeout):
            stamps.append(point_stamped.header.stamp.sec)
            if len(stamps) == 1:
                raise tf2_ros.ExtrapolationException('too old')
            return SimpleNamespace(point=Point(x=point_stamped.point.x + 1.0))

        buffer = mock.Mock(transform=transform)
        point = geometry.transform_point(
            buffer, 'odom', 2.0, 0.0, 0.0, 'laser_frame', Time(seconds=5).to_msg(), mock.Mock()
        )
        self.assertEqual(stamps, [5, 0])
        self.assertAlmostEqual(point.x, 3.0)

    def test_returns_none_when_no_transform_exists(self):
        buffer = mock.Mock()
        buffer.transform.side_effect = tf2_ros.LookupException('no frame')
        logger = mock.Mock()
        point = geometry.transform_point(
            buffer, 'odom', 2.0, 0.0, 0.0, 'laser_frame', Time().to_msg(), logger
        )
        self.assertIsNone(point)
        logger.debug.assert_called_once()


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


if __name__ == '__main__':
    unittest.main()
