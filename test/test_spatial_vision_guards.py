"""Regression tests for the failure modes that used to kill or stall spatial_vision."""

import importlib.util
import os
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
import numpy as np
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo

from r_one import spatial_vision
from r_one.spatial_vision import (
    debug_window_supported,
    intrinsics_are_valid,
    SpatialVisionNode,
    validate_parameter,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def camera_info(k):
    msg = CameraInfo()
    msg.k = [float(value) for value in k]
    return msg


def fake_node(**overrides):
    """Build the attributes that the node methods under test read from self."""
    logger = mock.Mock()
    attrs = dict(
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        bridge=mock.Mock(),
        model=mock.Mock(),
        state_lock=threading.RLock(),
        frame_counter=0,
        process_every_n=1,
        img_size=320,
        speed_threshold=0.15,
        sync_slop=0.08,
        ema_alpha=0.35,
        tracking_filter='ema',
        show_debug_window=False,
        time_sync=None,
        get_logger=lambda: logger,
        process_perception_frame=mock.Mock(),
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def load_launch(name):
    path = os.path.join(REPO_ROOT, 'launch', name)
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def walk_entities(entities):
    """Yield launch actions, descending into group and timer children."""
    for entity in entities:
        yield entity
        children = getattr(entity, '_GroupAction__actions', None) \
            or getattr(entity, '_TimerAction__actions', None)
        if children:
            yield from walk_entities(children)


def resolve_arguments(description, **given):
    """Evaluate declared launch arguments in order, as ros2 launch does."""
    context = LaunchContext()
    context.launch_configurations.update(given)
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.visit(context)
    return context


def process_commands(description, context):
    """Return the command line of every process whose condition holds."""
    commands = []
    for entity in walk_entities(description.entities):
        if not isinstance(entity, ExecuteProcess) or isinstance(entity, Node):
            continue
        condition = entity.condition
        if condition is not None and not condition.evaluate(context):
            continue
        commands.append([
            ''.join(part.perform(context) for part in token) for token in entity.cmd
        ])
    return commands


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
            ('ema_alpha', 0.0),
            ('ema_alpha', 1.5),
            ('tracking_filter', 'particle'),
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
            ('ema_alpha', 1.0),
            ('tracking_filter', 'KALMAN'),
            ('depth_source', 'depth_image'),
            ('target_frame', 'anything'),
        ):
            with self.subTest(name=name, value=value):
                self.assertIsNone(validate_parameter(name, value))

    def test_rejected_multi_parameter_set_applies_nothing(self):
        node = fake_node()
        result = SpatialVisionNode.parameters_callback(node, [
            Parameter('speed_threshold', value=0.9),
            Parameter('ema_alpha', value=2.0),
        ])
        self.assertFalse(result.successful)
        self.assertEqual(node.speed_threshold, 0.15)

    def test_valid_multi_parameter_set_applies_everything(self):
        node = fake_node()
        result = SpatialVisionNode.parameters_callback(node, [
            Parameter('speed_threshold', value=0.9),
            Parameter('ema_alpha', value=0.5),
        ])
        self.assertTrue(result.successful)
        self.assertEqual((node.speed_threshold, node.ema_alpha), (0.9, 0.5))


class TestNodeStartup(unittest.TestCase):
    def assert_startup_rejected(self, ros_argument):
        rclpy.init(args=['--ros-args', '-p', ros_argument])
        try:
            with mock.patch.object(spatial_vision, 'YOLO', None), \
                    mock.patch.object(spatial_vision, 'YOLO_AVAILABLE', False):
                with self.assertRaises(ValueError):
                    SpatialVisionNode()
        finally:
            rclpy.shutdown()

    def test_zero_frame_interval_is_rejected(self):
        self.assert_startup_rejected('process_every_n_frames:=0')

    def test_unknown_depth_source_is_rejected(self):
        self.assert_startup_rejected('depth_source:=sonar')

    def test_unknown_tracking_filter_is_rejected(self):
        self.assert_startup_rejected('tracking_filter:=particle')


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
        node.model.track.side_effect = RuntimeError('CUDA out of memory')
        self.frame(node)
        node.get_logger().error.assert_called_once()

    def test_conversion_error_skips_the_frame(self):
        node = fake_node(fx=500.0, cx=320.0)
        node.bridge.imgmsg_to_cv2.side_effect = ValueError('bad encoding')
        self.frame(node)
        node.get_logger().error.assert_called_once()
        node.model.track.assert_not_called()


class TestDepthRanging(unittest.TestCase):
    def setUp(self):
        # A person 2 m away, seen as two legs with the 6 m background between them
        self.image = np.full((100, 100), 6.0, dtype=np.float32)
        self.image[:, 38:44] = 2.0
        self.image[:, 57:63] = 2.0

    def test_box_centre_between_the_legs_ranges_the_person(self):
        distance = SpatialVisionNode.extract_depth_image_distance(
            None, 50.0, 50.0, self.image, half_width=10.0, half_height=20.0
        )
        self.assertAlmostEqual(distance, 2.0)

    def test_invalid_depth_is_ignored(self):
        self.image[:, :] = np.inf
        self.assertIsNone(
            SpatialVisionNode.extract_depth_image_distance(None, 50.0, 50.0, self.image)
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


class TestDepthCallback(unittest.TestCase):
    def test_missing_bridge_is_ignored(self):
        node = fake_node(bridge=None)
        SpatialVisionNode.sync_depth_callback(node, object(), object())
        node.process_perception_frame.assert_not_called()
        node.get_logger().error.assert_not_called()

    def test_bad_depth_image_skips_the_frame(self):
        node = fake_node()
        node.bridge.imgmsg_to_cv2.side_effect = ValueError('bad encoding')
        SpatialVisionNode.sync_depth_callback(node, object(), object())
        node.process_perception_frame.assert_not_called()
        node.get_logger().error.assert_called_once()


class TestSceneLaunch(unittest.TestCase):
    def setUp(self):
        self.description = load_launch('spatial_scene.launch.py')

    def test_simulation_uses_depth_camera_and_sim_clock(self):
        # The CPU lidar cannot see Gazebo actors, so the demo must not range with it.
        context = resolve_arguments(self.description)
        self.assertEqual(context.launch_configurations['depth_source'], 'depth_image')
        self.assertEqual(context.launch_configurations['use_sim_time'], 'true')

    def test_hardware_uses_lidar_and_wall_clock(self):
        context = resolve_arguments(self.description, sim='false')
        self.assertEqual(context.launch_configurations['depth_source'], 'lidar')
        self.assertEqual(context.launch_configurations['use_sim_time'], 'false')

    def test_robot_tf_runs_without_the_simulator(self):
        context = resolve_arguments(self.description, sim='false')
        included = [
            entity for entity in self.description.entities
            if type(entity).__name__ == 'IncludeLaunchDescription'
        ]
        self.assertEqual(len(included), 1)
        self.assertTrue(included[0].condition.evaluate(context))

    def test_gui_argument_selects_the_gazebo_binary(self):
        executables = [
            os.path.basename(command[0])
            for command in process_commands(
                self.description, resolve_arguments(self.description, gui='false')
            )
        ]
        self.assertEqual(executables, ['gzserver'])
        executables = [
            os.path.basename(command[0])
            for command in process_commands(
                self.description, resolve_arguments(self.description)
            )
        ]
        self.assertEqual(executables, ['gazebo'])

    def test_tracker_settings_are_exposed(self):
        context = resolve_arguments(self.description)
        self.assertEqual(context.launch_configurations['tracking_filter'], 'ema')
        self.assertEqual(context.launch_configurations['ema_alpha'], '0.35')


class TestPerceptionLaunch(unittest.TestCase):
    def setUp(self):
        self.description = load_launch('spatial_perception.launch.py')

    def rviz(self):
        return next(
            entity for entity in self.description.entities
            if getattr(entity, 'node_executable', None) == 'rviz2'
        )

    def test_debug_window_is_off_by_default(self):
        context = resolve_arguments(self.description)
        self.assertEqual(context.launch_configurations['show_debug_window'], 'false')

    def test_open_rviz_controls_rviz(self):
        rviz = self.rviz()
        self.assertTrue(rviz.condition.evaluate(resolve_arguments(self.description)))
        self.assertFalse(
            rviz.condition.evaluate(resolve_arguments(self.description, open_rviz='false'))
        )


class TestBagLaunch(unittest.TestCase):
    def setUp(self):
        self.description = load_launch('spatial_bag_playback.launch.py')

    def player_command(self, **arguments):
        context = resolve_arguments(self.description, **arguments)
        # The player is delayed so the detector is ready before the bag starts.
        return [
            command for command in process_commands(self.description, context)
            if command[:3] == ['ros2', 'bag', 'play']
        ]

    def test_loop_is_on_by_default(self):
        self.assertIn('--loop', self.player_command()[0])

    def test_loop_false_is_honoured(self):
        self.assertNotIn('--loop', self.player_command(loop='false')[0])

    def test_default_bag_path_matches_the_recorder(self):
        # ros2 bag record writes a directory, so the default must name that directory.
        context = resolve_arguments(self.description)
        path = context.launch_configurations['bag_path']
        self.assertTrue(path.endswith(os.path.join('sample_bags', 'sample_spatial_run')), path)


if __name__ == '__main__':
    unittest.main()
