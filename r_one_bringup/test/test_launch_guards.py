"""Regression tests for launch arguments and launch-time defaults."""

import importlib.util
import os
import unittest

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_launch(name):
    path = os.path.join(PACKAGE_ROOT, 'launch', name)
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

    def test_run_log_is_off_by_default(self):
        context = resolve_arguments(self.description)
        self.assertEqual(context.launch_configurations['run_log_path'], '')


if __name__ == '__main__':
    unittest.main()
