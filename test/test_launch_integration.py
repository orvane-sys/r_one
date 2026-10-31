import time
import unittest
import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from rclpy.node import Node


@pytest.mark.launch_test
def generate_test_description():
    situational_awareness = launch_ros.actions.Node(
        package='r_one',
        executable='situational_awareness_node',
        name='situational_awareness_test',
        output='screen'
    )
    mock_ai = launch_ros.actions.Node(
        package='r_one',
        executable='mock_ai_node',
        name='mock_ai_test',
        output='screen'
    )

    return launch.LaunchDescription([
        situational_awareness,
        mock_ai,
        launch_testing.actions.ReadyToTest()
    ])


class TestNodeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.test_node = Node('test_integration_listener')

    def tearDown(self):
        self.test_node.destroy_node()

    def test_nodes_active_and_topics_advertised(self):
        """Verifies that nodes start and their output topics appear in the ROS 2 graph."""
        start_time = time.time()
        topics_found = False

        while time.time() - start_time < 5.0:
            topics = dict(self.test_node.get_topic_names_and_types())
            if '/robot_alerts' in topics and '/ai/detections' in topics:
                topics_found = True
                break
            time.sleep(0.1)

        self.assertTrue(
            topics_found,
            "Expected /robot_alerts and /ai/detections to be present in ROS 2 graph"
        )


@launch_testing.post_shutdown_test()
class TestShutdownExitCodes(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        """Verifies all processes exited cleanly (SIGINT/SIGTERM or 0)."""
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15]
        )
