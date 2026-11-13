#!/usr/bin/env python3

import rclpy
try:
    from rclpy.executors import ExternalShutdownException
except ImportError:
    ExternalShutdownException = KeyboardInterrupt
from rclpy.node import Node
from std_msgs.msg import String


class MockAINode(Node):
    def __init__(self):
        super().__init__('mock_ai_camera')

        self.publisher_ = self.create_publisher(String, 'ai/detections', 10)
        self.timer = self.create_timer(0.2, self.timer_callback)
        self.get_logger().info('Mock AI Started! Broadcasting test detections...')

    def timer_callback(self):
        msg = String()
        msg.data = 'Object: Human, Location: Center'
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MockAINode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
