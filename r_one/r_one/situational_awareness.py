#!/usr/bin/env python3

import math
import rclpy
try:
    from rclpy.executors import ExternalShutdownException
except ImportError:
    ExternalShutdownException = KeyboardInterrupt
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class SituationalAwarenessNode(Node):
    def __init__(self):
        super().__init__('situational_awareness_module')

        self.declare_parameter('high_risk_distance', 0.5)
        self.declare_parameter('caution_distance', 1.0)
        self.declare_parameter('robot_radius_ignore', 0.30)

        self.high_risk_distance = (
            self.get_parameter('high_risk_distance')
            .get_parameter_value()
            .double_value
        )
        self.caution_distance = (
            self.get_parameter('caution_distance')
            .get_parameter_value()
            .double_value
        )
        self.robot_radius_ignore = (
            self.get_parameter('robot_radius_ignore')
            .get_parameter_value()
            .double_value
        )

        self.subscription = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, 10
        )
        self.publisher_ = self.create_publisher(String, 'robot_alerts', 10)

        self.get_logger().info(
            'Situational Awareness Started! Listening to scan and publishing alerts...'
        )

    def lidar_callback(self, msg: LaserScan):
        min_threshold = max(self.robot_radius_ignore, msg.range_min)
        valid_ranges = [
            r
            for r in msg.ranges
            if min_threshold < r < msg.range_max
            and not (math.isnan(r) or math.isinf(r))
        ]

        if not valid_ranges:
            return

        closest_distance = min(valid_ranges)
        alert_msg = String()

        if closest_distance < self.high_risk_distance:
            alert_msg.data = 'HIGH_RISK_STOP'
            self.get_logger().error(
                f'Alert: {alert_msg.data} (Distance: {closest_distance:.2f}m)'
            )
        elif closest_distance < self.caution_distance:
            alert_msg.data = 'CAUTION_SLOW_DOWN'
            self.get_logger().warn(
                f'Alert: {alert_msg.data} (Distance: {closest_distance:.2f}m)'
            )
        else:
            alert_msg.data = 'SAFE_PROCEED'
            self.get_logger().info(
                f'Alert: {alert_msg.data} (Distance: {closest_distance:.2f}m)'
            )

        self.publisher_.publish(alert_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SituationalAwarenessNode()
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
