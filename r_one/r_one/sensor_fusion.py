#!/usr/bin/env python3

import math
from geometry_msgs.msg import Twist
import rclpy
try:
    from rclpy.executors import ExternalShutdownException
except ImportError:
    ExternalShutdownException = KeyboardInterrupt
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_module')

        self.latest_ai_memory = 'Unknown Object'

        self.ai_sub = self.create_subscription(
            String, 'ai/detections', self.ai_callback, 10
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, 10
        )
        self.vel_pub = self.create_publisher(Twist, 'cmd_vel_fusion', 10)

        self.get_logger().info(
            'Sensor Fusion AEB Node Started! Ready to publish wheel overrides to cmd_vel_fusion...'
        )

    def ai_callback(self, msg: String):
        self.latest_ai_memory = msg.data

    def lidar_callback(self, msg: LaserScan):
        # 1. 360-degree safety perimeter (ignore readings within robot body)
        valid_360_ranges = [
            r
            for r in msg.ranges
            if 0.40 < r < msg.range_max
            and not (math.isnan(r) or math.isinf(r))
        ]

        if valid_360_ranges:
            absolute_closest = min(valid_360_ranges)
            if absolute_closest < 0.45:
                self.get_logger().error(
                    f'🛑 EMERGENCY STOP: Object at {absolute_closest:.2f}m!'
                )
                stop_msg = Twist()
                stop_msg.linear.x = 0.0
                stop_msg.angular.z = 0.0
                self.vel_pub.publish(stop_msg)
                return

        # 2. Forward 60-degree Field of View (+/- 30 degrees from robot forward axis)
        # Evaluated purely via azimuth angles to remain robust across any LiDAR scan convention
        valid_front_ranges = []
        for i, r in enumerate(msg.ranges):
            if 0.10 < r < msg.range_max and not (math.isnan(r) or math.isinf(r)):
                angle = msg.angle_min + i * msg.angle_increment
                norm_angle = math.atan2(math.sin(angle), math.cos(angle))
                if abs(norm_angle) <= math.radians(30):
                    valid_front_ranges.append(r)

        if not valid_front_ranges:
            self.get_logger().info('🛣️ PATH CLEAR: Open space ahead.')
            return

        front_closest = min(valid_front_ranges)

        if front_closest < 0.50:
            self.get_logger().error(
                f'🛑 FRONT COLLISION AVOIDANCE: Obstacle at {front_closest:.2f}m! Braking.'
            )
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.vel_pub.publish(stop_msg)
        elif front_closest < 0.80:
            self.get_logger().warn(
                f'⚠️ CAUTION: Obstacle is {front_closest:.2f}m away. Creeping.'
            )
            slow_msg = Twist()
            slow_msg.linear.x = 0.05
            slow_msg.angular.z = 0.0
            self.vel_pub.publish(slow_msg)
        else:
            self.get_logger().info(
                f'✅ SAFE: Forward distance {front_closest:.2f}m.'
            )


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
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
