import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String  # <-- NEW: Import standard string message

class SituationalAwarenessNode(Node):
    def __init__(self):
        super().__init__('situational_awareness_module')
        
        # 1. Subscriber (Listening to the LiDAR)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        # 2. Publisher (Broadcasting the Alerts) <-- NEW
        self.publisher_ = self.create_publisher(String, '/robot_alerts', 10)
        
        self.get_logger().info("Situational Awareness Module Started! Listening and Publishing...")

    def lidar_callback(self, msg):
        # NEW: Ignore 0.0 (errors) AND ignore anything under 0.30m (the robot's own body)
        valid_ranges = [r for r in msg.ranges if r > 0.30]
        
        if not valid_ranges:
            return

        closest_distance = min(valid_ranges)
        
        alert_msg = String() 

        # -- ZONE CLASSIFICATION LOGIC --
        if closest_distance < 0.5:
            alert_msg.data = "HIGH_RISK_STOP"
            self.get_logger().error(f"Alert: {alert_msg.data} (Distance: {closest_distance:.2f}m)")
        elif closest_distance < 1.0:
            alert_msg.data = "CAUTION_SLOW_DOWN"
            self.get_logger().warn(f"Alert: {alert_msg.data} (Distance: {closest_distance:.2f}m)")
        else:
            alert_msg.data = "SAFE_PROCEED"
            self.get_logger().info(f"Alert: {alert_msg.data} (Distance: {closest_distance:.2f}m)")
            
        self.publisher_.publish(alert_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SituationalAwarenessNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()