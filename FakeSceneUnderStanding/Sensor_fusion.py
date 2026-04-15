import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist # <--- NEW: Importing the movement message

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_module')
        
        self.latest_ai_memory = "Unknown Object"
        
        # 1. Listen to the AI Camera
        self.ai_sub = self.create_subscription(String, '/ai/detections', self.ai_callback, 10)
        
        # 2. Listen to the LiDAR
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        
        # 3. NEW: The Muscle Connection (Publisher for wheel commands)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel_fusion', 10)
        
        self.get_logger().info("Sensor Fusion AEB Node Started! Ready to hijack wheels if necessary...")

    def ai_callback(self, msg):
        self.latest_ai_memory = msg.data

    def lidar_callback(self, msg):
        # --- 1. THE 360-DEGREE SAFETY SYSTEM (Lizard Brain) ---
        valid_360_ranges = [r for r in msg.ranges if 0.40 < r < float('inf')]
        
        if valid_360_ranges:
            absolute_closest = min(valid_360_ranges)
            if absolute_closest < 0.45: 
                self.get_logger().error(f"🛑 REAR/SIDE EMERGENCY: Object at {absolute_closest:.2f}m!")
                stop_msg = Twist()
                stop_msg.linear.x = 0.0
                stop_msg.angular.z = 0.0
                self.vel_pub.publish(stop_msg) # WE PUBLISH TO OVERRIDE
                return 

        # --- 2. THE 60-DEGREE PERCEPTION SYSTEM (Smart Brain) ---
        left_fov = msg.ranges[0:30]
        right_fov = msg.ranges[330:360]
        camera_view_ranges = left_fov + right_fov

        valid_front_ranges = [r for r in camera_view_ranges if 0.10 < r < float('inf')]
        
        if not valid_front_ranges:
            self.get_logger().info("🛣️ PATH CLEAR: Open space ahead. (Manual Control Active)")
            # ❌ REMOVED CRUISE CONTROL: Do NOT publish here! 
            # Silence allows the keyboard to take over.
            return 

        front_closest = min(valid_front_ranges)
        
        if front_closest < 0.50:
            self.get_logger().error(f"🛑 FRONT COLLISION AVOIDANCE: Wall at {front_closest:.2f}m! BRAKING.")
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.vel_pub.publish(stop_msg) # WE PUBLISH TO OVERRIDE
            
        elif front_closest < 0.8:
            self.get_logger().warn(f"⚠️ CAUTION: Object is {front_closest:.2f}m away. Creeping forward.")
            slow_msg = Twist()
            slow_msg.linear.x = 0.05
            slow_msg.angular.z = 0.0
            self.vel_pub.publish(slow_msg) # WE PUBLISH TO FORCE SLOW DOWN
            
        else:
            self.get_logger().info(f"✅ SAFE: Object is {front_closest:.2f}m away. (Manual Control Active)")
            # ❌ REMOVED CRUISE CONTROL: Do NOT publish here!

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
