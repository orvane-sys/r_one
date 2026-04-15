import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MockAINode(Node):
    def __init__(self):
        super().__init__('mock_ai_camera')
        
        # Publishing fake AI detections
        self.publisher_ = self.create_publisher(String, '/ai/detections', 10)
        
        # Publish exactly 5 times a second (5Hz)
        self.timer = self.create_timer(0.2, self.timer_callback)
        self.get_logger().info("Mock AI Started! Broadcasting fake human detections...")

    def timer_callback(self):
        msg = String()
        # We are faking that YOLO saw a "Human" in the "Center" of the camera
        msg.data = "Object: Human, Location: Center"
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MockAINode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()