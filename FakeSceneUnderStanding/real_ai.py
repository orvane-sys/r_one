import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('real_yolo_vision')
        
        # Tool to translate ROS images to standard OpenCV images
        self.bridge = CvBridge()
        
        self.get_logger().info("Loading YOLOv8 Nano Brain... (This takes a second)")
        # Load the pre-trained model. On the very first run, this will download a tiny 6MB weights file.
        self.model = YOLO('yolov8n.pt') 
        self.get_logger().info("Brain loaded! Connecting to camera...")
        
        # --- LAPTOP COOLING VARIABLES ---
        self.frame_counter = 0
        self.process_every_n_frames = 10 # Only run AI on every 10th frame
        
        # 1. Subscribe to the robot's camera
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        # 2. Publish to the exact same topic your Sensor Fusion is already listening to!
        self.pub = self.create_publisher(String, '/ai/detections', 10)

    def image_callback(self, msg):
        # --- FRAME SKIPPING LOGIC ---
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n_frames != 0:
            return # Skip this frame entirely to save CPU!
            
        # Convert the ROS video feed into a format YOLO can understand
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Run the AI on the current frame
        results = self.model(cv_image, verbose=False)
        
        # Figure out what we are looking at
        detection_string = "Empty Space"
        
        # If the AI sees at least one object...
        if len(results[0].boxes) > 0:
            # Get the ID of the most prominent object
            class_id = int(results[0].boxes[0].cls[0])
            # Translate the ID to English (e.g., "person", "dog", "stop sign")
            detection_string = self.model.names[class_id]
            
        # Send the object name to the Sensor Fusion node!
        msg_out = String()
        msg_out.data = detection_string
        self.pub.publish(msg_out)
        
        # -- BONUS: Show the video feed to the user --
        annotated_frame = results[0].plot() # Draw the bounding boxes
        
        # Add this line to make the window resizable!
        cv2.namedWindow("Robot's Live Vision", cv2.WINDOW_NORMAL) 
        
        cv2.imshow("Robot's Live Vision", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    rclpy.spin(node)
    
    # Clean up the video window when shutting down
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()