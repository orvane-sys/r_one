import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import math

class SpatialVisionNode(Node):
    def __init__(self):
        super().__init__('spatial_vision')
        self.bridge = CvBridge()
        self.get_logger().info("Loading YOLOv8 Nano Brain...")
        self.model = YOLO('yolov8n.pt') 
        
        # --- 1. SENSOR SUBSCRIPTIONS ---
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # --- 2. DATA STORAGE ---
        self.fx = None
        self.cx = None
        self.latest_scan = None
        
        # Laptop Cooling
        self.frame_counter = 0
        self.process_every_n_frames = 5 
        
        self.get_logger().info("Spatial Vision Node Ready! Waiting for sensor data...")

    def info_callback(self, msg):
        # Extract the camera intrinsics from the K matrix
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.k[0]
        self.cx = msg.k[2]

    def scan_callback(self, msg):
        # Always keep the most recent LiDAR sweep in memory
        self.latest_scan = msg

    def image_callback(self, msg):
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n_frames != 0:
            return 
            
        # Don't run AI until we have our intrinsics and LiDAR data!
        if self.fx is None or self.latest_scan is None:
            return 
            
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # Run the AI in TRACKING mode so it remembers objects
        results = self.model.track(cv_image, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            class_id = int(box.cls[0])
            obj_name = self.model.names[class_id]
            
            # --- EXTRACT THE UNIQUE ID ---
            track_id = "N/A"
            if box.id is not None:
                track_id = int(box.id[0])
            
            # --- THE FUSION MATH ---
            x1, y1, x2, y2 = box.xyxy[0]
            u = float((x1 + x2) / 2)
            angle_rad = math.atan((self.cx - u) / self.fx)
            angle_index = int(angle_rad / self.latest_scan.angle_increment)
            distance_z = self.latest_scan.ranges[angle_index]
            
            if 0.1 < distance_z < 10.0: 
                pos_x = distance_z * math.cos(angle_rad)
                pos_y = distance_z * math.sin(angle_rad)
                
                # Print the output WITH the Unique License Plate ID!
                self.get_logger().info(f"📍 [ID: {track_id}] {obj_name.upper()} | Forward (X): {pos_x:.2f}m | Lateral (Y): {pos_y:.2f}m")
        
        annotated_frame = results[0].plot() 
        cv2.namedWindow("Robot's 3D Spatial Vision", cv2.WINDOW_NORMAL) 
        cv2.imshow("Robot's 3D Spatial Vision", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SpatialVisionNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()