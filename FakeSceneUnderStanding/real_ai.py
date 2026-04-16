import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import math
import time

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
        
        # --- 2. DATA STORAGE & MEMORY ---
        self.fx = None
        self.cx = None
        self.latest_scan = None
        
        # NEW: Memory buffer for kinematics {track_id: (prev_x, prev_y, prev_time)}
        self.object_history = {}
        
        self.frame_counter = 0
        self.process_every_n_frames = 10 
        
        self.get_logger().info("Spatial Vision Node Ready! Waiting for sensor data...")

    def info_callback(self, msg):
        self.fx = msg.k[0]
        self.cx = msg.k[2]

    def scan_callback(self, msg):
        self.latest_scan = msg

    def image_callback(self, msg):
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n_frames != 0:
            return 
            
        if self.fx is None or self.latest_scan is None:
            return 
            
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model.track(cv_image, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        current_time = time.time()
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                obj_name = self.model.names[class_id]
                
                track_id = "N/A"
                if box.id is not None:
                    track_id = int(box.id[0])
                
                x1, y1, x2, y2 = box.xyxy[0]
                u = float((x1 + x2) / 2)
                angle_rad = math.atan((self.cx - u) / self.fx)
                
                try:
                    angle_index = int(angle_rad / self.latest_scan.angle_increment)
                    distance_z = self.latest_scan.ranges[angle_index]
                except IndexError:
                    continue 
                
                if 0.1 < distance_z < 10.0: 
                    pos_x = distance_z * math.cos(angle_rad)
                    pos_y = distance_z * math.sin(angle_rad)
                    
                    velocity = 0.0
                    
                    # --- NEW: KINEMATICS MATH ---
                    if track_id != "N/A":
                        if track_id in self.object_history:
                            prev_x, prev_y, prev_time = self.object_history[track_id]
                            dt = current_time - prev_time
                            
                            if dt > 0:
                                dx = pos_x - prev_x
                                dy = pos_y - prev_y
                                distance_moved = math.sqrt(dx**2 + dy**2)
                                velocity = distance_moved / dt
                                
                        # Update the memory buffer with the newest position
                        self.object_history[track_id] = (pos_x, pos_y, current_time)
                    
                    # Print the output with Speed!
                    self.get_logger().info(f"📍 [ID:{track_id}] {obj_name.upper()} | X:{pos_x:.2f}m Y:{pos_y:.2f}m | Speed: {velocity:.2f} m/s")
        
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