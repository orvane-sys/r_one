# ROS 2 Spatial Scene Understanding (PSM_015)

## Introduction: What Does This Repository Do?
This repository transforms a robot's basic 2D camera feed into a fully functional **3D Spatial AI and World Model**. 

Out of the box, robots can "see" pixels using a camera and "feel" distances using LiDAR, but these two data streams are entirely separate. This project bridges that gap. By mathematically fusing **YOLOv8 Object Detection** with **2D LiDAR Laser Scans**, this repository enables a ROS 2 robot to:
1. **Identify** objects in its environment (e.g., people, tables, chairs).
2. **Locate** them in physical 3D space with precise Metric Coordinates (X and Y distances).
3. **Track** multiple targets simultaneously using unique IDs to avoid "amnesia" between frames.
4. **Calculate** real-time kinematics (speed and velocity in m/s) of moving targets.
5. **Remember** the environment by building a persistent "World Model" that stores the locations of static objects even when the robot turns its head away.

This code serves as the perception foundation for advanced autonomous behaviors like threat assessment, dynamic obstacle avoidance, and human-following.

---

## Key Features & Objectives Achieved

### 1. LiDAR-Camera Data Fusion (HUMANOID_PSM_011)
**The Concept:** 2D cameras lack depth, and LiDAR lacks object classification. We fused them.
**How it Works:** The node subscribes to `/camera/camera_info` to get the focal length and optical center. When YOLO detects a bounding box, it calculates the horizontal angle of the object and looks up that exact angle in the `/scan` LiDAR array.
**Result:** Real-world Metric Coordinates (Forward X, Lateral Y) for every recognized object.

### 2. Multi-Target Tracking & Unique IDs
**The Concept:** The robot needs to know if the cup it sees now is the same cup it saw 5 frames ago.
**How it Works:** Integrated **ByteTrack** (via Ultralytics) to assign persistent Unique integer IDs to objects across frames. 
**Result:** Solves the "amnesia/flickering" problem. Objects are reliably tracked as `[ID: 1]`, `[ID: 2]`, etc.

### 3. Kinematics (Velocity Vectors)
**The Concept:** Calculating how fast objects are moving in the world.
**How it Works:** Built a short-term memory buffer (`self.object_history`) that logs the X, Y coordinates and the timestamp of each object. Uses standard physics (v = d/t) and the Pythagorean theorem to calculate distance moved between frames.
**Result:** Real-time speed output in meters per second (m/s).

### 4. Object Permanence & The "World Model"
**The Concept:** The robot must remember where things are, even when it looks away.
**How it Works:** If an object's speed is greater than 0.15 m/s, it is tagged `[DYNAMIC]`. Otherwise, it is `[STATIC]`. All `[STATIC]` objects are saved to a global `world_model` dictionary.
**Result:** The robot maintains a persistent map of the furniture and environment, achieving true object permanence without the massive compute load of heavy 3D voxel mapping software.

---

## Installation & Prerequisites

**1. Standard ROS 2 Packages**
Ensure you have ROS 2, Gazebo, and the TurtleBot3 packages installed.
```
sudo apt install ros-<distro>-turtlebot3-gazebo ros-<distro>-turtlebot3-teleop
```
**2. Python Dependencies**
Install the Ultralytics YOLO library and the `lapx` math library (required for ByteTrack's internal calculations).

pip3 install ultralytics
pip3 install lapx

**How to Run the Simulation**
You will need three separate terminals to run the simulation, teleop, and the AI node.

**Terminal 1: Launch the Simulator**
Important: You must use the "waffle" model! The default "burger" model does not have a camera, which will prevent the script from running.
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py

**Terminal 2: Launch the Steering Wheel (Teleop)**
export TURTLEBOT3_MODEL=waffle
ros2 run turtlebot3_teleop teleop_keyboard

**Terminal 3: Launch the Spatial AI Node**
cd ~/ros2_ws/src/r_one/FakeSceneUnderstanding
python3 real_ai.py
