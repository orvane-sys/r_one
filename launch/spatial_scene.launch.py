import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_r_one = get_package_share_directory('r_one')

    # Default to the animated spatial perception demo world
    default_world = os.path.join(pkg_r_one, 'worlds', 'spatial_demo.world')
    if not os.path.exists(default_world):
        default_world = os.path.join(pkg_r_one, 'worlds', 'empty.world')

    default_rviz_config = os.path.join(pkg_r_one, 'config', 'drive_bot.rviz')

    # --- Launch Arguments ---
    sim_arg = DeclareLaunchArgument(
        'sim',
        default_value='true',
        description='Whether to launch the Gazebo simulation environment and robot spawner',
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='Path to Gazebo world file (defaults to spatial_demo.world)',
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to launch RViz2 with pre-configured 3D perception displays',
    )

    target_frame_arg = DeclareLaunchArgument(
        'target_frame',
        default_value='odom',
        description='Global reference frame for 3D coordinates and world model (e.g. odom or map)',
    )

    depth_source_arg = DeclareLaunchArgument(
        'depth_source',
        default_value='lidar',
        description="Depth source for 3D coordinate estimation: 'lidar' or 'depth_image'",
    )

    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='camera/depth/image_raw',
        description='Depth image topic name when depth_source is set to depth_image',
    )

    speed_threshold_arg = DeclareLaunchArgument(
        'speed_threshold',
        default_value='0.15',
        description='Velocity threshold (m/s) above which objects are classified as DYNAMIC',
    )

    sync_slop_arg = DeclareLaunchArgument(
        'sync_slop',
        default_value='0.08',
        description='Time synchronizer tolerance window in seconds between sensors',
    )

    process_every_n_arg = DeclareLaunchArgument(
        'process_every_n_frames',
        default_value='5',
        description='Process every N-th frame to reduce compute overhead',
    )

    show_camera_window_arg = DeclareLaunchArgument(
        'show_camera_window',
        default_value='false',
        description='Open OpenCV 2D detection debug window (RViz displays camera natively)',
    )

    yolo_model_arg = DeclareLaunchArgument(
        'yolo_model',
        default_value='yolov8n.pt',
        description='YOLOv8 weights file or model path',
    )

    img_size_arg = DeclareLaunchArgument(
        'img_size',
        default_value='320',
        description='Input image resolution for YOLO inference',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Enforce simulation clock across all nodes',
    )

    # --- 1. Simulation Components (Optional) ---
    rsp_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(pkg_r_one, 'launch', 'rsp.launch.py')]
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items(),
    )

    gazebo_cmd = [
        'gazebo',
        '--verbose',
        '-s',
        'libgazebo_ros_init.so',
        '-s',
        'libgazebo_ros_factory.so',
        LaunchConfiguration('world'),
    ]

    gazebo_process = ExecuteProcess(
        cmd=gazebo_cmd,
        output='screen',
    )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic',
            'robot_description',
            '-entity',
            'my_bot',
            '-x',
            '-2.0',
            '-y',
            '-0.5',
            '-z',
            '0.1',
        ],
        output='screen',
    )

    diff_drive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_cont'],
        output='screen',
    )

    joint_broad_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_broad'],
        output='screen',
    )

    simulation_group = GroupAction(
        condition=IfCondition(LaunchConfiguration('sim')),
        actions=[
            rsp_launch,
            gazebo_process,
            spawn_robot,
            diff_drive_spawner,
            joint_broad_spawner,
        ],
    )

    # --- 2. Spatial Perception AI Node ---
    spatial_vision_node = Node(
        package='r_one',
        executable='spatial_vision_node',
        name='spatial_vision',
        output='screen',
        parameters=[
            {
                'target_frame': LaunchConfiguration('target_frame'),
                'depth_source': LaunchConfiguration('depth_source'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'speed_threshold': LaunchConfiguration('speed_threshold'),
                'sync_slop': LaunchConfiguration('sync_slop'),
                'process_every_n_frames': LaunchConfiguration(
                    'process_every_n_frames'
                ),
                'show_debug_window': LaunchConfiguration('show_camera_window'),
                'yolo_model': LaunchConfiguration('yolo_model'),
                'img_size': LaunchConfiguration('img_size'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }
        ],
    )

    # --- 3. RViz2 Visualization (Optional) ---
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', default_rviz_config],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )

    return LaunchDescription([
        sim_arg,
        world_arg,
        rviz_arg,
        target_frame_arg,
        depth_source_arg,
        depth_topic_arg,
        speed_threshold_arg,
        sync_slop_arg,
        process_every_n_arg,
        show_camera_window_arg,
        yolo_model_arg,
        img_size_arg,
        use_sim_time_arg,
        simulation_group,
        spatial_vision_node,
        rviz_node,
    ])
