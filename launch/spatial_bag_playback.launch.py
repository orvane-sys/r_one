import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_r_one = get_package_share_directory('r_one')
    rviz_config = os.path.join(pkg_r_one, 'config', 'drive_bot.rviz')
    default_bag = os.path.join(
        os.getcwd(), 'sample_bags', 'sample_spatial_run.mcap'
    )

    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value=default_bag,
        description='Path to the .mcap rosbag file or directory',
    )

    loop_arg = DeclareLaunchArgument(
        'loop',
        default_value='true',
        description='Loop bag playback continuously',
    )

    rate_arg = DeclareLaunchArgument(
        'rate',
        default_value='1.0',
        description='Playback rate multiplier (e.g. 1.0, 0.5 for slow motion)',
    )

    target_frame_arg = DeclareLaunchArgument(
        'target_frame',
        default_value='odom',
        description='Global reference frame for 3D coordinates (odom or map)',
    )

    depth_source_arg = DeclareLaunchArgument(
        'depth_source',
        default_value='lidar',
        description="Depth sensor backend: 'lidar' or 'depth_image'",
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz2 visualizer',
    )

    # Rosbag player process with simulation clock publishing
    bag_player = ExecuteProcess(
        cmd=[
            'ros2',
            'bag',
            'play',
            LaunchConfiguration('bag_path'),
            '--clock',
            '--rate',
            LaunchConfiguration('rate'),
            '--loop',
        ],
        output='screen',
    )

    # Spatial vision perception node
    spatial_vision_node = Node(
        package='r_one',
        executable='spatial_vision_node',
        name='spatial_vision',
        output='screen',
        parameters=[
            {
                'target_frame': LaunchConfiguration('target_frame'),
                'depth_source': LaunchConfiguration('depth_source'),
                'speed_threshold': 0.15,
                'sync_slop': 0.08,
                'process_every_n_frames': 2,
                'show_debug_window': False,
                'yolo_model': 'yolov8n.pt',
                'img_size': 320,
                'use_sim_time': True,
            }
        ],
    )

    # RViz2 display
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )

    return LaunchDescription([
        bag_path_arg,
        loop_arg,
        rate_arg,
        target_frame_arg,
        depth_source_arg,
        rviz_arg,
        bag_player,
        spatial_vision_node,
        rviz_node,
    ])
