import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('r_one')
    rviz_config_path = os.path.join(pkg_dir, 'config', 'drive_bot.rviz')

    target_frame_arg = DeclareLaunchArgument(
        'target_frame',
        default_value='odom',
        description='Target frame for global 3D coordinates (e.g. odom or map)'
    )

    open_rviz_arg = DeclareLaunchArgument(
        'open_rviz',
        default_value='true',
        description='Whether to launch RViz2 automatically'
    )

    spatial_vision_node = Node(
        package='r_one',
        executable='spatial_vision_node',
        name='spatial_vision',
        output='screen',
        parameters=[{
            'target_frame': LaunchConfiguration('target_frame'),
            'speed_threshold': 0.15,
            'sync_slop': 0.08,
            'process_every_n_frames': 5,
            'show_debug_window': True,
            'yolo_model': 'yolov8n.pt',
            'img_size': 320
        }]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    return LaunchDescription([
        target_frame_arg,
        open_rviz_arg,
        spatial_vision_node,
        rviz_node
    ])
