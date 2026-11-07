import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory('aruco_plane_localizer'),
        'config',
        'plane_transform.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=config_path,
            description='Path to the plane transform YAML file.'),
        Node(
            package='aruco_plane_localizer',
            executable='plane_transform_node',
            name='plane_transform_node',
            output='screen',
            parameters=[{'config_file': LaunchConfiguration('config_file')}],
        ),
    ])
