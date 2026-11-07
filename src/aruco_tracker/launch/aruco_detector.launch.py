import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("aruco_tracker"), "config", "params.yaml"
    )

    return LaunchDescription([
        Node(
            package="aruco_tracker",
            executable="aruco_detector_node",
            name="aruco_detector",
            output="screen",
            parameters=[params_file],
        )
    ])
