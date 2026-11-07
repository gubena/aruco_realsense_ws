import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    aruco_tracker_params = os.path.join(
        get_package_share_directory("aruco_tracker"), "config", "params.yaml")
    plane_transform_config = os.path.join(
        get_package_share_directory("aruco_plane_localizer"), "config", "plane_transform.yaml")
    realsense_launch = os.path.join(
        get_package_share_directory("realsense2_camera"), "launch", "garrins_rs.launch.py")

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch)
        ),
        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package="aruco_tracker",
                    executable="aruco_detector_node",
                    name="aruco_detector",
                    output="screen",
                    parameters=[aruco_tracker_params],
                )
            ],
        ),
        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package="aruco_plane_localizer",
                    executable="plane_transform_node",
                    name="plane_transform_node",
                    output="screen",
                    parameters=[{'config_file': plane_transform_config}],
                )
            ],
        ),
    ])
