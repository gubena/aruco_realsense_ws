import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    params = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'camera_params.yaml')
    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            namespace='camera',
            parameters=[params],
            output='screen',
        )
    ])