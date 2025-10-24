from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    
    rviz_config_dir = os.path.join(
        get_package_share_directory('mediapipe_ros2'),
        'rviz', 'rviz2.rviz'
    )

    return LaunchDescription([
        Node(
            package='mediapipe_ros2',
            executable='holistic_pose_node',
            name='holistic_pose_node',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            output='screen'
        ),
    ])