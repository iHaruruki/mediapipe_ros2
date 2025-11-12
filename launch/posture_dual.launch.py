from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    # RViz config
    rviz_config_dir = os.path.join(
        get_package_share_directory('mediapipe_ros2'),
        'rviz', 'rviz2.rviz'
    )

    # Arguments
    cam1_ns = LaunchConfiguration('cam1_ns')
    cam2_ns = LaunchConfiguration('cam2_ns')
    out1_ns = LaunchConfiguration('out1_ns')
    out2_ns = LaunchConfiguration('out2_ns')
    frame1 = LaunchConfiguration('frame1')
    frame2 = LaunchConfiguration('frame2')

    node_cam1 = Node(
        package='mediapipe_ros2',
        executable='holistic_pose_csv_node',
        name='holistic_pose_csv_node_cam1',
        output='screen',
        parameters=[{
            # Input topics per camera 1
            'color_topic': [ '/', cam1_ns, '/color/image_raw' ],
            'color_info_topic': [ '/', cam1_ns, '/color/camera_info' ],
            'depth_topic': [ '/', cam1_ns, '/depth/image_raw' ],
            'depth_info_topic': [ '/', cam1_ns, '/depth/camera_info' ],
            # TF settings per camera 1
            'camera_frame': frame1,
            'child_prefix': [ out1_ns, '/landmark' ],
        }],
        remappings=[
            # Remap absolute output topics to unique names per camera
            ('/holistic/annotated_image', [ '/', out1_ns, '/holistic/annotated_image' ]),
            ('/holistic/pose/landmarks', [ '/', out1_ns, '/holistic/pose/landmarks' ]),
            ('/holistic/pose/landmarks/csv', [ '/', out1_ns, '/holistic/pose/landmarks/csv' ]),
        ],
    )

    node_cam2 = Node(
        package='mediapipe_ros2',
        executable='holistic_pose_csv_node',
        name='holistic_pose_csv_node_cam2',
        output='screen',
        parameters=[{
            # Input topics per camera 2
            'color_topic': [ '/', cam2_ns, '/color/image_raw' ],
            'color_info_topic': [ '/', cam2_ns, '/color/camera_info' ],
            'depth_topic': [ '/', cam2_ns, '/depth/image_raw' ],
            'depth_info_topic': [ '/', cam2_ns, '/depth/camera_info' ],
            # TF settings per camera 2
            'camera_frame': frame2,
            'child_prefix': [ out2_ns, '/landmark' ],
        }],
        remappings=[
            ('/holistic/annotated_image', [ '/', out2_ns, '/holistic/annotated_image' ]),
            ('/holistic/pose/landmarks', [ '/', out2_ns, '/holistic/pose/landmarks' ]),
            ('/holistic/pose/landmarks/csv', [ '/', out2_ns, '/holistic/pose/landmarks/csv' ]),
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        output='screen',
    )

    return LaunchDescription([
        # Declare camera namespaces (input)
        DeclareLaunchArgument(
            'cam1_ns', default_value='camera1',
            description='Input namespace for camera 1 topics (e.g., camera1 or camera)' 
        ),
        DeclareLaunchArgument(
            'cam2_ns', default_value='camera2',
            description='Input namespace for camera 2 topics (e.g., camera2)'
        ),

        # Declare output base namespaces for holistic topics and TF child prefixes
        DeclareLaunchArgument(
            'out1_ns', default_value='cam1',
            description='Output base namespace for camera 1 publishes/remappings'
        ),
        DeclareLaunchArgument(
            'out2_ns', default_value='cam2',
            description='Output base namespace for camera 2 publishes/remappings'
        ),

        # Frames (adjust to your camera driver frames)
        DeclareLaunchArgument(
            'frame1', default_value='camera1_depth_optical_frame',
            description='TF parent frame for camera 1 depth'
        ),
        DeclareLaunchArgument(
            'frame2', default_value='camera2_depth_optical_frame',
            description='TF parent frame for camera 2 depth'
        ),

        node_cam1,
        node_cam2,
        rviz_node,
    ])
