from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'mediapipe_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
         (os.path.join('share', package_name), glob('urdf/*')),
        (os.path.join('share', package_name, 'rviz'),   glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Haruki Isono',
    maintainer_email='haruki.isono861@gmail.com',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'face_mesh_node = mediapipe_ros2.face_mesh_node:main',
            'holistic_pose_node = mediapipe_ros2.holistic_pose:main',
            'holistic_pose_csv_node = mediapipe_ros2.holistic_pose_csv:main',
            'subscribe_landmark_node = mediapipe_ros2.sub_landmarks:main',
        ],
    },
)
