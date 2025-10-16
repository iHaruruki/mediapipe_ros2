from setuptools import find_packages, setup

package_name = 'mediapipe_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='ryo.saegusa@syblab.org',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'face_mesh_node = mediapipe_ros2.face_mesh_node:main',
            'holistic_pose_node = mediapipe_ros2.holistic_pose:main'
        ],
    },
)
