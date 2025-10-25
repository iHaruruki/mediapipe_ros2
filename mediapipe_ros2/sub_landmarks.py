#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from mediapipe_ros2_msgs.msg import PoseLandmark  # header, name, index, x, y


class LandmarkSubscriber(Node):
    def __init__(self):
        super().__init__('landmark_subscriber')

        # Parameters
        self.declare_parameter('topic', '/holistic/pose/landmarks/csv')
        self.declare_parameter('decimals', 2)
        self.topic = self.get_parameter('topic').value
        self.decimals = int(self.get_parameter('decimals').value)

        #qos = QoSPresetProfiles.SENSOR_DATA.value
        self.subscription = self.create_subscription(
            PoseLandmark,          # ← PoseLandmark に変更
            self.topic,
            self.subscriber_callback,
            10
        )
        self.get_logger().info('LandmarkSubscriber node started.')

    def subscriber_callback(self, msg: PoseLandmark):
        # ランドマーク名と座標のみ表示（プレーン）
        d = self.decimals
        print(f"{msg.name},{msg.x:.{d}f},{msg.y:.{d}f}", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()