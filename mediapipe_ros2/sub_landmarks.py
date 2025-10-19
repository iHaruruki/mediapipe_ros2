#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class LandmarkSubscriber(Node):
    def __init__(self):
        super().__init__('landmark_subscriber')  # ← クラス初期化でNode名を指定

        # --- Subscriber ---
        self.subscription = self.create_subscription(
            Float32MultiArray,              # メッセージ型
            '/holistic/pose_landmarks',     # トピック名
            self.subscriber_callback,       # コールバック関数
            10                              # QoS深度
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info('LandmarkSubscriber node started.')

    def subscriber_callback(self, msg: Float32MultiArray):
        num = len(msg.data)
        self.get_logger().info(f'Message length: {num}')
        for i, value in enumerate(msg.data):
            self.get_logger().info(f'Landmark {i}: {value:.4f}')


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()