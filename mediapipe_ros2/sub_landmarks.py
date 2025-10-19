import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class LandmarkSubscriber(Node):
    def __init__('landmark_subscriber')
    self.subscription = self.create_subscription(
        Float32MultiArray,
        'landmarks',
        self.subscriber_callback,
        10)
    self.subscription  # prevent unused variable warning

    def subscriber_callback(self, msg):
        num = len(msg.data)
        self.get_logger().info('Message length: %d' % num)
        for i in range(num):
            self.get_logger().info('Landmark %d: %f' % (i, msg.data[i]))

def main(args=None):
    rclpy.init(args=args)

    landmark_subscriber = LandmarkSubscriber()

    rclpy.spin(landmark_subscriber)

    landmark_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        