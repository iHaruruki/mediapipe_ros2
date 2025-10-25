# Copyright (c) 2025 Haruki Isono
# This software is released under the MIT License, see LICENSE.

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
import message_filters
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from mediapipe_ros2_msgs import PoseLandmark


POSE_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


class HolisticPoseTFNode(Node):
    def __init__(self):
        # ノード名は要求どおりに固定
        super().__init__('holisticnode')

        # ==== CV Bridge ====
        self.bridge = CvBridge()

        # ==== MediaPipe (Holistic: pose 全身) ====
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic

        # ==== Parameters ====
        self.declare_parameter('min_detection_confidence', 0.6)
        self.declare_parameter('min_tracking_confidence', 0.6)
        self.declare_parameter('model_complexity', 1)           # 0/1/2
        self.declare_parameter('enable_segmentation', False)

        # ROI（任意）
        self.declare_parameter('roi_enabled', False)
        self.declare_parameter('roi_x', 0)
        self.declare_parameter('roi_y', 0)
        self.declare_parameter('roi_width', 400)
        self.declare_parameter('roi_height', 300)

        # Topics / Frames
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')  # 親フレーム
        self.declare_parameter('child_prefix', 'mediapipe_pose')         # 子フレームの接頭辞

        # TF 配信設定（デフォルトON）
        self.declare_parameter('publish_pose_tf', True)
        self.declare_parameter('tf_rate_hz', 30.0)

        # ==== Read params ====
        min_det = float(self.get_parameter('min_detection_confidence').value)
        min_trk = float(self.get_parameter('min_tracking_confidence').value)
        model_complexity = int(self.get_parameter('model_complexity').value)
        enable_seg = bool(self.get_parameter('enable_segmentation').value)

        self.roi_enabled = bool(self.get_parameter('roi_enabled').value)
        self.roi_x = int(self.get_parameter('roi_x').value)
        self.roi_y = int(self.get_parameter('roi_y').value)
        self.roi_width = int(self.get_parameter('roi_width').value)
        self.roi_height = int(self.get_parameter('roi_height').value)

        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.depth_info_topic = self.get_parameter('depth_info_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.child_prefix = self.get_parameter('child_prefix').value

        self.publish_pose_tf = bool(self.get_parameter('publish_pose_tf').value)
        self.tf_rate_hz = float(self.get_parameter('tf_rate_hz').value)

        # ==== MediaPipe Holistic（pose中心）====
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=enable_seg,
            smooth_segmentation=True,
            refine_face_landmarks=False,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk
        )

        # ==== ROI GUI ====
        self.dragging = False
        self.start_point = None
        self.end_point = None
        self._setup_opencv_window()

        # ==== Publishers ====
        self.annotated_pub = self.create_publisher(Image, '/holistic/annotated_image', 10)
        self.pose_landmarks_pub = self.create_publisher(Float32MultiArray, '/holistic/pose_landmarks', 10)

        # ==== TF Broadcaster ====
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_tf_time = self.get_clock().now()

        # ==== Subscribers with synchronization (color + depth + depth_info) ====
        color_sub = message_filters.Subscriber(self, Image, self.color_topic, qos_profile=10)
        depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=10)
        depth_info_sub = message_filters.Subscriber(self, CameraInfo, self.depth_info_topic, qos_profile=10)

        ats = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, depth_info_sub], queue_size=20, slop=0.05
        )
        ats.registerCallback(self.synced_callback)

        self.get_logger().info('Holistic Pose TF node ready: pose 33 pts → 3D → TF配信（デフォルトON）')

    # ====================== GUI (ROI) ======================
    def _setup_opencv_window(self):
        try:
            cv2.namedWindow('Holistic Pose - ROI Selection', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('Holistic Pose - ROI Selection', self._mouse_callback)
        except Exception as e:
            self.get_logger().error(f'Failed to setup OpenCV window: {str(e)}')

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging and self.start_point:
                self.dragging = False
                self.end_point = (x, y)
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                self.roi_x = x1; self.roi_y = y1
                self.roi_width = x2 - x1; self.roi_height = y2 - y1
                self.roi_enabled = True
                self.set_parameters([
                    Parameter('roi_enabled', Parameter.Type.BOOL, True),
                    Parameter('roi_x', Parameter.Type.INTEGER, self.roi_x),
                    Parameter('roi_y', Parameter.Type.INTEGER, self.roi_y),
                    Parameter('roi_width', Parameter.Type.INTEGER, self.roi_width),
                    Parameter('roi_height', Parameter.Type.INTEGER, self.roi_height),
                ])
                self.get_logger().info(f'ROI set: x={self.roi_x}, y={self.roi_y}, w={self.roi_width}, h={self.roi_height}')

    # ====================== Core ======================
    def synced_callback(self, color_msg: Image, depth_msg: Image, depth_info: CameraInfo):
        # --- color ---
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'color cv bridge error: {e}')
            return

        # --- depth (meters) ---
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
            if depth_msg.encoding in ('16UC1', 'mono16'):
                depth_m = depth.astype(np.float32) / 1000.0  # mm → m
            elif depth_msg.encoding in ('32FC1',):
                depth_m = depth.astype(np.float32)
            else:
                depth_m = depth.astype(np.float32)  # best-effort
        except Exception as e:
            self.get_logger().error(f'depth cv bridge error: {e}')
            return

        annotated_image, pose_lm_flat, _ = self.process_image(color)

        # Publish annotated image
        ann = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        ann.header = color_msg.header
        self.annotated_pub.publish(ann)

        # Publish 2D pose landmarks
        self._publish_array(self.pose_landmarks_pub, pose_lm_flat)

        # === TF配信（既定ON） ===
        if self.publish_pose_tf and pose_lm_flat:
            fx = depth_info.k[0]; fy = depth_info.k[4]
            cx = depth_info.k[2]; cy = depth_info.k[5]

            now = self.get_clock().now()
            if (now - self.last_tf_time).nanoseconds >= (1e9 / self.tf_rate_hz):
                self.last_tf_time = now
                self._broadcast_landmarks_tf(
                    pose_lm_flat, depth_m, fx, fy, cx, cy, color_msg
                )

        # ROIウィンドウ
        disp = annotated_image.copy()
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            cv2.rectangle(disp, (self.roi_x, self.roi_y),
                          (self.roi_x + self.roi_width, self.roi_y + self.roi_height), (0, 255, 0), 2)
            cv2.putText(disp, 'ROI', (self.roi_x, self.roi_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.dragging and self.start_point and self.end_point:
            cv2.rectangle(disp, self.start_point, self.end_point, (255, 0, 0), 2)
            cv2.putText(disp, 'Selecting ROI...',
                        (self.start_point[0], self.start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(disp, 'Drag to select ROI  (q: close / r: reset ROI)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Holistic Pose - ROI Selection', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
        elif key == ord('r'):
            self.roi_enabled = False
            self.set_parameters([Parameter('roi_enabled', Parameter.Type.BOOL, False)])
            self.get_logger().info('ROI reset')

    # ---------- helpers ----------
    def _publish_array(self, pub, flat):
        msg = Float32MultiArray()
        msg.data = flat
        pub.publish(msg)

    def _robust_depth(self, depth_m, v, u):
        """3x3 median (ignore zeros) → meters"""
        h, w = depth_m.shape
        v0 = max(0, v-1); v1 = min(h, v+2)
        u0 = max(0, u-1); u1 = min(w, u+2)
        patch = depth_m[v0:v1, u0:u1].reshape(-1)
        vals = patch[np.isfinite(patch) & (patch > 0.0)]
        if vals.size == 0:
            return np.nan
        return float(np.median(vals))

    def _broadcast_landmarks_tf(self, flat_xyz, depth_m, fx, fy, cx, cy, color_msg):
        """Depth（colorに整列済み前提）から3Dを出して関節ごとにTF配信。"""
        n = len(flat_xyz) // 3
        for i in range(n):
            u = float(flat_xyz[3*i + 0])
            v = float(flat_xyz[3*i + 1])

            # 画像境界にクリップ
            u_i = int(np.clip(u, 0, depth_m.shape[1]-1))
            v_i = int(np.clip(v, 0, depth_m.shape[0]-1))
            z = self._robust_depth(depth_m, v_i, u_i)  # meters

            if not np.isfinite(z) or z <= 0.0:
                continue

            # pinhole back-projection
            X = (u - cx) / fx * z
            Y = (v - cy) / fy * z

            # optical(X右,Y下,Z前) → ROS camera_link(X前,Y左,Z上)
            t = TransformStamped()
            t.header.stamp = color_msg.header.stamp
            t.header.frame_id = self.camera_frame
            name = POSE_NAMES[i] if i < len(POSE_NAMES) else f"landmark_{i}"
            t.child_frame_id = f'{self.child_prefix}/{name}'

            t.transform.translation.x = float(X)
            t.transform.translation.y = float(Y)
            t.transform.translation.z = float(z)
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform(t)

    def process_image(self, cv_image):
        """Holisticで全身ポーズのみ処理。描画＋フル画像座標へ復元した33点配列を返す。"""
        height, width = cv_image.shape[:2]

        # ROI crop
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            roi_x = int(np.clip(self.roi_x, 0, width-1))
            roi_y = int(np.clip(self.roi_y, 0, height-1))
            roi_x2 = int(np.clip(roi_x + self.roi_width, 0, width))
            roi_y2 = int(np.clip(roi_y + self.roi_height, 0, height))
            processing_image = cv_image[roi_y:roi_y2, roi_x:roi_x2]
            roi_offset = (roi_x, roi_y)
        else:
            processing_image = cv_image
            roi_offset = (0, 0)
            roi_x = roi_y = 0
            roi_x2 = width
            roi_y2 = height

        # BGR → RGB
        image_rgb = cv2.cvtColor(processing_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = self.holistic.process(image_rgb)

        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Pose（33点）を描画
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # ROIを元画像へ戻す
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            full_annotated = cv_image.copy()
            full_annotated[roi_y:roi_y2, roi_x:roi_x2] = annotated
        else:
            full_annotated = annotated

        # 2Dランドマーク（x_pix, y_pix, z_mp）
        pose_landmarks = self._extract_pose_landmarks(results, width, height, roi_offset, (roi_x, roi_y, roi_x2, roi_y2))

        return full_annotated, pose_landmarks, (self.roi_x, self.roi_y, self.roi_width, self.roi_height, self.roi_enabled)

    def _extract_pose_landmarks(self, results, width, height, roi_offset, roi_bbox):
        landmarks = []
        if results and results.pose_landmarks:
            roi_w = (roi_bbox[2] - roi_bbox[0])
            roi_h = (roi_bbox[3] - roi_bbox[1])
            for lm in results.pose_landmarks.landmark:
                x = lm.x * roi_w + roi_offset[0]
                y = lm.y * roi_h + roi_offset[1]
                z = lm.z  # MediaPipe相対Z（参考値）
                landmarks.extend([x, y, z])
        return landmarks


def main(args=None):
    rclpy.init(args=args)
    node = HolisticPoseTFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
