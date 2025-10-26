# Copyright (c) 2025 Haruki Isono
# This software is released under the MIT License, see LICENSE.

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSPresetProfiles  # Added: QoS presets
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
import message_filters
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from mediapipe_ros2_msgs.msg import PoseLandmark


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
NUM_LANDMARKS = len(POSE_NAMES)  # = 33


class HolisticPoseTFNode(Node):
    def __init__(self):
        super().__init__('holistic_node')

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
        self.declare_parameter('color_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')  # 親フレーム
        self.declare_parameter('child_prefix', 'landmark')         # 子フレームの接頭辞

        # Landmark2D message publish settings
        self.declare_parameter('publish_landmark2d', True)
        #self.declare_parameter('landmark2d_topic', '/holistic/pose/landmark')

        # TF 配信設定（デフォルトON）
        self.declare_parameter('publish_pose_tf', True)
        self.declare_parameter('tf_rate_hz', 30.0)

        # しきい値（未検出/不確かはTF送らない）
        self.declare_parameter('visibility_threshold', 0.6)  # 0.0〜1.0
        self.declare_parameter('presence_threshold',  0.0)   # 0.0〜1.0
        self.declare_parameter('min_depth_m', 0.1)          # 無効扱いの最小距離[m]
        self.declare_parameter('max_depth_m', 8.0)          # 無効扱いの最大距離[m]

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
        self.color_info_topic = self.get_parameter('color_info_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.depth_info_topic = self.get_parameter('depth_info_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.child_prefix = self.get_parameter('child_prefix').value

        self.publish_landmark2d = bool(self.get_parameter('publish_landmark2d').value)
        #self.landmark2d_topic = self.get_parameter('landmark2d_topic').value

        self.publish_pose_tf = bool(self.get_parameter('publish_pose_tf').value)
        self.tf_rate_hz = float(self.get_parameter('tf_rate_hz').value)

        self.visibility_thr = float(self.get_parameter('visibility_threshold').value)
        self.presence_thr   = float(self.get_parameter('presence_threshold').value)
        self.min_depth_m    = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m    = float(self.get_parameter('max_depth_m').value)

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
        self.pose_landmarks_pub = self.create_publisher(Float32MultiArray, '/holistic/pose/landmarks', 10)
        self.lm2d_pub = self.create_publisher(PoseLandmark, '/holistic/pose/landmarks/csv', 10)

        # ==== TF Broadcaster ====
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_tf_time = self.get_clock().now()

        # ==== Subscribers with synchronization ====
        #sensor_qos = QoSPresetProfiles.SENSOR_DATA.value  # Added: QoS preset
        color_sub = message_filters.Subscriber(self, Image, self.color_topic, qos_profile=10)
        color_info_sub = message_filters.Subscriber(self, CameraInfo, self.color_info_topic, qos_profile=10)
        depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=10)
        depth_info_sub = message_filters.Subscriber(self, CameraInfo, self.depth_info_topic, qos_profile=10)

        ats = message_filters.ApproximateTimeSynchronizer(
            [color_sub, color_info_sub, depth_sub, depth_info_sub], queue_size=20, slop=0.05
        )
        ats.registerCallback(self.synced_callback)

        self.get_logger().info('Holistic Pose TF node ready')

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
                # Fixed: use value= in Parameter
                self.set_parameters([
                    Parameter('roi_enabled', value=True),
                    Parameter('roi_x', value=int(self.roi_x)),
                    Parameter('roi_y', value=int(self.roi_y)),
                    Parameter('roi_width', value=int(self.roi_width)),
                    Parameter('roi_height', value=int(self.roi_height)),
                ])
                self.get_logger().info(f'ROI set: x={self.roi_x}, y={self.roi_y}, w={self.roi_width}, h={self.roi_height}')

    # ====================== Core ======================
    def synced_callback(self, color_msg: Image, color_info: CameraInfo, depth_msg: Image, depth_info: CameraInfo):
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

        annotated_image, (pose_lm_flat, vis_list, pres_list), _ = self.process_image(color)

        # Publish annotated image
        ann = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        ann.header = color_msg.header
        self.annotated_pub.publish(ann)

        # Publish 2D pose landmarks（可視化やログ用）
        self._publish_array(self.pose_landmarks_pub, pose_lm_flat)

        # Added: Publish Landmark2D messages (per landmark)
        if self.publish_landmark2d:
            for landmark_id in range(NUM_LANDMARKS):
                base = 3 * landmark_id
                x = float(pose_lm_flat[base + 0]) if base + 0 < len(pose_lm_flat) else float('nan')
                y = float(pose_lm_flat[base + 1]) if base + 1 < len(pose_lm_flat) else float('nan')
                msg = PoseLandmark()
                msg.header = color_msg.header  # color基準（フル画像座標）
                msg.name = POSE_NAMES[landmark_id] if landmark_id < len(POSE_NAMES) else f"landmark_{landmark_id}"
                msg.index = landmark_id
                msg.x = x
                msg.y = y
                self.lm2d_pub.publish(msg)

        # === TF配信（既定ON） ===
        if self.publish_pose_tf and pose_lm_flat:
            fx = depth_info.k[0]; fy = depth_info.k[4]
            cx = depth_info.k[2]; cy = depth_info.k[5]

            now = self.get_clock().now()
            if (now - self.last_tf_time).nanoseconds >= (1e9 / self.tf_rate_hz):
                self.last_tf_time = now
                self._broadcast_landmarks_tf(
                    pose_lm_flat, vis_list, pres_list,
                    depth_m, fx, fy, cx, cy, color_msg
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
            self.set_parameters([Parameter('roi_enabled', value=False)])
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

    def _broadcast_landmarks_tf(self, flat_xyz, vis_list, pres_list,
                                depth_m, fx, fy, cx, cy, color_msg):
        H, W = depth_m.shape
        n = len(flat_xyz) // 3

        for i in range(n):
            # 1) 可視度・存在度フィルタ
            vvis = vis_list[i] if i < len(vis_list) else 0.0
            vpres = pres_list[i] if i < len(pres_list) else 0.0
            if vvis < self.visibility_thr or vpres < self.presence_thr:
                continue

            # 2) 画面外は送らない（※クリップしない）
            u = float(flat_xyz[3*i + 0])
            v = float(flat_xyz[3*i + 1])
            if not (0.0 <= u < float(W) and 0.0 <= v < float(H)):
                continue

            # 3) 深度のロバスト取得（ゼロ/NaN/外れ値はスキップ）
            u_i = int(u); v_i = int(v)
            z = self._robust_depth(depth_m, v_i, u_i)  # meters
            if (not np.isfinite(z)) or (z < self.min_depth_m) or (z > self.max_depth_m):
                continue

            # 4) バックプロジェクション（pinhole）
            X = (u - cx) / fx * z
            Y = (v - cy) / fy * z

            # 5) TF送信
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

        # 2Dランドマーク（x_pix, y_pix, z_mp）＋ visibility/presence
        pose_landmarks, vis_list, pres_list = self._extract_pose_landmarks(
            results, width, height, roi_offset, (roi_x, roi_y, roi_x2, roi_y2)
        )

        return full_annotated, (pose_landmarks, vis_list, pres_list), (self.roi_x, self.roi_y, self.roi_width, self.roi_height, self.roi_enabled)

    def _extract_pose_landmarks(self, results, width, height, roi_offset, roi_bbox):
        # 事前に固定長を NaN/0.0 で初期化
        xyz_flat = [float('nan')] * (NUM_LANDMARKS * 3)
        vis_list = [0.0] * NUM_LANDMARKS
        pres_list = [0.0] * NUM_LANDMARKS

        if results and results.pose_landmarks:
            roi_w = (roi_bbox[2] - roi_bbox[0])
            roi_h = (roi_bbox[3] - roi_bbox[1])
            lm_list = results.pose_landmarks.landmark
            for i in range(min(NUM_LANDMARKS, len(lm_list))):
                lm = lm_list[i]
                x = lm.x * roi_w + roi_offset[0]
                y = lm.y * roi_h + roi_offset[1]
                z = lm.z  # MediaPipe相対Z（参考値）

                base = 3 * i
                xyz_flat[base + 0] = float(x)
                xyz_flat[base + 1] = float(y)
                xyz_flat[base + 2] = float(z)
                vis_list[i] = float(getattr(lm, 'visibility', 0.0))
                pres_list[i] = float(getattr(lm, 'presence',  0.0))

        return xyz_flat, vis_list, pres_list


def main(args=None):
    rclpy.init(args=args)
    node = HolisticPoseTFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Safe close
        try:
            if hasattr(node, 'holistic') and node.holistic is not None:
                node.holistic.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()