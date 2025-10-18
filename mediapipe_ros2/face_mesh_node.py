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


class FaceMeshNode(Node):
    def __init__(self):
        super().__init__('facemeshnode')

        # ==== CV Bridge ====
        self.bridge = CvBridge()

        # ==== MediaPipe (FaceMesh only) ====
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        # ==== Parameters ====
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)

        self.declare_parameter('roi_enabled', False)
        self.declare_parameter('roi_x', 0)
        self.declare_parameter('roi_y', 0)
        self.declare_parameter('roi_width', 400)
        self.declare_parameter('roi_height', 300)

        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('camera_frame', 'camera_link')

        # 顔 478 点の TF は重いので既定 OFF
        self.declare_parameter('publish_face_tf', False)
        self.declare_parameter('tf_rate_hz', 30.0)

        # ==== Read params ====
        min_det = float(self.get_parameter('min_detection_confidence').value)
        min_trk = float(self.get_parameter('min_tracking_confidence').value)

        self.roi_enabled = bool(self.get_parameter('roi_enabled').value)
        self.roi_x = int(self.get_parameter('roi_x').value)
        self.roi_y = int(self.get_parameter('roi_y').value)
        self.roi_width = int(self.get_parameter('roi_width').value)
        self.roi_height = int(self.get_parameter('roi_height').value)

        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.depth_info_topic = self.get_parameter('depth_info_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.publish_face_tf = bool(self.get_parameter('publish_face_tf').value)
        self.tf_rate_hz = float(self.get_parameter('tf_rate_hz').value)

        # ==== MediaPipe FaceMesh ====
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk
        )

        # ==== ROI GUI ====
        self.dragging = False
        self.start_point = None
        self.end_point = None
        self.setup_opencv_window()

        # ==== Publishers (Face only) ====
        self.annotated_pub = self.create_publisher(Image, '/facemesh/annotated_image', 10)
        self.face_landmarks_pub = self.create_publisher(Float32MultiArray, '/facemesh/face_landmarks', 10)

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

        self.get_logger().info('FaceMesh node initialized (Face only; depth→3D & optional TF available)')

    # ====================== GUI (ROI) ======================
    def setup_opencv_window(self):
        try:
            cv2.namedWindow('FaceMesh - ROI Selection', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('FaceMesh - ROI Selection', self.mouse_callback)
            self.get_logger().info('OpenCV window setup for ROI selection')
        except Exception as e:
            self.get_logger().error(f'Failed to setup OpenCV window: {str(e)}')

    def mouse_callback(self, event, x, y, flags, param):
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
                self.roi_x = x1
                self.roi_y = y1
                self.roi_width = x2 - x1
                self.roi_height = y2 - y1
                self.roi_enabled = True
                self.set_parameters([
                    Parameter('roi_enabled', Parameter.Type.BOOL, True),
                    Parameter('roi_x', Parameter.Type.INTEGER, self.roi_x),
                    Parameter('roi_y', Parameter.Type.INTEGER, self.roi_y),
                    Parameter('roi_width', Parameter.Type.INTEGER, self.roi_width),
                    Parameter('roi_height', Parameter.Type.INTEGER, self.roi_height),
                ])
                self.get_logger().info(
                    f'ROI set: x={self.roi_x}, y={self.roi_y}, w={self.roi_width}, h={self.roi_height}'
                )

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

        annotated_image, face_lm, roi_ctx = self.process_image(color)

        # Publish annotated image
        ann = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        ann.header = color_msg.header
        self.annotated_pub.publish(ann)

        # Publish 2D face landmarks (x_pix, y_pix, z_mp ...)
        self._publish_array(self.face_landmarks_pub, face_lm)

        # 3D projection & TF
        fx = depth_info.k[0]; fy = depth_info.k[4]
        cx = depth_info.k[2]; cy = depth_info.k[5]

        now = self.get_clock().now()
        if (now - self.last_tf_time).nanoseconds >= (1e9 / self.tf_rate_hz):
            self.last_tf_time = now
            if self.publish_face_tf and face_lm:
                self._broadcast_landmarks_tf(face_lm, depth_m, fx, fy, cx, cy, color_msg)

        # Show ROI helper window
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
        cv2.imshow('FaceMesh - ROI Selection', disp)
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
        """Broadcast TF for each face landmark using depth (registered to color)."""
        n = len(flat_xyz) // 3
        for i in range(n):
            u = float(flat_xyz[3*i + 0])
            v = float(flat_xyz[3*i + 1])

            u_i = int(np.clip(u, 0, depth_m.shape[1]-1))
            v_i = int(np.clip(v, 0, depth_m.shape[0]-1))
            z = self._robust_depth(depth_m, v_i, u_i)  # meters

            if not np.isfinite(z) or z <= 0.0:
                continue

            X = (u - cx) / fx * z
            Y = (v - cy) / fy * z
            # Convert from optical to ROS camera frame convention
            t = TransformStamped()
            t.header.stamp = color_msg.header.stamp
            t.header.frame_id = self.camera_frame
            t.child_frame_id = f'face_{i}'
            # Optical frame (X right, Y down, Z forward) → ROS (X forward, Y left, Z up)
            t.transform.translation.x = float(z)
            t.transform.translation.y = -float(X)
            t.transform.translation.z = -float(Y)
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)

    def process_image(self, cv_image):
        """Run FaceMesh on ROI (if enabled), draw, and return annotated image + flat face landmarks."""
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

        face_results = self.face_mesh.process(image_rgb)

        image_rgb.flags.writeable = True
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw face mesh (tesselation + contours + irises)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

                self.mp_drawing.draw_landmarks(
                    annotated, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())

                self.mp_drawing.draw_landmarks(
                    annotated, face_landmarks, self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Stitch back into full image if ROI
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            full_annotated = cv_image.copy()
            full_annotated[roi_y:roi_y2, roi_x:roi_x2] = annotated
        else:
            full_annotated = annotated

        # Extract face landmarks in full-image pixel coords (x_pix, y_pix, z_mp)
        face_landmarks = self.extract_face_landmarks(face_results, width, height, roi_offset, (roi_x, roi_y, roi_x2, roi_y2))

        roi_ctx = (self.roi_x, self.roi_y, self.roi_width, self.roi_height, self.roi_enabled)
        return full_annotated, face_landmarks, roi_ctx

    def extract_face_landmarks(self, results, width, height, roi_offset, roi_bbox):
        landmarks = []
        if results and results.multi_face_landmarks:
            # max_num_faces=1 なので先頭のみ使用
            face_lm = results.multi_face_landmarks[0]
            roi_w = (roi_bbox[2] - roi_bbox[0])
            roi_h = (roi_bbox[3] - roi_bbox[1])
            for lm in face_lm.landmark:
                x = lm.x * roi_w + roi_offset[0]
                y = lm.y * roi_h + roi_offset[1]
                z = lm.z  # MediaPipe の相対Z（参考値）
                landmarks.extend([x, y, z])
        return landmarks


def main(args=None):
    rclpy.init(args=args)
    node = FaceMeshNode()
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
