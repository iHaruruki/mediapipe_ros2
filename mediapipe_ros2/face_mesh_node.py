import cv2
import os
import argparse
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def main():
    parser = argparse.ArgumentParser(description="MediaPipe Face Mesh webcam viewer")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width",  type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720,  help="Capture height")
    args = parser.parse_args()

    # Small, light drawing spec (your preference)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {args.camera}")
        return

    # Try to set resolution (may be ignored by some drivers)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # FaceMesh context
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,          # enables iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert BGR -> RGB for inference
            image.flags.writeable = False
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # Draw on the original BGR frame
            image.flags.writeable = True
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Tesselation (full mesh)
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    # Contours (use your small landmark spec)
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    # Irises
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )

            # Selfie view
            cv2.imshow("MediaPipe Face Mesh", cv2.flip(image, 1))
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q to quit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()