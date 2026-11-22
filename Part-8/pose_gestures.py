"""Path D: Pose estimation and gesture control.

YOLO26-pose finds 17 keypoints on every person (nose, eyes, ears, shoulders, elbows, wrists, hips,
knees and ankles). We use them to turn your arms into a controller.

Press 'q' to quit.
"""

import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# Which camera to use. Change it here, or for a single run with: CAMERA=1 uv run Part-8/pose_gestures.py
camera_id = int(os.environ.get("CAMERA", "0"))

# The size of the square image the neural network looks at
image_size = int(os.environ.get("IMGSZ", "320"))

# Keypoint numbers in the COCO pose format. "Left" means the person's own left,
# which is on the RIGHT of the image when they are facing the camera
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10

# How sure the network has to be about a keypoint before we trust it, from 0 to 1
min_keypoint_confidence = 0.5


def hand_raised(keypoints, confidences, wrist, shoulder):
    if confidences[wrist] < min_keypoint_confidence or confidences[shoulder] < min_keypoint_confidence:
        return False

    # Images count y DOWNWARDS from the top, so "above" means a smaller y
    return keypoints[wrist][1] < keypoints[shoulder][1]


def read_gesture(keypoints, confidences):
    left_up = hand_raised(keypoints, confidences, LEFT_WRIST, LEFT_SHOULDER)
    right_up = hand_raised(keypoints, confidences, RIGHT_WRIST, RIGHT_SHOULDER)

    if left_up and right_up:
        return "BOTH HANDS UP: TAKE OFF"
    if left_up:
        return "LEFT HAND UP: GO LEFT"
    if right_up:
        return "RIGHT HAND UP: GO RIGHT"
    return "NO GESTURE"


def pose_gestures(cam_id, img_size):
    video_capture = cv2.VideoCapture(cam_id)

    if not video_capture.isOpened():
        print(f"Error: Could not open camera {cam_id}.")
        sys.exit(1)

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Only keep the newest frame waiting. Otherwise, when the AI is slower than the camera,
    # frames queue up and we end up looking at the past
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # The folder made by Part-7/export_ncnn.py. Each one only works at the image size it was exported at
    model_folder = Path(f"yolo26n-pose_{img_size}_ncnn_model")

    if not model_folder.exists():
        print(f"Error: Could not find {model_folder}.")
        print(f"Export it first with: uv run Part-7/export_ncnn.py yolo26n-pose {img_size}")
        sys.exit(1)

    print("Loading optimised NCNN YOLO26-pose model...")
    pose_model = YOLO(model_folder)

    print("Pose gestures started! Raise a hand. Press 'q' to quit.")

    last_gesture = None
    frames_per_second = 0
    previous_time = time.perf_counter()

    while True:
        read_success, video_frame = video_capture.read()

        if not read_success:
            print("Error: Frame dropped.")
            break

        ai_results = pose_model(video_frame, verbose=False)

        # plot() draws a skeleton joining up the keypoints
        annotated_frame = ai_results[0].plot()

        # keypoints.xy holds [x, y] for each keypoint of each person, keypoints.conf how sure it is of each
        pose_keypoints = ai_results[0].keypoints
        gesture = "NO ONE IN VIEW"

        if pose_keypoints is not None and pose_keypoints.conf is not None and len(pose_keypoints) > 0:
            # People come out sorted by confidence, so [0] is the person the network is surest about
            gesture = read_gesture(pose_keypoints.xy[0].tolist(), pose_keypoints.conf[0].tolist())

        # A robot would act on this. We print it whenever it changes
        if gesture != last_gesture:
            print(f"Command: {gesture}")
            last_gesture = gesture

        current_time = time.perf_counter()
        instant_fps = 1 / (current_time - previous_time)
        previous_time = current_time
        frames_per_second = instant_fps if frames_per_second == 0 else 0.9 * frames_per_second + 0.1 * instant_fps

        cv2.putText(
            annotated_frame,
            f"{frames_per_second:.1f} FPS | {gesture}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Pose Gestures", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    pose_gestures(camera_id, image_size)
