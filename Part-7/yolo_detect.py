import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# Which camera to use. Change it here, or for a single run with: CAMERA=1 uv run Part-7/yolo_detect.py
camera_id = int(os.environ.get("CAMERA", "0"))

# The width and height of the square image the neural network looks at.
# Bigger sees small, far away things better but is slower. Try: IMGSZ=320 uv run Part-7/yolo_detect.py
image_size = int(os.environ.get("IMGSZ", "640"))


def run_yolo(cam_id, img_size):
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
    model_folder = Path(f"yolo26n_{img_size}_ncnn_model")

    if not model_folder.exists():
        print(f"Error: Could not find {model_folder}.")
        print(f"Export it first with: uv run Part-7/export_ncnn.py yolo26n {img_size}")
        sys.exit(1)

    print("Loading optimised NCNN YOLO26 model...")
    detection_model = YOLO(model_folder)

    print("YOLO stream started! Press 'q' in the video window to quit.")

    frames_per_second = 0
    previous_time = time.perf_counter()

    while True:
        read_success, video_frame = video_capture.read()

        if not read_success:
            print("Error: Frame dropped.")
            break

        # Pass the frame to the neural network. It returns a list of results, one per image
        ai_results = detection_model(video_frame, verbose=False)

        # plot() draws the boxes, labels and confidence scores for us
        annotated_frame = ai_results[0].plot()

        # Frames per second of the WHOLE loop (camera + AI + drawing), smoothed so it is readable
        current_time = time.perf_counter()
        instant_fps = 1 / (current_time - previous_time)
        previous_time = current_time
        frames_per_second = instant_fps if frames_per_second == 0 else 0.9 * frames_per_second + 0.1 * instant_fps

        # How long the neural network ALONE took on this frame
        inference_ms = ai_results[0].speed["inference"]

        cv2.putText(
            annotated_frame,
            f"{frames_per_second:.1f} FPS | AI {inference_ms:.0f} ms | {img_size}px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        cv2.imshow("YOLO26 Live Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_yolo(camera_id, image_size)
