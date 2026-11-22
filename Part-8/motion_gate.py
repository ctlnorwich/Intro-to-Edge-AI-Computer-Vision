"""Path B: Motion-gated detection.

Cheap, classical motion detection runs on every frame. The expensive neural network only wakes up
when something moves, the same trick trail cameras and smart doorbells use to save battery.

Press 'q' to quit.
"""

import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# Which camera to use. Change it here, or for a single run with: CAMERA=1 uv run Part-8/motion_gate.py
camera_id = int(os.environ.get("CAMERA", "0"))

# The size of the square image the neural network looks at
image_size = int(os.environ.get("IMGSZ", "320"))

# How big (in pixels) a moving blob has to be to wake the AI up
min_motion_area = 1500

# How long to keep the AI awake after the last movement
stay_awake_seconds = 2.0

# How quickly the background "forgets" the past, from 0 to 1. Higher adapts to lighting faster,
# but someone standing still will fade into the background sooner too
background_learning_rate = 0.05


def motion_gate(cam_id, img_size):
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

    print("Motion gate started! Press 'q' to quit.")

    background = None
    last_motion_time = 0.0
    total_frames = 0
    ai_frames = 0

    while True:
        read_success, current_frame = video_capture.read()

        if not read_success:
            print("Error: Frame dropped.")
            break

        # --- STAGE 1: CLASSICAL MOTION DETECTION (cheap, runs every frame) ---
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        blurred_current = cv2.GaussianBlur(gray_current, (21, 21), 0)

        if background is None:
            # A running average needs decimals, so store the background as floats
            background = blurred_current.astype("float32")
            continue

        # Unlike Part 6, the background is a running average that slowly blends in every new frame,
        # so it copes with the lights changing or a chair being moved
        frame_delta = cv2.absdiff(blurred_current, cv2.convertScaleAbs(background))
        cv2.accumulateWeighted(blurred_current, background, background_learning_rate)

        _, thresh_frame = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
        motion_contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if any(cv2.contourArea(contour) > min_motion_area for contour in motion_contours):
            last_motion_time = time.perf_counter()

        # --- STAGE 2: THE NEURAL NETWORK (expensive, only runs when awake) ---
        ai_awake = time.perf_counter() - last_motion_time < stay_awake_seconds
        total_frames += 1

        if ai_awake:
            ai_frames += 1
            ai_results = detection_model(current_frame, verbose=False)
            display_frame = ai_results[0].plot()
            status_text = "AWAKE: AI on"
            status_colour = (0, 0, 255)
        else:
            display_frame = current_frame
            status_text = "ASLEEP: AI off"
            status_colour = (0, 255, 0)

        # The "duty cycle": what percentage of frames the neural network actually ran on
        duty_cycle = 100 * ai_frames / total_frames

        cv2.putText(
            display_frame,
            f"{status_text} | AI ran on {duty_cycle:.0f}% of frames",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_colour,
            2,
        )

        cv2.imshow("Motion-Gated Detection", display_frame)
        cv2.imshow("Threshold (The Maths)", thresh_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    motion_gate(camera_id, image_size)
