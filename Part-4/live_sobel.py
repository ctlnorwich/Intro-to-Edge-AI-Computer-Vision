import os
import sys

import cv2

# Which camera to use. Change it here, or for a single run with: CAMERA=1 uv run Part-4/live_sobel.py
camera_id = int(os.environ.get("CAMERA", "0"))


def stream_edges(cam_id):
    # Initialise the video capture stream
    video_capture = cv2.VideoCapture(cam_id)

    if not video_capture.isOpened():
        print(f"Error: Could not open camera {cam_id}.")
        sys.exit(1)

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Edge detection stream opened successfully! Press 'q' to quit.")

    while True:
        # 1. READ: grab the latest frame
        read_success, video_frame = video_capture.read()

        if not read_success:
            print("Error: Frame dropped.")
            break

        # 2. PROCESS: the exact same maths as sobel_edge.py, once per frame
        gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)

        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)

        combined_edges = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

        # 3. DISPLAY: show the original and the result
        cv2.imshow("Original Live Stream", video_frame)
        cv2.imshow("Real-Time Sobel Edges", combined_edges)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_edges(camera_id)
