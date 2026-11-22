import os
import sys
import time

import cv2

# Which camera to use. Change it here, or for a single run with: CAMERA=1 uv run Part-6/motion_detect.py
camera_id = int(os.environ.get("CAMERA", "0"))


def detect_motion(cam_id):
    # Initialise the video capture stream
    video_capture = cv2.VideoCapture(cam_id)

    if not video_capture.isOpened():
        print(f"Error: Could not open camera {cam_id}.")
        sys.exit(1)

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Motion detection started!")
    print("IMPORTANT: Make sure the camera is pointed at a static background when starting.")
    print("Warming up camera sensor for 2 seconds...")

    # Let the camera's auto-exposure and auto-white-balance settle. The camera only starts
    # streaming on the first read(), so keep reading (and throwing away) frames for 2 seconds
    warm_up_end = time.perf_counter() + 2
    while time.perf_counter() < warm_up_end:
        read_success, first_frame = video_capture.read()

    # Use the last frame of the warm up as our static background baseline
    if not read_success:
        print("Error: Could not read the first frame.")
        sys.exit(1)

    # Convert the baseline frame to grayscale and blur it
    # Blurring removes high-frequency noise that might look like tiny movements
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    blurred_first_frame = cv2.GaussianBlur(gray_first_frame, (21, 21), 0)

    while True:
        read_success, current_frame = video_capture.read()

        if not read_success:
            print("Error: Frame dropped.")
            break

        # Process the current frame the exact same way as the baseline
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        blurred_current = cv2.GaussianBlur(gray_current, (21, 21), 0)

        # Compute the absolute difference between the current frame and the baseline
        frame_delta = cv2.absdiff(blurred_first_frame, blurred_current)

        # Any pixel that changed by more than 25 (out of 255) becomes white, everything else black
        _, thresh_frame = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)

        # Dilate (grow) the white areas to fill in gaps
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        # Find the contours (outlines) of the white shapes
        motion_contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the total area of the video frame to filter out lighting glitches
        frame_height, frame_width = current_frame.shape[:2]
        max_contour_area = (frame_width * frame_height) * 0.75

        # Loop through all the separate moving blobs we found
        for contour in motion_contours:
            contour_area = cv2.contourArea(contour)

            # If the blob is too small (noise) OR too large (a lighting change), ignore it
            if contour_area < 500 or contour_area > max_contour_area:
                continue

            # Calculate the bounding box for the moving blob and draw it on the colour frame
            (box_x, box_y, box_w, box_h) = cv2.boundingRect(contour)
            cv2.rectangle(current_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)

        cv2.imshow("Motion Detection", current_frame)
        cv2.imshow("Threshold (The Maths)", thresh_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_motion(camera_id)
