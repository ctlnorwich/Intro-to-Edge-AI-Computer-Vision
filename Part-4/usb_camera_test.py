import os
import sys

import cv2

# Which camera to use. 0 is usually the first USB camera you plug in.
# Change it here, or for a single run with: CAMERA=1 uv run Part-4/usb_camera_test.py
camera_id = int(os.environ.get("CAMERA", "0"))


def stream_video(cam_id):
    # Initialise the video capture stream using the camera ID
    video_capture = cv2.VideoCapture(cam_id)

    # Check if the camera initialised successfully
    if not video_capture.isOpened():
        print(f"Error: Could not open camera {cam_id}.")
        print("Check v4l2-ctl --list-devices and try a different CAMERA number.")
        sys.exit(1)

    # Ask the camera for a small 640x480 picture. Fewer pixels = less work for the Pi
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera opened successfully! Press 'q' in the video window to quit.")

    # Create an infinite loop to continuously grab frames from the camera
    while True:
        # read() returns two values:
        # read_success (a boolean saying if a frame was grabbed)
        # video_frame (the actual image, a NumPy array of shape height x width x 3)
        read_success, video_frame = video_capture.read()

        # If a frame drops or the camera disconnects, break the loop
        if not read_success:
            print("Error: Frame dropped or camera disconnected.")
            break

        # Display the live video frame in a window
        cv2.imshow("USB Camera Test Stream", video_frame)

        # Wait 1 millisecond for a key press. If it's 'q', leave the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up: release the camera and close all windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_video(camera_id)
