import os
import sys
import time

import cv2

# Which camera to use. Change it here, or for a single run with: CAMERA=1 uv run Part-5/face_detect.py
camera_id = int(os.environ.get("CAMERA", "0"))

# Which pre-trained Haar cascade to use. OpenCV ships a folder of them, try swapping this for
# "haarcascade_eye.xml", "haarcascade_smile.xml" or even "haarcascade_frontalcatface.xml"
cascade_file = "haarcascade_frontalface_default.xml"


def detect_faces(cam_id):
    # Load the cascade: thousands of simple light/dark rectangle tests, chosen by training in 2001
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_file)

    if face_cascade.empty():
        print(f"Error: Could not load {cascade_file}.")
        sys.exit(1)

    video_capture = cv2.VideoCapture(cam_id)

    if not video_capture.isOpened():
        print(f"Error: Could not open camera {cam_id}.")
        sys.exit(1)

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Viola-Jones face detection started! Press 'q' to quit.")

    while True:
        read_success, video_frame = video_capture.read()

        if not read_success:
            print("Error: Frame dropped.")
            break

        detect_start = time.perf_counter()

        # Haar features only care about brightness, so throw away the colour
        gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

        # Spread the brightness values out so faces in dim light still have contrast
        gray_frame = cv2.equalizeHist(gray_frame)

        # Slide a window over the image at lots of different sizes, running the cascade at each spot
        detected_faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,  # shrink the image by 10% each pass to find bigger faces
            minNeighbors=5,  # a face must be found by 5 overlapping windows to count
            minSize=(40, 40),  # ignore anything smaller than 40x40 pixels
        )

        detect_ms = (time.perf_counter() - detect_start) * 1000

        # Draw a box around every face found
        for box_x, box_y, box_w, box_h in detected_faces:
            cv2.rectangle(video_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)

        cv2.putText(
            video_frame,
            f"{len(detected_faces)} face(s) | {detect_ms:.0f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Viola-Jones Face Detection", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_faces(camera_id)
