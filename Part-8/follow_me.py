"""Path A: Follow-me tracker.

YOLO26 finds people, ByteTrack gives each one an ID that sticks between frames, and we work out the
steering commands a drone or camera gimbal would need to keep the person you choose in the centre.

Click on a person to lock on. Press 'c' to clear the lock and 'q' to quit.
"""

import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# Which camera to use. Change it here, or for a single run with: CAMERA=1 uv run Part-8/follow_me.py
camera_id = int(os.environ.get("CAMERA", "0"))

# The size of the square image the neural network looks at
image_size = int(os.environ.get("IMGSZ", "320"))

# Which COCO classes to track. 0 is person; try 32 (sports ball) or 16 (dog)
classes_to_track = [0]

# How far the target can drift from the centre (as a fraction of half the frame) before we steer
dead_zone = 0.15

# How much of the frame's height we want the target to fill. Bigger means follow closer
desired_height = 0.6

# Where the mouse was clicked, set by on_mouse_click() and used by the main loop
clicked_point = None


def on_mouse_click(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def find_clicked_track(boxes, point):
    # Return the ID of the box that was clicked on (the smallest one, if boxes overlap)
    best_id, best_area = None, None
    for (x1, y1, x2, y2), track_id in zip(boxes.xyxy.tolist(), boxes.id.int().tolist()):
        if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
            area = (x2 - x1) * (y2 - y1)
            if best_area is None or area < best_area:
                best_id, best_area = track_id, area
    return best_id


def find_predicted_box(tracker, target_id):
    # When the target is hidden, ByteTrack keeps it as a "lost" track for a while, and its
    # Kalman filter keeps predicting where it has moved to. Return that guess as [x, y, w, h]
    for lost_track in tracker.lost_stracks:
        if lost_track.track_id == target_id:
            return lost_track.xywh.tolist()
    return None


def steering_commands(target_box, frame_w, frame_h):
    centre_x, centre_y, _, box_h = target_box

    # How far off we are, from -1 to +1. A real drone would multiply these by a gain and send them
    # to its flight controller: this is the "P" in a PID controller
    error_x = (centre_x - frame_w / 2) / (frame_w / 2)
    error_y = (centre_y - frame_h / 2) / (frame_h / 2)
    error_size = desired_height - box_h / frame_h

    commands = []
    if error_x > dead_zone:
        commands.append(f"YAW RIGHT {abs(error_x):.2f}")
    elif error_x < -dead_zone:
        commands.append(f"YAW LEFT {abs(error_x):.2f}")

    if error_y > dead_zone:
        commands.append(f"TILT DOWN {abs(error_y):.2f}")
    elif error_y < -dead_zone:
        commands.append(f"TILT UP {abs(error_y):.2f}")

    if error_size > dead_zone:
        commands.append("FORWARD")
    elif error_size < -dead_zone:
        commands.append("BACK")

    return commands or ["HOLD"]


def follow_me(cam_id, img_size):
    global clicked_point

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

    window_name = "Follow-Me Tracker"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    print("Tracker started! Click on a person to lock on, 'c' to clear, 'q' to quit.")

    target_id = None
    frames_per_second = 0
    previous_time = time.perf_counter()

    while True:
        read_success, video_frame = video_capture.read()

        if not read_success:
            print("Error: Frame dropped.")
            break

        frame_h, frame_w = video_frame.shape[:2]

        # track() = detect, then match the boxes to the last frame's with ByteTrack.
        # persist=True says this frame follows on from the last one, so the same person keeps the same ID
        ai_results = detection_model.track(
            video_frame, persist=True, tracker="bytetrack.yaml", classes=classes_to_track, verbose=False
        )
        tracked_boxes = ai_results[0].boxes
        visible_ids = tracked_boxes.id.int().tolist() if tracked_boxes.id is not None else []

        annotated_frame = ai_results[0].plot()

        # Lock onto whoever was clicked
        if clicked_point is not None:
            if visible_ids:
                target_id = find_clicked_track(tracked_boxes, clicked_point)
            clicked_point = None

        # Draw a crosshair in the centre, and the dead zone around it where we don't bother steering
        centre = (frame_w // 2, frame_h // 2)
        zone_w, zone_h = int(dead_zone * frame_w / 2), int(dead_zone * frame_h / 2)
        cv2.rectangle(
            annotated_frame,
            (centre[0] - zone_w, centre[1] - zone_h),
            (centre[0] + zone_w, centre[1] + zone_h),
            (255, 255, 255),
            1,
        )
        cv2.drawMarker(annotated_frame, centre, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

        target_box = None
        status_text = "Click on a person to lock on"
        status_colour = (255, 255, 255)

        if target_id in visible_ids:
            # We can see the target
            target_box = tracked_boxes.xywh[visible_ids.index(target_id)].tolist()
            status_text = f"LOCKED ON: id {target_id}"
            status_colour = (0, 255, 0)
        elif target_id is not None:
            # We can't see the target, so ask the Kalman filter where it has probably gone
            target_box = find_predicted_box(detection_model.predictor.trackers[0], target_id)
            if target_box is not None:
                status_text = f"HIDDEN: predicting where id {target_id} went"
                status_colour = (0, 165, 255)
            else:
                status_text = f"LOST id {target_id}: click to lock on again"
                status_colour = (0, 0, 255)
                target_id = None

        if target_box is not None:
            centre_x, centre_y, box_w, box_h = target_box
            top_left = (int(centre_x - box_w / 2), int(centre_y - box_h / 2))
            bottom_right = (int(centre_x + box_w / 2), int(centre_y + box_h / 2))
            cv2.rectangle(annotated_frame, top_left, bottom_right, status_colour, 4)
            cv2.arrowedLine(annotated_frame, centre, (int(centre_x), int(centre_y)), status_colour, 2)

            for line_number, command in enumerate(steering_commands(target_box, frame_w, frame_h)):
                cv2.putText(
                    annotated_frame,
                    command,
                    (10, frame_h - 20 - 30 * line_number),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_colour,
                    2,
                )

        current_time = time.perf_counter()
        instant_fps = 1 / (current_time - previous_time)
        previous_time = current_time
        frames_per_second = instant_fps if frames_per_second == 0 else 0.9 * frames_per_second + 0.1 * instant_fps

        cv2.putText(
            annotated_frame,
            f"{frames_per_second:.1f} FPS | {status_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_colour,
            2,
        )

        cv2.imshow(window_name, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            target_id = None

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    follow_me(camera_id, image_size)
