"""Path C: Open-vocabulary detection with YOLOE-26.

YOLO26 only knows the 80 things in its training data. YOLOE can look for anything you can describe in words.

usage: uv run Part-8/open_vocab.py [prompt] [prompt] ...

  uv run Part-8/open_vocab.py cup "mobile phone" glasses

Press 'p' to type new prompts in the terminal, 'b' to bake them into a fast NCNN model and 'q' to quit.
"""

import os
import shutil
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO, YOLOE

# Which camera to use. Change it here, or for a single run with: CAMERA=1 uv run Part-8/open_vocab.py
camera_id = int(os.environ.get("CAMERA", "0"))

# The size of the square image the neural network looks at
image_size = int(os.environ.get("IMGSZ", "320"))

# What to look for, from the command line, or these if none are given
start_prompts = sys.argv[1:] or ["person", "cup", "mobile phone"]


def bake_to_ncnn(prompt_model, img_size):
    # The prompts have already been turned into numbers (embeddings) by the text encoder. Exporting folds
    # them into the network's weights, so the NCNN model ONLY knows these prompts, but runs much faster
    output_folder = Path(f"yoloe-26n-seg_baked_{img_size}_ncnn_model")
    exported_folder = prompt_model.export(format="ncnn", imgsz=img_size)

    if output_folder.exists():
        shutil.rmtree(output_folder)
    shutil.move(exported_folder, output_folder)

    return YOLO(output_folder)


def open_vocab(cam_id, img_size, prompts):
    video_capture = cv2.VideoCapture(cam_id)

    if not video_capture.isOpened():
        print(f"Error: Could not open camera {cam_id}.")
        sys.exit(1)

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Only keep the newest frame waiting. Otherwise, when the AI is slower than the camera,
    # frames queue up and we end up looking at the past
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Loading YOLOE-26 (the first run also downloads a 254 MB text encoder)...")
    prompt_model = YOLOE("yoloe-26n-seg.pt")

    # Run the words through the text encoder, so the model knows what to look for
    prompt_model.set_classes(prompts)

    active_model = prompt_model
    mode_text = "PyTorch, live prompts"

    print("Open-vocabulary detection started! 'p' for new prompts, 'b' to bake, 'q' to quit.")

    frames_per_second = 0
    previous_time = time.perf_counter()

    while True:
        read_success, video_frame = video_capture.read()

        if not read_success:
            print("Error: Frame dropped.")
            break

        ai_results = active_model(video_frame, imgsz=img_size, verbose=False)

        # YOLOE also outlines the exact pixels of each object (segmentation), plot() colours them in
        annotated_frame = ai_results[0].plot()

        current_time = time.perf_counter()
        instant_fps = 1 / (current_time - previous_time)
        previous_time = current_time
        frames_per_second = instant_fps if frames_per_second == 0 else 0.9 * frames_per_second + 0.1 * instant_fps

        cv2.putText(
            annotated_frame,
            f"{frames_per_second:.1f} FPS | {mode_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated_frame,
            f"Looking for: {', '.join(prompts)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        cv2.imshow("YOLOE Open-Vocabulary Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("p"):
            # The video freezes while you type, click on the terminal window first!
            new_prompts = input("\nType new prompts separated by commas, then press Enter: ")
            prompts = [prompt.strip() for prompt in new_prompts.split(",") if prompt.strip()] or prompts
            prompt_model.set_classes(prompts)
            active_model = prompt_model
            mode_text = "PyTorch, live prompts"
            frames_per_second = 0
            previous_time = time.perf_counter()

        if key == ord("b"):
            print(f"\nBaking {prompts} into an NCNN model. This takes a minute...")
            active_model = bake_to_ncnn(prompt_model, img_size)
            mode_text = "NCNN, baked prompts"
            frames_per_second = 0
            previous_time = time.perf_counter()

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    open_vocab(camera_id, image_size, start_prompts)
