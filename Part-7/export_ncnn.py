"""Download a YOLO model and convert it to NCNN so it runs fast on the Pi's Arm CPU.

usage: uv run Part-7/export_ncnn.py [model] [image size]

  uv run Part-7/export_ncnn.py                     # yolo26n at 640 pixels
  uv run Part-7/export_ncnn.py yolo26n 320         # the same model, at 320 pixels
  uv run Part-7/export_ncnn.py yolo26n-pose 320    # the pose estimation model
"""

import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


def export_to_ncnn(model_name, image_size):
    output_folder = Path(f"{model_name}_{image_size}_ncnn_model")

    # Downloads the PyTorch (.pt) weights from the internet the first time
    pytorch_model = YOLO(f"{model_name}.pt")

    # Trace the network and convert every layer into NCNN's format. The input size is fixed here,
    # so a model exported at 640 pixels will always look at 640x640, whatever you ask it later
    exported_folder = pytorch_model.export(format="ncnn", imgsz=image_size)

    # Ultralytics always calls the folder <model>_ncnn_model, so add the size to keep several side by side
    if output_folder.exists():
        shutil.rmtree(output_folder)
    shutil.move(exported_folder, output_folder)

    print(f"\nSaved {output_folder} for the Pi!")
    return output_folder


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n"
    image_size = int(sys.argv[2]) if len(sys.argv) > 2 else 640

    export_to_ncnn(model_name, image_size)
