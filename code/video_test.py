import os
import sys
import subprocess

weights_test = "runs/train/data-custom-vehicle/weights/best.pt"

device = '0'

test_command = [
    sys.executable, 'detect_dual.py',
    '--conf', '0.5',
    '--img', '640',
    '--device', device,
    '--name', 'video_test',
    '--weights', weights_test,
    '--source', 'data_ext/video_02.mp4'
]
# Thực thi câu lệnh trong thư mục yolov9
subprocess.run(test_command, check=True)

