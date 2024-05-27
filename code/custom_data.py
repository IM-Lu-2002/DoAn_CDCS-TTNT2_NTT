import os
import subprocess
import sys

# Đường dẫn đến thư mục gốc của dự án
home_path = 'C:/Users/lemin/OneDrive/Máy tính/DoAn_NDDT'

# Đường dẫn tuyệt đối đến các thư mục con
yolo_path       = os.path.join(home_path, 'yolov9')
data_yaml_path  = os.path.join(yolo_path, "640-Hutech-Vehicle-2", "data.yaml")
cfg_path        = os.path.join(yolo_path, "models", "detect", "yolov9-e-custom.yaml")
weights_path    = os.path.join(yolo_path, "weights", "yolov9-c-converted.pt")
hyp_path        = os.path.join(yolo_path, "data", "hyps", "hyp.scratch-high.yaml")

# Sử dụng Roboflow API
os.chdir(yolo_path)
sys.path.append(yolo_path)  # Đảm bảo Python có thể tìm thấy mô-đun trong thư mục này
from roboflow import Roboflow
rf = Roboflow(api_key="ox3h03sedr8n8BNy9O4O")
project = rf.workspace("ngohoangphuc").project("640-hutech-vehicle-nq2ts")
version = project.version(2)
dataset = version.download("yolov9")

device = '0'

# Xây dựng câu lệnh detect
command = [
    sys.executable, 'train_dual.py',
    '--name', 'data-custom-vehicle',
    '--workers', '8',
    '--device', device,
    '--batch', '4',
    '--img', '640',
    '--min-items', '0',
    '--epochs', '100',
    '--close-mosaic', '15',  # Sửa lỗi thiếu dấu phẩy trong phiên bản trước
    '--data', data_yaml_path,
    '--cfg', cfg_path,
    '--weights', weights_path,
    '--hyp', hyp_path,
]

# Thực thi câu lệnh trong thư mục yolov9
subprocess.run(command)
#
# image_test = os.path.join(yolo_path, "runs/detect/test2/test_01.jpg")
# img = Image.open(image_test)
# img.show()
