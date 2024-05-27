import os
import sys
import subprocess

# Xây dựng dường dẫn đến file kết quả đã train được
weights_test = "weights/best.pt"

# Định nghĩa thiết bị sử dụng để huấn luyện, '0' thường là GPU đầu tiên
device = '0'

# Xây dựng câu lệnh detect để huấn luyện
test_command = [
    sys.executable, 'detect_dual.py',       # Sử dụng file detect_dual.py để train
    '--conf', '0.6',                        # Ngưỡng tin cậy cho phát hiện
    '--img', '640',                         # Kích thước hình ảnh đầu vào
    '--device', device,                     # Thiết bị để thực thi
    '--name', 'image_test',                 # Tên thư mục lưu kết quả
    '--weights', weights_test,              # Đường dẫn đến file đã train custom
    '--source', 'data_ext/image_03.jpg',     # Nguồn dữ liệu đầu vào là một video
]

# Thực thi câu lệnh để bắt đầu quá trình huấn luyện
subprocess.run(test_command, check=True)
