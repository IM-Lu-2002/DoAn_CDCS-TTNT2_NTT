import cv2
import torch
import numpy as np
import yaml
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from screeninfo import get_monitors

# Config value
video_path = "data_ext/video_02.mp4"    # Dường dẫn video
conf_threshold = 0.5                    # Ngưỡng tin ậy để lọc kết quả phát hiện
tracking_class_ids = [1, 2, 3, 5, 7]    # Danh sách ID lớp dối tượng cần theo dỗi

# Thiết lập mô hình Yolov9
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = DetectMultiBackend(weights="weights/bestt.pt", device=device, fuse=True)
model   = AutoShape(model) # Tự động điều chỉnh kích thước và định dạng đầu vào

# Khởi tạo tracker DeepSort với khung hình tối đa cho các đối tượng theo dõi
tracker = DeepSort(max_age=30)

# Đọc file YAML để lấy danh sách tên các lớp đối tượng từ tập coco.yaml
with open("data_ext/dataa.yaml", 'r') as f:
    data_yaml   = yaml.safe_load(f)
    class_names = data_yaml.get('names', [])  # Lấy danh sách tên lớp

# Tạo màu ngẫu nhiên cho mỗi lớp
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3))

# Khởi tạo danh sách theo dõi
tracks = []

# Lấy kích thước màn hình và tính nửa kích thước
monitor         = get_monitors()[0]
display_width   = monitor.width // 2
display_height  = monitor.height // 2

# Khởi tạo VideoCapture
cap = cv2.VideoCapture(video_path)

# Vòng lặp xử lý từng khung hình của video
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện đối tượng trong khung hình
        results = model(frame)
        detect = []      # Danh sách để lưu các đối tượng được phát hiện
        for detect_object in results.pred[0]:
            label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
            class_id = int(label)
            if confidence < conf_threshold or class_id not in tracking_class_ids:
                continue    # Bỏ qua những đối tượng không đạt ngưỡng tin cậy hoặc không nằm trong danh sách lớp cần theo dõi

            # Chuẩn bị thông tin đối tượng cho DeepSort
            x1, y1, x2, y2 = map(int, bbox)
            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        # Cập nhật thông tin theo dõi đối tượng, gán ID bằng DeepSort
        tracks = tracker.update_tracks(detect, frame=frame)

        # Vẽ và hiển thị khung hình cho đối tượng theo dõi
        for track in tracks:
            if track.is_confirmed() and track.get_det_class() in tracking_class_ids:
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                color = colors[class_id]
                B, G, R = map(int, color)
                label = f"{class_names[class_id]}-{track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Điều chỉnh kích thước khung hình
        frame_display = cv2.resize(frame, (display_width, display_height))
        cv2.imshow("Object Tracking", frame_display)

        # hiển thị kêt quả và cho phép người dùng thoát bằng cách nhấn nút 'q'
        if cv2.waitKey(1) == ord("q"):
            break

# Giải phóng tài nguyên và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
