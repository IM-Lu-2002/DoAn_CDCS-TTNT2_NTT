import cv2
import torch
import numpy as np
import yaml
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

# Config value
conf_threshold = 0.5
tracking_class_ids = [0, 1, 3, 4, 5, 6, 7]  # Cập nhật dựa trên model của bạn

# Khởi tạo DeepSort
tracker = DeepSort(max_age=180)

# Khởi tạo YOLOv9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(weights="weights/best.pt", device=device, fuse=True)
model = AutoShape(model)

# Đọc tên lớp từ file data.yaml
with open("data_ext/data.yaml", 'r') as f:
    data_yaml = yaml.safe_load(f)
    class_names = data_yaml.get('names', [])  # Lấy danh sách tên lớp

# Tạo màu ngẫu nhiên cho mỗi lớp
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3))

# Khởi tạo VideoCapture để sử dụng webcam
cap = cv2.VideoCapture(0)  # Sử dụng webcam mặc định của máy tính

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Đưa qua model để detect
        results = model(frame)

        detect = []
        for detect_object in results.pred[0]:
            label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
            class_id = int(label)
            if confidence < conf_threshold or class_id not in tracking_class_ids:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            detect.append([[x1, y1, x2-x1, y2-y1], confidence, class_id])

        # Cập nhật, gán ID bằng DeepSort
        tracks = tracker.update_tracks(detect, frame=frame)

        # Vẽ lên màn hình các khung chữ nhật kèm ID
        for track in tracks:
            if track.is_confirmed() and track.time_since_update <= 2 and track.get_det_class() in tracking_class_ids:
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

        # Hiển thị hình ảnh
        cv2.imshow("Object Tracking", frame)
        if cv2.waitKey(10) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
