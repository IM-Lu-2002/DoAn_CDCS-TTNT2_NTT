import cv2
import torch
import yaml
import numpy as np

from screeninfo import get_monitors
from count_config import get_count_config
from models.common import DetectMultiBackend, AutoShape
from deep_sort_realtime.deepsort_tracker import DeepSort

# Cấu hình ngưỡng tin cậy và ID của các lớp đối tượng cần theo dõi
conf_threshold      = 0.6
tracking_class_ids  = [1, 2, 3, 5, 7]
video_path = 'data_ext/video_03.mp4'

# Thiết lập mô hình Yolov9
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = DetectMultiBackend(weights="weights/bestt.pt", device=device, fuse=True)
model   = AutoShape(model)  # Điều chỉnh tự động kích thước đầu vào cho mô hình

# Khởi tạo tracker DeepSort với khung hình tối đa cho các đối tượng theo dõi
tracker = DeepSort(max_age=50)

# Đọc file YAML để lấy danh sách tên các lớp đối tượng từ tập coco.yaml
with open("data_ext/dataa.yaml", 'r') as f:
    data_yaml = yaml.safe_load(f)
    class_names = data_yaml.get('names', [])
np.random.seed(42)
colors  = np.random.randint(0, 255, size=(len(class_names), 3))

# Khởi tạo biến đếm đối tượng và ngưỡng báo động
object_counters  = {class_id: 0 for class_id in tracking_class_ids}
alert_thresholds = {1: 10, 2: 5, 3: 15, 5: 20, 7: 25}
tracks  = []
counter = 0
vehicle_ids = set()

# Lấy cấu hình đếm đối tượng từ file count_config.py
# # left_right => count_config.py
# config = get_count_config('left_right')
# count_height    = config['height']
# count_left     = config['left']
# bottom_margin     = config['bottom_margin']
# top_margin    = config['top_margin']
# left_margin     = config['left_margin']

# top_bottom => count_config.py
config = get_count_config('top_bottom')
count_line_position = config['position']
count_height    = config['height']
left_margin     = config['left_margin']
right_margin    = config['right_margin']

# Định nghĩa kích thước chuẩn mà tất cả video sẽ được thay đổi để phù hợp
STANDARD_WIDTH = 640
STANDARD_HEIGHT = 480

# Thiết lập kích thước hiển thị duựa trên màn hình
monitor         = get_monitors()[0]
display_width   = monitor.width // 2
display_height  = monitor.height // 2

# Đọc video từ đường dẫn đã cung cấp
cap = cv2.VideoCapture(video_path)

# Chính xử lý từng frame của video
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Thay đổi kích thước khung hình để chuẩn hóa
        frame = cv2.resize(frame, (STANDARD_WIDTH, STANDARD_HEIGHT))

        # Nhận diện đối tượng trong khung hình
        results = model(frame)
        detect = []     # Danh sách để lưu các đối tượng được phát hiện
        for detect_object in results.pred[0]:
            label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
            class_id = int(label)
            if confidence < conf_threshold or class_id not in tracking_class_ids:
                continue

            # Chuẩn hóa bounding box và thêm vào danh sách nhận diện
            x1, y1, x2, y2 = map(int, bbox)
            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        # Cập nhật vị trí đối tượng thông qua DeepSort
        tracks = tracker.update_tracks(detect, frame=frame)

        # Vẽ và kiểm tra đối tượng qua khu vực định trước để đếm
        for track in tracks:
            if track.is_confirmed() and track.time_since_update <= 2 and track.get_det_class() in tracking_class_ids:
                track_id = track.track_id
                ltrb     = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)

                # Kiểm tra và đếm đối tượng qua khu vực định trước
                # # left_right = > count_config.py
                # if y1 <= count_left and y2 >= count_height:

                # top_bottom = > count_config.py
                if y1 <= count_line_position and y2 >= count_line_position + count_height:
                    if track_id not in vehicle_ids:
                        vehicle_ids.add(track_id)
                        counter += 1
                        object_counters[class_id] += 1

                # Vẽ khung đối tượng
                color = colors[class_id]
                B, G, R = map(int, color)
                label = f"{class_names[class_id]}-{track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 1)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 8, y1), (B, G, R), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Vẽ khu vực đếm và hiển thị số lượng đã đếm
        # # left_right => count_config.py
        # cv2.line(frame, (count_left, bottom_margin), (count_left, top_margin), (255, 0, 255), 2)

        # top_bottom = > count_config.py
        cv2.rectangle(frame, (left_margin, count_line_position),
                      (frame.shape[1] - right_margin, count_line_position + count_height), (255, 0, 255), 2)

        # # left_right = > count_config.py
        # cv2.putText(frame, f"Count All: {counter}", (left_margin, top_margin + 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


        # Hiển thị "Count All:" ngay trên thanh ngang đếm
        # top_bottom = > count_config.py
        cv2.putText(frame, f"Count All: {counter}", (left_margin + 5, count_line_position - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


        # Hiển thị số lượng mỗi loại đối tượng trên video
        for idx, class_id in enumerate(tracking_class_ids):
            color = tuple([int(c) for c in colors[class_id]])
            cv2.putText(frame, f"{class_names[class_id]} Count: {object_counters[class_id]}",
                        (5, 50 + 30 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Hiển thị khung hình đã được điều chỉnh kích thước
        frame_display = cv2.resize(frame, (display_width, display_height))
        cv2.imshow("Count Vehicle", frame_display)

        # Kết thúc khi nhấn 'q'
        if cv2.waitKey(50) == ord("q"):
            break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
