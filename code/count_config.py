# Định nghĩa khu vực đếm đối tượng
count_line_positions = {
    'top_bottom': {
        'position': 430,                        # Vị trí dọc của dòng đếm từ trên xuống
        'height': 3,                            # Độ dày của dòng đếm
        'left_margin': 10,                      # Giới hạn bênh trái của dòng đếm
        'right_margin': 5,                      # Giới hạn bênh phải của dòng đếm
    },
    'left_right': {
        'height': 3,
        'left': 600,
        'bottom_margin': 550,
        'top_margin': 10,
        'left_margin': 10
    }
}

# Hàm lấy cấu hình đếm của đối tượng
def get_count_config(vd):
    return count_line_positions[vd]