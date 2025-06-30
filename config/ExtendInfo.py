import numpy as np

def create_extend_info():
    # === Plane 1: Plane toàn số 0 (Plane of Zeros) ===
    # Cung cấp một điểm tham chiếu "tắt" hoặc "trung tính".
    zero_plane = np.zeros((8, 8), dtype=np.float32)

    # === Plane 2: Plane toàn số 1 (Plane of Ones) ===
    # Hoạt động như một "bias" (thiên vị), cho phép mạng học các đặc trưng
    # không phụ thuộc vào vị trí quân cờ.
    one_plane = np.ones((8, 8), dtype=np.float32)

    # === Planes 3-6: Các plane mã hóa tọa độ (Coordinate Planes) ===
    # Giúp mạng "hiểu" về hình học bàn cờ ngay từ đầu.

    # Plane 3: Mã hóa hàng tuyến tính (Linear Rank Encoding)
    # Giá trị tăng dần từ 0.0 (hàng 1) đến 1.0 (hàng 8).
    # Giúp mạng nhận biết khái niệm "tiến lên" hoặc vị trí tương đối trên bàn cờ.
    rank_plane_linear = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        # Chuẩn hóa giá trị về khoảng [0, 1]
        rank_plane_linear[i, :] = i / 7.0

        # Plane 4: Mã hóa cột tuyến tính (Linear File Encoding)
    # Tương tự, giá trị tăng từ 0.0 (cột a) đến 1.0 (cột h).
    file_plane_linear = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        file_plane_linear[:, i] = i / 7.0

    # Plane 5: Mã hóa hàng đối xứng (Symmetric Rank Encoding)
    # Giá trị cao ở hai biên và thấp ở trung tâm.
    # Giúp mạng nhận biết các khái niệm "cánh" vs "trung tâm".
    rank_plane_symmetric = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        # Hàm abs(i - 3.5) tạo ra chuỗi đối xứng: 3.5, 2.5, 1.5, 0.5, 0.5, ...
        # Chia cho 3.5 để chuẩn hóa về khoảng [~0.14, 1.0]
        rank_plane_symmetric[i, :] = abs(i - 3.5) / 3.5

    # Plane 6: Mã hóa cột đối xứng (Symmetric File Encoding)
    # Tương tự cho cột.
    file_plane_symmetric = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        file_plane_symmetric[:, i] = abs(i - 3.5) / 3.5

    # Ghép 6 plane lại thành một mảng 3D duy nhất
    # Thứ tự có thể thay đổi, nhưng đây là một cách sắp xếp hợp lý.
    all_planes = np.array([
        zero_plane,
        one_plane,
        rank_plane_linear,
        file_plane_linear,
        rank_plane_symmetric,
        file_plane_symmetric
    ])

    return all_planes