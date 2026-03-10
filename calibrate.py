"""
Công cụ Calibrate - Hiệu chỉnh tọa độ vùng trên phiếu trắc nghiệm.

Sử dụng:
    python calibrate.py <đường_dẫn_ảnh_mẫu>

Chức năng:
    1. Tải ảnh mẫu, tìm góc phiếu, warp về kích thước chuẩn
    2. Hiển thị ảnh đã warp với grid overlay
    3. Cho phép click để xem tọa độ % tại vị trí click
    4. Lưu config sau khi chỉnh sửa

Hướng dẫn:
    - Chạy script với ảnh mẫu (template hoặc phiếu thật)
    - Xem ảnh warp, di chuột để xem tọa độ %
    - Click chuột trái để in tọa độ % ra console
    - Ghi lại các tọa độ và cập nhật vào config.json
    - Nhấn 'q' để thoát, 'd' để bật/tắt debug grid, 's' để lưu ảnh debug
"""
import sys
import os
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.preprocessor import load_image, preprocess, preprocess_to_binary
from core.detector import find_corners, warp_perspective
from core.visualizer import create_debug_image


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Calibrator:
    def __init__(self, image_path, config_path="config.json"):
        self.config = load_config(config_path)
        self.config_path = config_path
        self.image_path = image_path
        self.show_grid = True
        self.click_points = []

        # Load & process
        self.img = load_image(image_path)
        binary = preprocess_to_binary(self.img, self.config)
        corners, method = find_corners(self.img, binary, self.config)
        print(f"Phương pháp phát hiện góc: {method}")

        warp_w = self.config.get("warp_width", 1800)
        warp_h = self.config.get("warp_height", 2500)
        self.warped = warp_perspective(self.img, corners, warp_w, warp_h)
        self.warp_h, self.warp_w = self.warped.shape[:2]

        print(f"Kích thước warp: {self.warp_w} x {self.warp_h}")
        print(f"\nHướng dẫn:")
        print(f"  - Click chuột trái: In tọa độ %")
        print(f"  - Phím 'd': Bật/tắt debug grid")
        print(f"  - Phím 's': Lưu ảnh debug")
        print(f"  - Phím 'q' / ESC: Thoát")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            x_pct = x / self.warp_w
            y_pct = y / self.warp_h
            display = self.get_display()
            info = f"({x_pct:.3f}, {y_pct:.3f}) | px({x}, {y})"
            cv2.putText(display, info, (10, self.warp_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Vẽ crosshair
            cv2.line(display, (x, 0), (x, self.warp_h), (0, 255, 0), 1)
            cv2.line(display, (0, y), (self.warp_w, y), (0, 255, 0), 1)
            cv2.imshow("Calibrate", display)

        elif event == cv2.EVENT_LBUTTONDOWN:
            x_pct = x / self.warp_w
            y_pct = y / self.warp_h
            self.click_points.append((x_pct, y_pct))
            point_num = len(self.click_points)
            print(f"  Điểm {point_num}: x={x_pct:.4f}, y={y_pct:.4f}  (px: {x}, {y})")

            if point_num % 2 == 0:
                p1 = self.click_points[-2]
                p2 = self.click_points[-1]
                print(f"  → Vùng: x1={p1[0]:.4f}, y1={p1[1]:.4f}, "
                      f"x2={p2[0]:.4f}, y2={p2[1]:.4f}")

    def get_display(self):
        if self.show_grid:
            warped_binary = preprocess_to_binary(self.warped, self.config)
            return create_debug_image(self.warped, warped_binary, self.config)
        return self.warped.copy()

    def run(self):
        cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibrate", 900, 1250)
        cv2.setMouseCallback("Calibrate", self.mouse_callback)

        display = self.get_display()
        cv2.imshow("Calibrate", display)

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('d'):
                self.show_grid = not self.show_grid
                display = self.get_display()
                cv2.imshow("Calibrate", display)
                print(f"Debug grid: {'ON' if self.show_grid else 'OFF'}")
            elif key == ord('s'):
                out_path = "calibrate_debug.jpg"
                cv2.imwrite(out_path, self.get_display())
                print(f"Ảnh debug đã lưu: {out_path}")

        cv2.destroyAllWindows()

        if self.click_points:
            print(f"\nTất cả các điểm đã click ({len(self.click_points)}):")
            for i, (xp, yp) in enumerate(self.click_points, 1):
                print(f"  {i}. x={xp:.4f}, y={yp:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: python calibrate.py <đường_dẫn_ảnh>")
        print("Ví dụ:   python calibrate.py anh/sample_image/PhieuTracNghiepTHPT2025.png")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Không tìm thấy file: {image_path}")
        sys.exit(1)

    cal = Calibrator(image_path)
    cal.run()
