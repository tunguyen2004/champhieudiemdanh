"""
Script xử lý hàng loạt ảnh phiếu trắc nghiệm.

Sử dụng:
    python process_batch.py <thư_mục_ảnh> [--output <thư_mục_output>] [--debug]

Ví dụ:
    python process_batch.py anh/test_imga
    python process_batch.py anh/test_imga --output results --debug
"""
import os
import sys
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import load_config, process_batch


def main():
    parser = argparse.ArgumentParser(description="Chấm phiếu trắc nghiệm THPT 2025 - Xử lý hàng loạt")
    parser.add_argument("input_dir", help="Thư mục chứa ảnh phiếu")
    parser.add_argument("--output", "-o", default="results", help="Thư mục xuất kết quả (mặc định: results)")
    parser.add_argument("--config", "-c", default="config.json", help="File cấu hình (mặc định: config.json)")
    parser.add_argument("--debug", "-d", action="store_true", help="Bật chế độ debug")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Không tìm thấy thư mục: {args.input_dir}")
        sys.exit(1)

    # Tìm tất cả file ảnh
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))

    # Loại trùng lặp và sắp xếp
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"Không tìm thấy file ảnh trong: {args.input_dir}")
        sys.exit(1)

    print(f"Tìm thấy {len(image_paths)} ảnh")
    print(f"Output: {args.output}")
    print(f"Config: {args.config}")
    print(f"Debug: {'ON' if args.debug else 'OFF'}")
    print("-" * 50)

    config = load_config(args.config)
    results = process_batch(image_paths, config, args.output, args.debug)

    # Thống kê
    total = len(results)
    success = sum(1 for r in results if r.get("res") is not None)
    errors = total - success
    warnings_count = sum(1 for r in results if r.get("warn"))

    print(f"\n{'='*50}")
    print(f"THỐNG KÊ:")
    print(f"  Tổng số phiếu:    {total}")
    print(f"  Thành công:        {success}")
    print(f"  Lỗi:              {errors}")
    print(f"  Có cảnh báo:      {warnings_count}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
