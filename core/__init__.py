"""
Core package - Pipeline xử lý chính cho hệ thống chấm phiếu trắc nghiệm THPT 2025.
"""
import cv2
import json
import os
import numpy as np

from .preprocessor import preprocess, preprocess_to_binary, load_image
from .detector import find_corners, warp_perspective, deskew, align_to_template
from .extractor import extract_all
from .visualizer import visualize_results, create_debug_image


def load_config(config_path="config.json"):
    """Đọc file cấu hình JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_image(image_path, config, debug=False):
    """
    Pipeline xử lý chính cho 1 ảnh phiếu trắc nghiệm.

    Args:
        image_path: Đường dẫn ảnh đầu vào
        config: Dict cấu hình (từ config.json)
        debug: Nếu True, trả về thêm ảnh debug

    Returns:
        (output_dict, marked_image, debug_image_or_None)
    """
    # 1. Đọc ảnh
    try:
        img = load_image(image_path)
    except ValueError as e:
        return {"err": [str(e)], "res": None}, None, None

    # 2. Tiền xử lý
    gray = preprocess(img, config)
    binary = preprocess_to_binary(img, config)

    # 3. Phát hiện phiếu và warp
    corners, method = find_corners(img, binary, config)
    warp_w = config.get("warp_width", 1800)
    warp_h = config.get("warp_height", 2500)

    warped = warp_perspective(img, corners, warp_w, warp_h)

    # 3.5. Deskew - chỉnh nghiêng sau khi warp
    warped, skew_angle = deskew(warped)
    if abs(skew_angle) > 0.3:
        warped = cv2.resize(warped, (warp_w, warp_h), interpolation=cv2.INTER_CUBIC)

    # 3.6. Căn chỉnh theo template
    warped, aligned = align_to_template(warped, config)
    if aligned:
        method += "+aligned"

    # Tạo binary Otsu cho extraction (độ tương phản tốt hơn adaptive)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe_e = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_e = clahe_e.apply(warped_gray)
    blurred_e = cv2.GaussianBlur(enhanced_e, (5, 5), 0)
    _, warped_binary = cv2.threshold(
        blurred_e, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4. Trích xuất đáp án
    results, errors, warnings = extract_all(warped_binary, config, warped_gray)

    # 5. Vẽ kết quả lên ảnh
    marked = visualize_results(warped, results, config)

    # 6. Ảnh debug (nếu cần)
    debug_img = None
    if debug:
        debug_img = create_debug_image(warped, warped_binary, config)

    # 7. Tạo output theo mẫu
    output = {
        "org": image_path,
        "out": "",
        "warn": "; ".join(warnings) if warnings else "",
        "err": errors,
        "res": {
            "fc": results["fc"],
            "tf": results["tf"],
            "dg": results["dg"]
        },
        "sbd": results["sbd"],
        "mdt": results["mdt"],
        "_detection_method": method,
        "_skew_angle": round(skew_angle, 2)
    }

    return output, marked, debug_img


def format_results(output, config):
    """
    Định dạng kết quả trích xuất thành text dễ đọc.
    Liệt kê đáp án theo 3 phần: FC, TF, DG.
    """
    lines = []
    sbd = output.get("sbd", "?")
    mdt = output.get("mdt", "?")
    lines.append("=" * 55)
    lines.append(f"  SBD: {sbd}    MĐT: {mdt}")
    lines.append("=" * 55)

    res = output.get("res", {})
    if not res:
        lines.append("  Không trích xuất được dữ liệu!")
        return "\n".join(lines)

    fc_labels = config["regions"]["fc"]["labels"]
    tf_labels = config["regions"]["tf"]["labels"]

    # --- PHẦN I ---
    fc = res.get("fc", {})
    lines.append("")
    lines.append("  PHẦN I - Trắc nghiệm nhiều lựa chọn (40 câu):")
    row = "  "
    for i in range(1, 41):
        ans = fc.get(str(i), [])
        if ans:
            txt = ",".join(fc_labels[a] for a in ans if a < len(fc_labels))
            marker = "●" if len(ans) == 1 else "✗"
            row += f"{i:>2}.{txt:<2}{marker} "
        else:
            row += f"{i:>2}._  "
        if i % 8 == 0:
            lines.append(row)
            row = "  "
    if row.strip():
        lines.append(row)
    fc_count = sum(1 for i in range(1, 41) if fc.get(str(i), []))
    lines.append(f"  => Đã trả lời: {fc_count}/40")

    # --- PHẦN II ---
    tf = res.get("tf", {})
    lines.append("")
    lines.append("  PHẦN II - Trắc nghiệm Đúng/Sai (8 câu × 4 ý):")
    tf_groups = config["regions"]["tf"]["groups"]
    sub_labels = ["a", "b", "c", "d"]
    for g_idx, group in enumerate(tf_groups):
        cau_num = g_idx + 1
        questions = group.get("questions", [])
        parts = []
        for r_idx, q_num in enumerate(questions):
            ans = tf.get(str(q_num), [])
            sub = sub_labels[r_idx] if r_idx < len(sub_labels) else str(r_idx)
            if ans:
                txt = ",".join(tf_labels[a] for a in ans if a < len(tf_labels))
            else:
                txt = "_"
            parts.append(f"{sub}={txt}")
        lines.append(f"  Câu {cau_num}: " + " | ".join(parts))
    tf_count = sum(1 for i in range(1, 33) if tf.get(str(i), []))
    lines.append(f"  => Đã trả lời: {tf_count}/32")

    # --- PHẦN III ---
    dg = res.get("dg", {})
    lines.append("")
    lines.append("  PHẦN III - Trả lời ngắn (6 câu):")
    for i in range(1, 7):
        ans = dg.get(str(i), "")
        lines.append(f"  Câu {i}: {ans if ans else '(trống)'}")

    lines.append("=" * 55)

    warn = output.get("warn", "")
    if warn:
        lines.append(f"  Cảnh báo: {warn}")
        lines.append("")

    return "\n".join(lines)


def process_batch(image_paths, config, output_dir="results", debug=False):
    """
    Xử lý hàng loạt ảnh.

    Args:
        image_paths: Danh sách đường dẫn ảnh
        config: Dict cấu hình
        output_dir: Thư mục lưu kết quả
        debug: Bật chế độ debug

    Returns:
        list of output_dicts
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for i, path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Đang xử lý: {os.path.basename(path)}")

        output, marked, debug_img = process_image(path, config, debug)

        # Lưu ảnh đã đánh dấu
        if marked is not None:
            base_name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"marked_{base_name}.jpg")
            cv2.imwrite(out_path, marked)
            output["out"] = out_path

            if debug and debug_img is not None:
                debug_dir = os.path.join(output_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                dbg_path = os.path.join(debug_dir, f"debug_{base_name}.jpg")
                cv2.imwrite(dbg_path, debug_img)

        all_results.append(output)

    # Lưu JSON tổng hợp
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nHoàn tất! Kết quả lưu tại: {output_dir}")
    for r in all_results:
        print(format_results(r, config))
    return all_results
