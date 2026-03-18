"""
Module vẽ kết quả nhận diện lên ảnh phiếu.
Đánh dấu các bubble đã phát hiện với màu sắc:
  - Xanh lá: đáp án đã chọn
  - Đỏ: lỗi (tô nhiều đáp án)
  - Vàng: cảnh báo
"""
import cv2
import numpy as np
from .extractor import get_bubble_rect, compute_fill_ratio


COLOR_FILLED = (0, 200, 0)      # Xanh lá - đã tô
COLOR_ERROR = (0, 0, 255)       # Đỏ - lỗi
COLOR_EMPTY = (200, 200, 200)   # Xám - trống
COLOR_TEXT = (0, 0, 200)        # Đỏ đậm cho text
COLOR_GRID = (180, 180, 180)    # Xám nhạt cho grid
MARK_RADIUS_DEFAULT = 1.0
TF_MARK_RADIUS_SCALE = 0.72


def draw_bubble_grid(img, region, fill_threshold, margin, labels=None, section_name=""):
    """Vẽ grid bubble lên ảnh và đánh dấu các ô đã tô."""
    h, w = img.shape[:2]
    rows = region["rows"]
    cols = region["cols"]

    # Chuyển sang binary để tính fill ratio
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    marked = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for r in range(rows):
        for c in range(cols):
            rect = get_bubble_rect(region, r, c, w, h, margin)
            x1, y1, x2, y2 = rect
            ratio = compute_fill_ratio(binary, rect)

            if ratio >= fill_threshold:
                # Đã tô - vẽ khung xanh lá
                cv2.rectangle(marked, (x1, y1), (x2, y2), COLOR_FILLED, 2)
                label = labels[c] if labels and c < len(labels) else str(c)
                cv2.putText(marked, label, (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_FILLED, 1)
            else:
                # Trống - vẽ khung xám nhạt
                cv2.rectangle(marked, (x1, y1), (x2, y2), COLOR_EMPTY, 1)

    return marked


def _mark_bubble(marked, overlay, rect, color, radius_scale=MARK_RADIUS_DEFAULT):
    """Đánh dấu bubble: vẽ overlay bán trong suốt + viền."""
    x1, y1, x2, y2 = rect
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx, ry = (x2 - x1) // 2, (y2 - y1) // 2
    r = max(int(min(rx, ry) * radius_scale), 1)
    cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.circle(marked, (cx, cy), r, color, 2)


def _get_tf_filled(tf_data, group, group_index, row_index):
    """Lấy đáp án TF theo schema mới (8 câu × a,b,c,d), có fallback schema cũ."""
    question_key = str(group.get("question", group_index + 1))
    question_data = tf_data.get(question_key, {})

    if isinstance(question_data, dict):
        sub_labels = ["a", "b", "c", "d"]
        if row_index < len(sub_labels):
            ans = question_data.get(sub_labels[row_index], [])
            return ans if isinstance(ans, list) else []
        return []

    questions = group.get("questions")
    start_q = group.get("start_question", 1)
    if questions and row_index < len(questions):
        legacy_key = str(questions[row_index])
    else:
        legacy_key = str(start_q + row_index)
    legacy_ans = tf_data.get(legacy_key, [])
    return legacy_ans if isinstance(legacy_ans, list) else []


def visualize_results(warped_img, results, config):
    """
    Vẽ toàn bộ kết quả nhận diện lên ảnh đã warp.
    Đánh dấu bubble đã tô bằng overlay bán trong suốt.
    """
    marked = warped_img.copy()
    overlay = marked.copy()
    h, w = marked.shape[:2]
    margin = config.get("bubble_margin", 0.15)
    fill_threshold = config.get("fill_threshold", 0.40)
    regions = config["regions"]

    gray = cv2.cvtColor(marked, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # --- SBD ---
    sbd_region = regions["sbd"]
    sbd_text = results.get("sbd", "")
    _draw_id_grid(marked, overlay, binary, sbd_region, margin, fill_threshold)

    # --- MĐT ---
    mdt_region = regions["mdt"]
    mdt_text = results.get("mdt", "")
    _draw_id_grid(marked, overlay, binary, mdt_region, margin, fill_threshold)

    # --- FC (Phần I) ---
    fc_data = results.get("fc", {})
    fc_labels = regions["fc"]["labels"]
    for group in regions["fc"]["groups"]:
        start_q = group["start_question"]
        for r in range(group["rows"]):
            q_num = str(start_q + r)
            filled = fc_data.get(q_num, [])
            for c in range(group["cols"]):
                rect = get_bubble_rect(group, r, c, w, h, margin)
                x1, y1, x2, y2 = rect
                if c in filled:
                    color = COLOR_ERROR if len(filled) > 1 else COLOR_FILLED
                    _mark_bubble(marked, overlay, rect, color)
                    label = fc_labels[c] if c < len(fc_labels) else str(c)
                    cv2.putText(marked, label, (x1 + 2, y2 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    cv2.rectangle(marked, (x1, y1), (x2, y2), COLOR_EMPTY, 1)

    # --- TF (Phần II) ---
    tf_data = results.get("tf", {})
    tf_labels = regions["tf"]["labels"]
    tf_margin = config.get("tf_bubble_margin", margin)
    for group_index, group in enumerate(regions["tf"]["groups"], start=1):
        for r in range(group["rows"]):
            filled = _get_tf_filled(tf_data, group, group_index, r)
            for c in range(group["cols"]):
                rect = get_bubble_rect(group, r, c, w, h, tf_margin)
                x1, y1, x2, y2 = rect
                if c in filled:
                    color = COLOR_ERROR if len(filled) > 1 else COLOR_FILLED
                    _mark_bubble(
                        marked, overlay, rect, color, radius_scale=TF_MARK_RADIUS_SCALE
                    )
                    label = tf_labels[c] if c < len(tf_labels) else str(c)
                    cv2.putText(marked, label, (x1 + 2, y2 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    cv2.rectangle(marked, (x1, y1), (x2, y2), COLOR_EMPTY, 1)

    # --- DG (Phần III) ---
    dg_data = results.get("dg", {})
    dg_labels = regions["dg"]["labels"]
    dg_rows = regions["dg"]["rows"]
    for group in regions["dg"]["groups"]:
        region = {
            "x1": group["x1"], "y1": group["y1"],
            "x2": group["x2"], "y2": group["y2"],
            "rows": dg_rows, "cols": group["cols"]
        }
        for r in range(dg_rows):
            for c in range(group["cols"]):
                rect = get_bubble_rect(region, r, c, w, h, margin)
                x1_, y1_, x2_, y2_ = rect
                ratio = compute_fill_ratio(binary, rect)
                if ratio >= fill_threshold:
                    _mark_bubble(marked, overlay, rect, COLOR_FILLED)
                    label = dg_labels[r] if r < len(dg_labels) else str(r)
                    cv2.putText(marked, label, (x1_ + 2, y2_ - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_FILLED, 1)
                else:
                    cv2.rectangle(marked, (x1_, y1_), (x2_, y2_), COLOR_EMPTY, 1)

    # Blend overlay (semi-transparent fill)
    cv2.addWeighted(overlay, 0.3, marked, 0.7, 0, marked)

    # --- Ghi SBD, MĐT lên góc ảnh ---
    cv2.putText(marked, f"SBD: {sbd_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
    cv2.putText(marked, f"MDT: {mdt_text}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)

    return marked


def _draw_id_grid(marked, overlay, binary, region, margin, fill_threshold):
    """Vẽ grid SBD/MĐT với overlay bán trong suốt."""
    h, w = marked.shape[:2]
    for col in range(region["cols"]):
        for row in range(region["rows"]):
            rect = get_bubble_rect(region, row, col, w, h, margin)
            x1, y1, x2, y2 = rect
            ratio = compute_fill_ratio(binary, rect)
            if ratio >= fill_threshold:
                _mark_bubble(marked, overlay, rect, COLOR_FILLED)
                digit = region["labels"][row] if row < len(region["labels"]) else "?"
                cv2.putText(marked, digit, (x1 + 2, y2 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_FILLED, 1)
            else:
                cv2.rectangle(marked, (x1, y1), (x2, y2), COLOR_EMPTY, 1)


def create_debug_image(warped, binary, config):
    """
    Tạo ảnh debug hiển thị tất cả các vùng và grid.
    Hữu ích cho việc calibrate tọa độ.
    """
    debug = warped.copy()
    h, w = debug.shape[:2]
    regions = config["regions"]

    # Vẽ viền các vùng
    for name, color in [("sbd", (255, 0, 0)), ("mdt", (0, 255, 0))]:
        r = regions[name]
        x1, y1 = int(r["x1"]*w), int(r["y1"]*h)
        x2, y2 = int(r["x2"]*w), int(r["y2"]*h)
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        cv2.putText(debug, name.upper(), (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for section, color in [("fc", (0, 0, 255)), ("tf", (255, 128, 0))]:
        for i, group in enumerate(regions[section]["groups"]):
            x1, y1 = int(group["x1"]*w), int(group["y1"]*h)
            x2, y2 = int(group["x2"]*w), int(group["y2"]*h)
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug, f"{section.upper()}_{i+1}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for group in regions["dg"]["groups"]:
        x1, y1 = int(group["x1"]*w), int(group["y1"]*h)
        x2, y2 = int(group["x2"]*w), int(group["y2"]*h)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (128, 0, 255), 2)
        cv2.putText(debug, f"DG_{group['question']}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 255), 2)

    return debug
