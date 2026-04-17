"""
Module trích xuất đáp án từ phiếu trắc nghiệm đã warp.
Hỗ trợ 3 loại: FC (nhiều lựa chọn), TF (đúng/sai), DG (trả lời ngắn - grid số).

Edge Cases được xử lý:
  - Case 8: Tô rồi xóa (vết nhòe xám) → detect_erased_mark()
  - Case 9: Bút chì nhạt (tương phản thấp) → detect_pencil_mark()
  - Case 10: Vết bẩn/noise (dấu vân tay, mực lem) → is_valid_mark()
  - Tổng hợp: robust_bubble_detection() kết hợp multi-stage pipeline
"""
import cv2
import numpy as np
import copy
from .mark_scorer import score_mark, choose_mark

_TEMPLATE_GRAY_CACHE = {}


def auto_calibrate_grid_y(grayscale, region, img_h, img_w, margin=0.15):
    """
    Auto-calibrate vị trí dọc (y) của grid bubble.

    Thử nhiều y-offset (±20 pixel), mỗi offset tính "detection confidence"
    = tổng chênh lệch (darkest - 2nd darkest) trên tất cả cột.
    Offset nào cho confidence cao nhất = grid đặt đúng vị trí nhất.

    Returns:
        dict: Bản sao của region với y1, y2 đã điều chỉnh
    """
    if grayscale is None:
        return region

    config_y1 = region["y1"]
    config_y2 = region["y2"]
    grid_height = config_y2 - config_y1
    rows = region["rows"]
    cols = region["cols"]

    best_y1 = config_y1
    best_score = 0

    # Thử offset từ -20px đến +20px (bước 2px)
    for offset_px in range(-20, 21, 2):
        dy = offset_px / img_h
        test_y1 = config_y1 + dy
        test_y2 = test_y1 + grid_height

        if test_y1 < 0 or test_y2 > 1.0:
            continue

        # Tạo region tạm với y đã dịch
        test_region = dict(region)
        test_region["y1"] = test_y1
        test_region["y2"] = test_y2

        # Tính confidence: tổng gap giữa darkest và 2nd darkest mỗi cột
        total_contrast = 0
        for col in range(cols):
            intensities = []
            for row in range(rows):
                rect = get_bubble_rect(test_region, row, col, img_w, img_h, margin)
                x1, y1, x2, y2 = rect
                roi = grayscale[y1:y2, x1:x2]
                mean_val = float(np.mean(roi)) if roi.size > 0 else 255.0
                intensities.append(mean_val)

            if len(intensities) >= 2:
                sorted_i = sorted(intensities)
                # Gap giữa darkest và 2nd darkest (càng lớn = phân biệt rõ hơn)
                total_contrast += sorted_i[1] - sorted_i[0]

        if total_contrast > best_score:
            best_score = total_contrast
            best_y1 = test_y1

    # Trả về region đã calibrate
    calibrated = dict(region)
    calibrated["y1"] = best_y1
    calibrated["y2"] = best_y1 + grid_height
    return calibrated


def get_bubble_rect(region, row, col, img_w, img_h, margin=0.15):
    """Tính tọa độ pixel của 1 bubble dựa trên vị trí (row, col) trong region."""
    x1 = int(region["x1"] * img_w)
    y1 = int(region["y1"] * img_h)
    x2 = int(region["x2"] * img_w)
    y2 = int(region["y2"] * img_h)

    rows = region["rows"]
    cols = region["cols"]

    cell_w = (x2 - x1) / cols
    cell_h = (y2 - y1) / rows

    bx1 = int(x1 + col * cell_w + cell_w * margin)
    by1 = int(y1 + row * cell_h + cell_h * margin)
    bx2 = int(x1 + (col + 1) * cell_w - cell_w * margin)
    by2 = int(y1 + (row + 1) * cell_h - cell_h * margin)

    return max(bx1, 0), max(by1, 0), min(bx2, img_w), min(by2, img_h)


def compute_fill_ratio(binary, rect):
    """Tính tỉ lệ pixel trắng (đã tô) trong vùng bubble."""
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = binary[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return cv2.countNonZero(roi) / roi.size


def _clamp_rect(rect, img_w, img_h):
    """Clamp rect vào biên ảnh, bảo toàn tính hợp lệ."""
    x1, y1, x2, y2 = rect
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(x1 + 1, min(int(x2), img_w))
    y2 = max(y1 + 1, min(int(y2), img_h))
    return x1, y1, x2, y2


def _refine_bubble_rect(
    binary, grayscale, rect, max_shift_ratio=0.20, shrink_ratio=0.92
):
    """
    Tinh chỉnh tâm bubble cục bộ để giảm lệch ROI.
    Dựa trên centroid của vùng tối/đã tô trong chính ô bubble.
    """
    x1, y1, x2, y2 = rect
    if x2 - x1 < 6 or y2 - y1 < 6:
        return rect

    if grayscale is not None:
        img_h, img_w = grayscale.shape[:2]
        roi_gray = grayscale[y1:y2, x1:x2]
        if roi_gray.size == 0:
            return rect
        roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
        min_side = max(min(roi_gray.shape[:2]), 5)
        block = min(21, min_side if min_side % 2 == 1 else min_side - 1)
        block = max(block, 5)
        local = cv2.adaptiveThreshold(
            roi_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            6
        )
    else:
        img_h, img_w = binary.shape[:2]
        roi_bin = binary[y1:y2, x1:x2]
        if roi_bin.size == 0:
            return rect
        local = roi_bin.copy()

    if binary is not None:
        roi_bin = binary[y1:y2, x1:x2]
        if roi_bin.size > 0:
            local = cv2.bitwise_or(local, roi_bin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    local = cv2.morphologyEx(local, cv2.MORPH_OPEN, kernel, iterations=1)
    if cv2.countNonZero(local) == 0:
        return rect

    moments = cv2.moments(local)
    if moments["m00"] <= 1e-6:
        return rect

    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])
    roi_w = float(x2 - x1)
    roi_h = float(y2 - y1)
    base_cx = (roi_w - 1.0) / 2.0
    base_cy = (roi_h - 1.0) / 2.0

    max_shift_x = roi_w * max_shift_ratio
    max_shift_y = roi_h * max_shift_ratio
    shift_x = float(np.clip(cx - base_cx, -max_shift_x, max_shift_x))
    shift_y = float(np.clip(cy - base_cy, -max_shift_y, max_shift_y))

    new_w = max(int(round(roi_w * shrink_ratio)), 4)
    new_h = max(int(round(roi_h * shrink_ratio)), 4)
    new_cx = x1 + base_cx + shift_x
    new_cy = y1 + base_cy + shift_y
    nx1 = int(round(new_cx - new_w / 2.0))
    ny1 = int(round(new_cy - new_h / 2.0))
    nx2 = nx1 + new_w
    ny2 = ny1 + new_h

    return _clamp_rect((nx1, ny1, nx2, ny2), img_w, img_h)


def _bubble_core_metrics(binary, grayscale, rect, core_ratio=0.62):
    """
    Đặc trưng bubble ở vùng lõi + vành để giảm nhiễu viền/in.
    Returns:
      core_fill, ring_fill, mean_intensity, full_fill
    """
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 255.0, 0.0

    roi_bin = binary[y1:y2, x1:x2]
    if roi_bin.size == 0:
        return 0.0, 0.0, 255.0, 0.0

    roi_h, roi_w = roi_bin.shape[:2]
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    center = (roi_w // 2, roi_h // 2)
    axis_x = max(int((roi_w * core_ratio) / 2), 1)
    axis_y = max(int((roi_h * core_ratio) / 2), 1)
    cv2.ellipse(mask, center, (axis_x, axis_y), 0, 0, 360, 255, -1)

    core_pixels = cv2.countNonZero(mask)
    if core_pixels == 0:
        return 0.0, 0.0, 255.0, 0.0

    full_pixels = roi_bin.size
    full_nonzero = cv2.countNonZero(roi_bin)
    core_nonzero = cv2.countNonZero(cv2.bitwise_and(roi_bin, roi_bin, mask=mask))
    ring_pixels = max(full_pixels - core_pixels, 1)
    ring_nonzero = max(full_nonzero - core_nonzero, 0)

    core_fill = core_nonzero / core_pixels
    ring_fill = ring_nonzero / ring_pixels
    full_fill = full_nonzero / full_pixels

    if grayscale is None:
        mean_intensity = 255.0 * (1.0 - core_fill)
    else:
        roi_gray = grayscale[y1:y2, x1:x2]
        mean_intensity = cv2.mean(roi_gray, mask=mask)[0] if roi_gray.size > 0 else 255.0

    return float(core_fill), float(ring_fill), float(mean_intensity), float(full_fill)


def _score_bubble_mark(binary, grayscale, rect, core_ratio=0.62):
    """
    Unified bubble scoring using mark_scorer + core/ring metrics.
    """
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return {
            "score": 0.0,
            "fill_ratio": 0.0,
            "core_fill": 0.0,
            "ring_noise": 0.0,
            "ring_fill": 0.0,
            "darkness": 0.0,
            "mean_intensity": 255.0,
            "full_fill": 0.0,
        }

    roi_bin = binary[y1:y2, x1:x2]
    roi_gray = grayscale[y1:y2, x1:x2] if grayscale is not None else None
    if roi_bin.size == 0:
        return {
            "score": 0.0,
            "fill_ratio": 0.0,
            "core_fill": 0.0,
            "ring_noise": 0.0,
            "ring_fill": 0.0,
            "darkness": 0.0,
            "mean_intensity": 255.0,
            "full_fill": 0.0,
        }

    score_payload = score_mark(roi_bin, roi_gray)
    core_fill, ring_fill, mean_intensity, full_fill = _bubble_core_metrics(
        binary,
        grayscale,
        rect,
        core_ratio=core_ratio
    )

    score_payload["core_fill"] = float(core_fill)
    score_payload["ring_fill"] = float(ring_fill)
    score_payload["mean_intensity"] = float(mean_intensity)
    score_payload["full_fill"] = float(full_fill)
    return score_payload


# ============================================================
# CASE 8: Phát hiện vết xóa (tô rồi xóa - còn vết nhòe xám)
# ============================================================
def detect_erased_mark(grayscale, rect):
    """
    Phát hiện vết xóa bằng phân tích histogram vùng xám.

    Nguyên lý:
      - Ô trống: hầu hết pixel sáng (200-255)
      - Ô tô đầy: hầu hết pixel tối (0-100)
      - Ô xóa: nhiều pixel xám nhạt (100-180) — vết tẩy/xóa

    Phân tích histogram 3 vùng:
      - dark_zone (0-100): pixel tối = mực tô thật
      - gray_zone (100-180): pixel xám = VẾT XÓA
      - light_zone (180-255): pixel sáng = nền trắng

    Returns:
        "erased": Có vết xóa (nên bỏ qua - coi như không tô)
        "filled": Tô thật
        "empty": Không tô
    """
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return "empty"
    roi = grayscale[y1:y2, x1:x2]
    if roi.size == 0:
        return "empty"

    total = roi.size
    # Đếm pixel trong 3 vùng intensity
    dark_count = np.sum(roi < 100)       # Mực tô thật sự
    gray_count = np.sum((roi >= 100) & (roi < 180))  # Vết xóa/nhòe
    light_count = np.sum(roi >= 180)     # Nền trắng

    dark_ratio = dark_count / total
    gray_ratio = gray_count / total

    # Nếu > 35% pixel nằm trong gray_zone VÀ rất ít pixel dark → vết xóa
    if gray_ratio > 0.35 and dark_ratio < 0.10:
        return "erased"

    # Nếu > 30% pixel tối → tô thật
    if dark_ratio > 0.30:
        return "filled"

    return "empty"


# ============================================================
# CASE 9: Phát hiện bút chì nhạt (tương phản thấp)
# ============================================================
def detect_pencil_mark(grayscale, rect):
    """
    Phát hiện nét bút chì nhạt bằng 2 phương pháp kết hợp.

    Phương pháp A - Adaptive threshold:
      Dùng cv2.adaptiveThreshold với window nhỏ (blockSize=11)
      để bắt chi tiết có tương phản thấp mà Otsu bỏ sót.

    Phương pháp B - Local contrast:
      So sánh mean intensity của ô bubble với vùng xung quanh.
      Bút chì: ô tối hơn xung quanh 12+ đơn vị intensity.

    Returns:
        True nếu phát hiện bút chì nhạt, False nếu không
    """
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return False
    roi = grayscale[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    # Phương pháp A: Adaptive threshold bắt bút chì
    binary_adaptive = cv2.adaptiveThreshold(
        roi, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=8
    )
    adaptive_fill = cv2.countNonZero(binary_adaptive) / roi.size

    # Phương pháp B: So sánh mean intensity với vùng xung quanh
    mean_cell = np.mean(roi)
    h_img, w_img = grayscale.shape[:2]
    pad = 15
    px1 = max(0, x1 - pad)
    py1 = max(0, y1 - pad)
    px2 = min(w_img, x2 + pad)
    py2 = min(h_img, y2 + pad)
    padded = grayscale[py1:py2, px1:px2]
    mean_surrounding = np.mean(padded)
    contrast_diff = mean_surrounding - mean_cell

    # Bút chì nhạt: adaptive fill >= 0.25 VÀ contrast diff > 12
    return adaptive_fill >= 0.25 and contrast_diff > 12


# ============================================================
# CASE 10: Lọc vết bẩn / noise (dấu vân tay, mực lem)
# ============================================================
def is_valid_mark(binary, rect):
    """
    Kiểm tra đánh dấu có hợp lệ không (loại bỏ noise/vết bẩn).

    Dùng 2 tiêu chí:
      1. Shape filtering: Contour lớn nhất phải có circularity > 0.25
         (tô đầy ≈ 1.0, X ≈ 0.4-0.6, noise rời rạc ≈ < 0.2)
      2. Spatial concentration: Chia ô thành grid 3×3, đếm số cell
         có pixel đen. Tô thật → ≥ 3 cells đen. Noise → rải rác 1-2 cells.

    Returns:
        True nếu là đánh dấu hợp lệ, False nếu là noise
    """
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return False
    roi = binary[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return False

    total_fill = cv2.countNonZero(roi) / roi.size
    if total_fill < 0.05:
        return False

    # --- Tiêu chí 1: Shape filtering (contour circularity) ---
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        return False

    roi_area = roi.shape[0] * roi.shape[1]
    area_ratio = area / roi_area

    # Contour quá nhỏ hoặc hình dạng quá bất thường → noise
    if area_ratio < 0.05 or circularity < 0.25:
        return False

    # --- Tiêu chí 2: Spatial concentration (grid 3x3) ---
    rh, rw = roi.shape
    cell_h, cell_w = max(rh // 3, 1), max(rw // 3, 1)
    filled_cells = 0
    for gr in range(3):
        for gc in range(3):
            cell = roi[gr * cell_h:(gr + 1) * cell_h, gc * cell_w:(gc + 1) * cell_w]
            if cell.size > 0 and (cv2.countNonZero(cell) / cell.size) > 0.15:
                filled_cells += 1

    # Đánh dấu hợp lệ phải tập trung ≥ 3 cells
    return filled_cells >= 3


# ============================================================
# TỔNG HỢP: Multi-stage robust bubble detection
# ============================================================
def robust_bubble_detection(binary, grayscale, rect, config):
    """
    Pipeline phát hiện bubble đa giai đoạn, xử lý tất cả edge cases.

    Quy trình:
      Stage 1: Binary fill ratio (nhanh) — dùng fill_threshold từ config
        - fill >= fill_threshold (0.30) → "filled" (tô đủ đậm, chấp nhận)
        - fill < 0.05 → "empty" (trống rõ ràng)
        - 0.05 ~ fill_threshold → Vùng nghi ngờ, cần kiểm tra thêm

      Stage 2 (chỉ cho vùng nghi ngờ): Lọc noise (Case 10)
        - Kiểm tra shape + spatial concentration
        - Nếu không hợp lệ → "noise" (vết bẩn/vân tay)

      Stage 3: Phát hiện vết xóa (Case 8)
        - Phân tích histogram vùng xám
        - gray_zone > 30% + dark < 10% → "erased" (tô rồi xóa)

      Stage 4: Phát hiện bút chì (Case 9)
        - Adaptive threshold + local contrast
        - Nếu nhận diện được → "pencil" (bút chì nhạt, tính là filled)

      Stage 5: Multi-threshold voting
        - Kết hợp Otsu + Fixed(127) + Adaptive → Vote 2/3

    QUAN TRỌNG:
      Các edge case handlers chỉ chạy cho ô có fill DƯỚI ngưỡng bình thường.
      Ô tô >= fill_threshold luôn được chấp nhận ngay lập tức (Stage 1).

    Args:
        binary: Ảnh binary (Otsu)
        grayscale: Ảnh xám gốc
        rect: (x1, y1, x2, y2) tọa độ pixel bubble
        config: Dict cấu hình

    Returns:
        (status, fill_ratio)
        status: "filled" | "empty" | "erased" | "noise" | "pencil"
        fill_ratio: 0.0 ~ 1.0 (dùng để so sánh giữa các bubble)
    """
    fill_threshold = config.get("fill_threshold", 0.30)
    fill_ratio = compute_fill_ratio(binary, rect)

    # Stage 1: Vượt ngưỡng bình thường → FILLED (giữ nguyên hành vi cũ)
    if fill_ratio >= fill_threshold:
        return "filled", fill_ratio

    # Stage 1b: Trống rõ ràng → có thể bút chì nhạt?
    if fill_ratio < 0.05:
        if grayscale is not None:
            if detect_pencil_mark(grayscale, rect):
                return "pencil", 0.30
        return "empty", fill_ratio

    # ===== Vùng nghi ngờ: 0.05 ~ fill_threshold =====
    # Đây là nơi edge cases có ý nghĩa: tô nhạt, X, bút chì, vết xóa, noise

    # Stage 2: Lọc noise (Case 10)
    if not is_valid_mark(binary, rect):
        return "noise", 0.0

    # Stage 3: Phát hiện vết xóa (Case 8)
    if grayscale is not None:
        erased = detect_erased_mark(grayscale, rect)
        if erased == "erased":
            return "erased", 0.0

    # Stage 4: Phát hiện bút chì nhạt (Case 9)
    if grayscale is not None:
        if detect_pencil_mark(grayscale, rect):
            return "pencil", fill_ratio

    # Stage 5: Multi-threshold voting
    x1, y1, x2, y2 = rect
    if x2 > x1 and y2 > y1 and grayscale is not None:
        roi_gray = grayscale[y1:y2, x1:x2]
        if roi_gray.size > 0:
            # Threshold 1: Otsu (đã có) → fill_ratio
            vote1 = fill_ratio >= fill_threshold

            # Threshold 2: Fixed threshold = 127
            _, bin_fixed = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY_INV)
            fill_fixed = cv2.countNonZero(bin_fixed) / roi_gray.size
            vote2 = fill_fixed >= fill_threshold

            # Threshold 3: Adaptive threshold
            if roi_gray.shape[0] >= 5 and roi_gray.shape[1] >= 5:
                bsize = max(roi_gray.shape[0] | 1, 3)
                if bsize % 2 == 0:
                    bsize += 1
                bsize = min(bsize, 31)
                bin_adaptive = cv2.adaptiveThreshold(
                    roi_gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    blockSize=bsize, C=8
                )
                fill_adaptive = cv2.countNonZero(bin_adaptive) / roi_gray.size
                vote3 = fill_adaptive >= fill_threshold
            else:
                vote3 = False

            votes = sum([vote1, vote2, vote3])
            if votes >= 2:
                return "filled", fill_ratio

    # Không đủ bằng chứng → coi như trống
    return "empty", fill_ratio


def detect_grid_answers(binary, region, fill_threshold, double_threshold, margin=0.15):
    """
    Quét grid bubble, trả về dict {row_index: [col_indices đã tô]} và danh sách cảnh báo.
    row = câu hỏi, col = đáp án.
    """
    h, w = binary.shape[:2]
    rows = region["rows"]
    cols = region["cols"]

    answers = {}
    warnings = []

    for r in range(rows):
        ratios = []
        for c in range(cols):
            rect = get_bubble_rect(region, r, c, w, h, margin)
            ratio = compute_fill_ratio(binary, rect)
            ratios.append((c, ratio))

        filled = [c for c, ratio in ratios if ratio >= fill_threshold]
        double_filled = [c for c, ratio in ratios if ratio >= double_threshold]

        if len(filled) > 1:
            warnings.append(f"Câu {r+1}: Tô nhiều đáp án {filled}")
        elif len(filled) == 0 and len(double_filled) > 1:
            warnings.append(f"Câu {r+1}: Nghi ngờ tô nhiều đáp án (mờ)")

        answers[r] = filled

    return answers, warnings


def _id_column_stats(binary, grayscale, region, col, img_w, img_h, margin=0.15):
    """
    Tính thống kê cho 1 cột ID (SBD/MDT):
    - min_row theo intensity (xám)
    - intensity_gap = second_min - min
    - fill_row theo fill_ratio (binary)
    - fill_top_ratio = max_fill / second_fill
    """
    intensities = []
    fills = []

    for row in range(region["rows"]):
        rect = get_bubble_rect(region, row, col, img_w, img_h, margin)
        x1, y1, x2, y2 = rect

        roi_gray = grayscale[y1:y2, x1:x2]
        mean_val = float(np.mean(roi_gray)) if roi_gray.size > 0 else 255.0
        intensities.append(mean_val)

        roi_bin = binary[y1:y2, x1:x2]
        fill_val = float(cv2.countNonZero(roi_bin) / roi_bin.size) if roi_bin.size > 0 else 0.0
        fills.append(fill_val)

    min_row = int(np.argmin(intensities))
    sorted_i = sorted(intensities)
    intensity_gap = float(sorted_i[1] - sorted_i[0]) if len(sorted_i) > 1 else 0.0

    fill_row = int(np.argmax(fills))
    sorted_f = sorted(fills, reverse=True)
    fill_top_ratio = float(sorted_f[0] / max(sorted_f[1], 1e-6)) if len(sorted_f) > 1 else 0.0

    return min_row, intensity_gap, fill_row, fill_top_ratio


def _recover_id_digit_with_local_offset(
    binary, grayscale, region, col, img_w, img_h, margin=0.15,
    search_px=12, step_px=2, min_votes=4
):
    """
    Fallback cho cột ID đang mơ hồ:
    - Dịch nhẹ trục y của cả grid quanh vị trí hiện tại
    - Chỉ nhận offset có bằng chứng mạnh (gap >= 10 và intensity_row == fill_row)
    - Chọn digit nếu số phiếu bầu đủ lớn và vượt rõ rệt
    """
    from collections import Counter

    base_y1 = region["y1"]
    grid_height = region["y2"] - region["y1"]
    votes = Counter()

    for offset_px in range(-search_px, search_px + 1, step_px):
        dy = offset_px / img_h
        test_y1 = base_y1 + dy
        test_y2 = test_y1 + grid_height
        if test_y1 < 0 or test_y2 > 1.0:
            continue

        test_region = dict(region)
        test_region["y1"] = test_y1
        test_region["y2"] = test_y2

        min_row, gap, fill_row, _ = _id_column_stats(
            binary, grayscale, test_region, col, img_w, img_h, margin
        )
        if gap >= 10 and min_row == fill_row:
            votes[min_row] += 1

    if not votes:
        return None

    ranked = votes.most_common()
    top_row, top_votes = ranked[0]
    second_votes = ranked[1][1] if len(ranked) > 1 else 0

    if top_votes < min_votes:
        return None
    if top_votes - second_votes < 2:
        return None

    return int(top_row)


def extract_sbd(binary, config, grayscale=None):
    """Trích xuất Số Báo Danh (6 chữ số). Dùng so sánh cường độ điểm ảnh."""
    region = config["regions"]["sbd"]
    h, w = binary.shape[:2]
    margin = config.get("bubble_margin", 0.15)

    # Auto-calibrate vị trí y của grid
    region = auto_calibrate_grid_y(grayscale, region, h, w, margin)

    sbd = ""
    marks = []
    for col in range(region["cols"]):
        if grayscale is not None:
            min_row, gap, _, _ = _id_column_stats(
                binary, grayscale, region, col, w, h, margin
            )
            if gap > 10:
                sbd += region["labels"][min_row]
                marks.append(min_row)
            else:
                recovered_row = _recover_id_digit_with_local_offset(
                    binary, grayscale, region, col, w, h, margin
                )
                if recovered_row is not None:
                    sbd += region["labels"][recovered_row]
                    marks.append(recovered_row)
                else:
                    sbd += "?"
                    marks.append(None)
        else:
            ratios = []
            for row in range(region["rows"]):
                rect = get_bubble_rect(region, row, col, w, h, margin)
                ratios.append(compute_fill_ratio(binary, rect))
            max_ratio = max(ratios)
            max_row = ratios.index(max_ratio)
            sorted_r = sorted(ratios, reverse=True)
            second = sorted_r[1] if len(sorted_r) > 1 else 0
            if second > 0 and max_ratio / second >= 1.15:
                sbd += region["labels"][max_row]
                marks.append(max_row)
            else:
                sbd += "?"
                marks.append(None)
    return sbd, marks, region


def extract_mdt(binary, config, grayscale=None):
    """Trích xuất Mã Đề Thi (3 chữ số). Dùng so sánh cường độ điểm ảnh."""
    region = config["regions"]["mdt"]
    h, w = binary.shape[:2]
    margin = config.get("bubble_margin", 0.15)

    # Auto-calibrate vị trí y của grid
    region = auto_calibrate_grid_y(grayscale, region, h, w, margin)

    mdt = ""
    marks = []
    for col in range(region["cols"]):
        if grayscale is not None:
            min_row, gap, _, _ = _id_column_stats(
                binary, grayscale, region, col, w, h, margin
            )
            if gap > 10:
                mdt += region["labels"][min_row]
                marks.append(min_row)
            else:
                recovered_row = _recover_id_digit_with_local_offset(
                    binary, grayscale, region, col, w, h, margin
                )
                if recovered_row is not None:
                    mdt += region["labels"][recovered_row]
                    marks.append(recovered_row)
                else:
                    mdt += "?"
                    marks.append(None)
        else:
            ratios = []
            for row in range(region["rows"]):
                rect = get_bubble_rect(region, row, col, w, h, margin)
                ratios.append(compute_fill_ratio(binary, rect))
            max_ratio = max(ratios)
            max_row = ratios.index(max_ratio)
            sorted_r = sorted(ratios, reverse=True)
            second = sorted_r[1] if len(sorted_r) > 1 else 0
            if second > 0 and max_ratio / second >= 1.15:
                mdt += region["labels"][max_row]
                marks.append(max_row)
            else:
                mdt += "?"
                marks.append(None)
    return mdt, marks, region


def extract_fc(binary, config, grayscale=None):
    """
    Trích xuất Phần I - Nhiều lựa chọn (40 câu, 4 đáp án A/B/C/D).

    Sử dụng robust_bubble_detection để xử lý edge cases:
      - Tô rồi xóa → bỏ qua (không chấm)
      - Bút chì nhạt → vẫn nhận diện được
      - Vết bẩn/noise → lọc bỏ

    Output: {"1": [0], "2": [1,2], ...}  (0=A, 1=B, 2=C, 3=D)
    """
    fc_config = config["regions"]["fc"]
    fill_threshold = config.get("fill_threshold", 0.40)
    double_threshold = config.get("double_fill_threshold", 0.30)
    margin = config.get("bubble_margin", 0.15)
    h, w = binary.shape[:2]

    fc_result = {}
    errors = []
    warnings = []

    for group in fc_config["groups"]:
        start_q = group["start_question"]
        for r in range(group["rows"]):
            q_num = start_q + r
            statuses = []
            ratios = []
            for c in range(group["cols"]):
                rect = get_bubble_rect(group, r, c, w, h, margin)
                status, fill = robust_bubble_detection(binary, grayscale, rect, config)
                statuses.append((c, status, fill))
                ratios.append((c, fill))

            # "filled" hoặc "pencil" đều tính là có tô
            filled = [c for c, st, _ in statuses if st in ("filled", "pencil")]
            # Theo dõi vết xóa để cảnh báo
            erased = [c for c, st, _ in statuses if st == "erased"]

            if len(filled) > 1:
                warnings.append(f"FC câu {q_num}: Tô {len(filled)} đáp án")
            elif len(filled) == 0:
                # Kiểm tra có bubble nào gần ngưỡng không
                near = [c for c, ratio in ratios if ratio >= double_threshold]
                if len(near) > 1:
                    warnings.append(f"FC câu {q_num}: Nghi ngờ tô nhiều đáp án")
            if erased:
                warnings.append(f"FC câu {q_num}: Phát hiện vết xóa")

            fc_result[str(q_num)] = filled

    return fc_result, errors, warnings


def _tf_core_metrics(binary, grayscale, rect, core_ratio=0.62):
    """
    Tính đặc trưng lõi bubble để giảm nhiễu từ viền/chữ/đường kẻ:
    - core_fill: fill_ratio trong vùng ellipse trung tâm
    - mean_intensity: cường độ xám trung bình trong vùng lõi
    """
    core_fill, ring_fill, mean_intensity, _ = _bubble_core_metrics(
        binary,
        grayscale,
        rect,
        core_ratio=core_ratio
    )
    return core_fill, ring_fill, mean_intensity


def _get_warped_template_gray(config, target_w, target_h):
    """Lấy template đã warp về cùng hệ trục với ảnh aligned (có cache)."""
    template_path = config.get(
        "template_path",
        "anh/sample_image/Copy of PhieuTracNghiepTHPT2025.png"
    )
    cache_key = (template_path, int(target_w), int(target_h))
    if cache_key in _TEMPLATE_GRAY_CACHE:
        return _TEMPLATE_GRAY_CACHE[cache_key]

    try:
        from .detector import TemplateAligner
        aligner = TemplateAligner()
        aligner._init_template(template_path, target_w, target_h)
        tpl_warped = getattr(aligner, "_tpl_warped", None)
        if tpl_warped is None:
            _TEMPLATE_GRAY_CACHE[cache_key] = None
        else:
            _TEMPLATE_GRAY_CACHE[cache_key] = cv2.cvtColor(tpl_warped, cv2.COLOR_BGR2GRAY)
    except Exception:
        _TEMPLATE_GRAY_CACHE[cache_key] = None

    return _TEMPLATE_GRAY_CACHE[cache_key]


def _tf_template_delta(grayscale, template_gray, rect, core_ratio=0.62):
    """
    Mức độ tô thêm so với template trắng (đo trên lõi bubble):
    template - current > 0 nghĩa là ảnh hiện tại tối hơn template.
    """
    if grayscale is None or template_gray is None:
        return 0.0

    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi_cur = grayscale[y1:y2, x1:x2]
    roi_tpl = template_gray[y1:y2, x1:x2]
    if roi_cur.size == 0 or roi_tpl.size == 0:
        return 0.0

    roi_h, roi_w = roi_cur.shape[:2]
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    center = (roi_w // 2, roi_h // 2)
    axis_x = max(int((roi_w * core_ratio) / 2), 1)
    axis_y = max(int((roi_h * core_ratio) / 2), 1)
    cv2.ellipse(mask, center, (axis_x, axis_y), 0, 0, 360, 255, -1)

    mask_pixels = cv2.countNonZero(mask)
    if mask_pixels == 0:
        return 0.0

    diff = roi_tpl.astype(np.int16) - roi_cur.astype(np.int16)
    positive_delta = np.clip(diff, 0, None)
    delta_mean = cv2.mean(positive_delta.astype(np.float32), mask=mask)[0]
    return float(delta_mean / 255.0)


def _tf_question_key(group, group_index):
    return str(group.get("question", group_index + 1))


def extract_tf(binary, config, grayscale=None):
    """
    Trích xuất Phần II - Đúng/Sai đúng theo cấu trúc phiếu:
    - 8 câu
    - mỗi câu có 4 ý a,b,c,d
    - mỗi ý chọn Đúng(0) hoặc Sai(1)

    Output:
    {
      "1": {"a":[0], "b":[1], "c":[0], "d":[1]},
      ...
      "8": {"a":[...], ...}
    }
    """
    tf_config = config["regions"]["tf"]
    tf_det_cfg = config.get("tf_detection", {})
    margin = config.get("tf_bubble_margin", config.get("bubble_margin", 0.15))
    core_ratio = tf_det_cfg.get("core_ratio", 0.62)
    weight_fill = tf_det_cfg.get("signal_weight_fill", 0.7)
    weight_darkness = tf_det_cfg.get("signal_weight_darkness", 0.3)
    weight_template = tf_det_cfg.get("signal_weight_template", 0.6)
    weight_ring_penalty = tf_det_cfg.get("signal_weight_ring_penalty", 0.28)
    min_signal = tf_det_cfg.get("min_signal", 0.24)
    min_signal_diff = tf_det_cfg.get("min_signal_diff", 0.08)
    min_fill_ratio = tf_det_cfg.get("min_fill_ratio", 1.20)
    strong_fill = tf_det_cfg.get("strong_fill", 0.32)
    weak_fill = tf_det_cfg.get("weak_fill", 0.24)
    max_alt_fill = tf_det_cfg.get("max_alt_fill", 0.26)
    both_fill = tf_det_cfg.get("both_fill", 0.33)
    ambiguous_signal_diff = tf_det_cfg.get("ambiguous_signal_diff", 0.02)
    min_template_delta = tf_det_cfg.get("min_template_delta", 0.11)
    min_template_diff = tf_det_cfg.get("min_template_diff", 0.02)
    min_core_vs_ring = tf_det_cfg.get("min_core_vs_ring", 1.02)
    auto_calibrate_tf = tf_det_cfg.get("auto_calibrate_y", False)
    refine_center = tf_det_cfg.get("refine_center", True)
    refine_max_shift_ratio = tf_det_cfg.get("refine_max_shift_ratio", 0.22)
    refine_shrink_ratio = tf_det_cfg.get("refine_shrink_ratio", 0.92)

    h, w = binary.shape[:2]
    sub_labels = ["a", "b", "c", "d"]
    template_gray = _get_warped_template_gray(config, w, h) if grayscale is not None else None

    calibrated_groups = []
    for group in tf_config["groups"]:
        if auto_calibrate_tf:
            cal_group = auto_calibrate_grid_y(grayscale, group, h, w, margin)
        else:
            cal_group = dict(group)
        calibrated_groups.append(cal_group)

    tf_result = {}
    errors = []
    warnings = []

    for group_idx, group in enumerate(calibrated_groups, start=1):
        question_key = _tf_question_key(group, group_idx)
        question_result = {}

        for row_idx in range(min(group["rows"], len(sub_labels))):
            sub_key = sub_labels[row_idx]
            bubble_metrics = []

            for col_idx in range(group["cols"]):
                base_rect = get_bubble_rect(group, row_idx, col_idx, w, h, margin)
                rect = base_rect
                if refine_center and grayscale is not None:
                    rect = _refine_bubble_rect(
                        binary,
                        grayscale,
                        base_rect,
                        max_shift_ratio=refine_max_shift_ratio,
                        shrink_ratio=refine_shrink_ratio
                    )
                if grayscale is not None:
                    core_fill, ring_fill, mean_intensity = _tf_core_metrics(
                        binary, grayscale, rect, core_ratio=core_ratio
                    )
                    darkness = (255.0 - mean_intensity) / 255.0
                    template_delta = _tf_template_delta(
                        grayscale, template_gray, rect, core_ratio=core_ratio
                    )
                    ring_noise = max(0.0, ring_fill - (core_fill * 0.60))
                    signal = (
                        weight_fill * core_fill
                        + weight_darkness * darkness
                        + weight_template * template_delta
                        - weight_ring_penalty * ring_noise
                    )
                    bubble_metrics.append({
                        "col": col_idx,
                        "status": "core",
                        "fill": core_fill,
                        "ring_fill": ring_fill,
                        "signal": signal,
                        "template_delta": template_delta
                    })
                else:
                    status, robust_fill = robust_bubble_detection(binary, grayscale, rect, config)
                    bubble_metrics.append({
                        "col": col_idx,
                        "status": status,
                        "fill": robust_fill,
                        "ring_fill": 0.0,
                        "signal": robust_fill,
                        "template_delta": 0.0
                    })

            first = bubble_metrics[0]
            second = bubble_metrics[1]
            if first["signal"] >= second["signal"]:
                top = first
                alt = second
            else:
                top = second
                alt = first

            signal_diff = top["signal"] - alt["signal"]
            fill_ratio = top["fill"] / max(alt["fill"], 1e-6)
            if grayscale is not None:
                top_strong = top["fill"] >= strong_fill and top["signal"] >= min_signal
                top_weak = top["fill"] >= weak_fill and top["signal"] >= min_signal
                alt_low = alt["fill"] <= max_alt_fill
                top_core_dominant = top["fill"] >= (top["ring_fill"] * min_core_vs_ring)
                template_diff = top["template_delta"] - alt["template_delta"]
                template_confident = (
                    top["template_delta"] >= min_template_delta
                    and template_diff >= min_template_diff
                )

                if (
                    top["fill"] >= both_fill and alt["fill"] >= both_fill
                    and signal_diff <= ambiguous_signal_diff
                    and abs(template_diff) <= min_template_diff
                ):
                    filled = []
                    warnings.append(
                        f"TF câu {question_key} ý {sub_key}: Mơ hồ Đ/S (cả 2 cột đều mạnh)"
                    )
                elif template_confident and top_core_dominant:
                    filled = [top["col"]]
                elif (
                    top_strong and top_core_dominant
                    and (alt_low or fill_ratio >= min_fill_ratio or signal_diff >= min_signal_diff)
                ):
                    filled = [top["col"]]
                elif (
                    top_weak and top_core_dominant
                    and signal_diff >= min_signal_diff and fill_ratio >= min_fill_ratio
                ):
                    filled = [top["col"]]
                else:
                    filled = []
            else:
                top_marked = top["status"] in ("filled", "pencil")
                alt_marked = alt["status"] in ("filled", "pencil")
                if top_marked and alt_marked and signal_diff <= ambiguous_signal_diff:
                    filled = []
                    warnings.append(
                        f"TF câu {question_key} ý {sub_key}: Mơ hồ Đ/S (cả 2 cột cùng được đánh dấu)"
                    )
                elif top_marked and (not alt_marked):
                    filled = [top["col"]]
                elif top_marked and (signal_diff >= min_signal_diff or fill_ratio >= min_fill_ratio):
                    filled = [top["col"]]
                else:
                    filled = []

            question_result[sub_key] = filled

        for missing_idx in range(len(question_result), len(sub_labels)):
            question_result[sub_labels[missing_idx]] = []

        tf_result[question_key] = question_result

    return tf_result, calibrated_groups, errors, warnings


def _dg_column_gray_stats(
    binary, grayscale, region, col, img_w, img_h, margin=0.15,
    core_ratio=0.64, refine_center=True, refine_max_shift_ratio=0.22,
    refine_shrink_ratio=0.92
):
    """
    Thống kê 1 cột DG trên ảnh xám:
    - top_row: hàng tối nhất
    - top_mean: mean intensity hàng tối nhất
    - second_mean: mean intensity hàng tối thứ 2
    - median_mean: trung vị intensity toàn cột
    """
    means = []
    fills = []
    ring_fills = []
    rows = region["rows"]
    for row in range(rows):
        base_rect = get_bubble_rect(region, row, col, img_w, img_h, margin)
        rect = base_rect
        if refine_center:
            rect = _refine_bubble_rect(
                binary,
                grayscale,
                base_rect,
                max_shift_ratio=refine_max_shift_ratio,
                shrink_ratio=refine_shrink_ratio
            )
        core_fill, ring_fill, mean_val, _ = _bubble_core_metrics(
            binary,
            grayscale,
            rect,
            core_ratio=core_ratio
        )
        means.append(mean_val)
        fills.append(core_fill)
        ring_fills.append(ring_fill)

    if not means:
        return None, 255.0, 255.0, 255.0, 0.0, 0.0

    order = sorted(range(len(means)), key=lambda idx: means[idx])
    top_row = int(order[0])
    top_mean = float(means[top_row])
    second_mean = float(means[order[1]]) if len(order) > 1 else 255.0
    median_mean = float(np.median(means))
    top_fill = float(fills[top_row])
    top_ring_fill = float(ring_fills[top_row])
    return top_row, top_mean, second_mean, median_mean, top_fill, top_ring_fill


def _dg_column_binary_stats(
    binary, grayscale, region, col, img_w, img_h, margin=0.15,
    core_ratio=0.64, refine_center=False, refine_max_shift_ratio=0.22,
    refine_shrink_ratio=0.92
):
    """
    Thống kê 1 cột DG trên binary:
    - top_row: hàng fill ratio cao nhất
    - top_fill: fill ratio cao nhất
    - second_fill: fill ratio cao thứ 2
    - median_fill: trung vị fill ratio toàn cột
    """
    fills = []
    ring_fills = []
    rows = region["rows"]
    for row in range(rows):
        base_rect = get_bubble_rect(region, row, col, img_w, img_h, margin)
        rect = base_rect
        if refine_center and grayscale is not None:
            rect = _refine_bubble_rect(
                binary,
                grayscale,
                base_rect,
                max_shift_ratio=refine_max_shift_ratio,
                shrink_ratio=refine_shrink_ratio
            )
        core_fill, ring_fill, _, _ = _bubble_core_metrics(
            binary,
            grayscale,
            rect,
            core_ratio=core_ratio
        )
        fills.append(float(core_fill))
        ring_fills.append(float(ring_fill))

    if not fills:
        return None, 0.0, 0.0, 0.0, 0.0

    order = sorted(range(len(fills)), key=lambda idx: fills[idx], reverse=True)
    top_row = int(order[0])
    top_fill = float(fills[top_row])
    second_fill = float(fills[order[1]]) if len(order) > 1 else 0.0
    median_fill = float(np.median(fills))
    top_ring_fill = float(ring_fills[top_row])
    return top_row, top_fill, second_fill, median_fill, top_ring_fill


def extract_dg(binary, config, grayscale=None):
    """
    Trích xuất Phần III - Trả lời ngắn (6 câu, grid số 0-9 + dấu).
    KHÔNG dùng OCR - quét bubble grid giống FC/TF.
    Dùng so sánh tương đối: bubble tô phải đậm hơn rõ rệt so với bubble chưa tô.
    Output: {"1": "-0,0", "2": "0,2", ...}
    """
    dg_config = config["regions"]["dg"]
    dg_labels = dg_config["labels"]
    dg_rows = dg_config["rows"]
    dg_det_cfg = config.get("dg_detection", {})
    margin = config.get("dg_bubble_margin", config.get("bubble_margin", 0.15))
    fill_threshold = config.get("fill_threshold", 0.40)
    min_gap_12 = dg_det_cfg.get("min_gap_12", 3.0)
    min_gap_median = dg_det_cfg.get("min_gap_median", 5.0)
    max_mark_mean = dg_det_cfg.get("max_mark_mean", 248.0)
    row11_guard = dg_det_cfg.get("row11_guard", True)
    row11_min_gap_12 = dg_det_cfg.get("row11_min_gap_12", 22.0)
    row11_min_gap_median = dg_det_cfg.get("row11_min_gap_median", 30.0)
    min_fill_gap = dg_det_cfg.get("min_fill_gap", 0.05)
    min_fill_median_gap = dg_det_cfg.get("min_fill_median_gap", 0.08)
    max_fill_noise = dg_det_cfg.get("max_fill_noise", 0.90)
    core_ratio = dg_det_cfg.get("core_ratio", 0.64)
    refine_center = dg_det_cfg.get("refine_center", True)
    refine_max_shift_ratio = dg_det_cfg.get("refine_max_shift_ratio", 0.22)
    refine_shrink_ratio = dg_det_cfg.get("refine_shrink_ratio", 0.92)
    min_core_fill = dg_det_cfg.get("min_core_fill", 0.12)
    min_core_vs_ring = dg_det_cfg.get("min_core_vs_ring", 0.95)
    h, w = binary.shape[:2]

    dg_result = {}
    dg_marks = {}
    dg_regions = {}
    errors = []
    warnings = []

    for group in dg_config["groups"]:
        q_num = group["question"]
        cols = group["cols"]
        answer = ""
        column_marks = []

        region = {
            "x1": group["x1"], "y1": group["y1"],
            "x2": group["x2"], "y2": group["y2"],
            "rows": dg_rows, "cols": cols
        }

        for c in range(cols):
            if grayscale is not None:
                top_row, top_mean, second_mean, median_mean, top_fill, top_ring_fill = _dg_column_gray_stats(
                    binary,
                    grayscale,
                    region,
                    c,
                    w,
                    h,
                    margin,
                    core_ratio=core_ratio,
                    refine_center=refine_center,
                    refine_max_shift_ratio=refine_max_shift_ratio,
                    refine_shrink_ratio=refine_shrink_ratio
                )
                if top_row is None:
                    column_marks.append(None)
                    continue

                gap_12 = second_mean - top_mean
                gap_median = median_mean - top_mean
                core_dominant = top_fill >= (top_ring_fill * min_core_vs_ring)
                is_mark = (
                    top_mean <= max_mark_mean
                    and gap_12 >= min_gap_12
                    and gap_median >= min_gap_median
                    and top_fill >= min_core_fill
                    and core_dominant
                )

                # Guard riêng cho hàng cuối (digit 9) để giảm false positive do mép đen đáy phiếu
                if (
                    row11_guard
                    and top_row == dg_rows - 1
                    and is_mark
                    and gap_12 < row11_min_gap_12
                    and gap_median < row11_min_gap_median
                ):
                    is_mark = False

                if is_mark and top_row < len(dg_labels):
                    answer += dg_labels[top_row]
                    column_marks.append(top_row)
                else:
                    column_marks.append(None)
            else:
                top_row, top_fill, second_fill, median_fill, top_ring_fill = _dg_column_binary_stats(
                    binary,
                    grayscale,
                    region,
                    c,
                    w,
                    h,
                    margin,
                    core_ratio=core_ratio,
                    refine_center=refine_center,
                    refine_max_shift_ratio=refine_max_shift_ratio,
                    refine_shrink_ratio=refine_shrink_ratio
                )
                if top_row is None:
                    column_marks.append(None)
                    continue

                gap_fill = top_fill - second_fill
                gap_fill_median = top_fill - median_fill
                core_dominant = top_fill >= (top_ring_fill * min_core_vs_ring)
                is_mark = (
                    top_fill >= fill_threshold
                    and top_fill <= max_fill_noise
                    and gap_fill >= min_fill_gap
                    and gap_fill_median >= min_fill_median_gap
                    and core_dominant
                )

                if (
                    row11_guard
                    and top_row == dg_rows - 1
                    and is_mark
                    and gap_fill < (min_fill_gap * 1.8)
                ):
                    is_mark = False

                if is_mark and top_row < len(dg_labels):
                    answer += dg_labels[top_row]
                    column_marks.append(top_row)
                else:
                    column_marks.append(None)

        dg_result[str(q_num)] = answer
        dg_marks[str(q_num)] = column_marks
        dg_regions[str(q_num)] = dict(region)

    return dg_result, dg_marks, dg_regions, errors, warnings


def extract_tf_mark_scorer(binary, config, grayscale=None):
    """
    TF extraction using mark_scorer directly with normalized confidence
    per sub-question (row with 2 options).
    """
    tf_config = config["regions"]["tf"]
    tf_det_cfg = config.get("tf_detection", {})
    margin = config.get("tf_bubble_margin", config.get("bubble_margin", 0.15))
    core_ratio = tf_det_cfg.get("core_ratio", 0.62)
    auto_calibrate_tf = tf_det_cfg.get("auto_calibrate_y", False)
    refine_center = tf_det_cfg.get("refine_center", True)
    refine_max_shift_ratio = tf_det_cfg.get("refine_max_shift_ratio", 0.22)
    refine_shrink_ratio = tf_det_cfg.get("refine_shrink_ratio", 0.92)

    mark_min_conf = tf_det_cfg.get("mark_min_confidence", 0.56)
    mark_min_gap = tf_det_cfg.get("mark_min_confidence_gap", 0.08)
    mark_multi_delta = tf_det_cfg.get("mark_multi_delta", 0.04)
    mark_temperature = tf_det_cfg.get("mark_temperature", 1.0)
    min_core_fill = tf_det_cfg.get("mark_min_core_fill", 0.12)
    max_ring_noise = tf_det_cfg.get("mark_max_ring_noise", 0.52)
    min_core_vs_ring = tf_det_cfg.get("min_core_vs_ring", 1.02)
    template_score_boost = tf_det_cfg.get("template_score_boost", 0.0)
    template_min_for_pick = tf_det_cfg.get("template_min_for_pick", 0.0)

    h, w = binary.shape[:2]
    sub_labels = ["a", "b", "c", "d"]
    template_gray = _get_warped_template_gray(config, w, h) if grayscale is not None else None

    calibrated_groups = []
    for group in tf_config["groups"]:
        if auto_calibrate_tf:
            cal_group = auto_calibrate_grid_y(grayscale, group, h, w, margin)
        else:
            cal_group = dict(group)
        calibrated_groups.append(cal_group)

    tf_result = {}
    tf_confidence = {}
    errors = []
    warnings = []

    for group_idx, group in enumerate(calibrated_groups, start=1):
        question_key = _tf_question_key(group, group_idx)
        question_result = {}
        question_conf = {}

        for row_idx in range(min(group["rows"], len(sub_labels))):
            sub_key = sub_labels[row_idx]
            option_scores = []

            for col_idx in range(group["cols"]):
                base_rect = get_bubble_rect(group, row_idx, col_idx, w, h, margin)
                rect = base_rect
                if refine_center and grayscale is not None:
                    rect = _refine_bubble_rect(
                        binary,
                        grayscale,
                        base_rect,
                        max_shift_ratio=refine_max_shift_ratio,
                        shrink_ratio=refine_shrink_ratio
                    )

                score_payload = _score_bubble_mark(
                    binary,
                    grayscale,
                    rect,
                    core_ratio=core_ratio
                )
                template_delta = _tf_template_delta(
                    grayscale, template_gray, rect, core_ratio=core_ratio
                ) if grayscale is not None else 0.0
                score_payload["template_delta"] = float(template_delta)
                score_payload["score"] = float(
                    score_payload["score"] + (template_score_boost * template_delta)
                )
                score_payload["col"] = int(col_idx)
                option_scores.append(score_payload)

            if not option_scores:
                question_result[sub_key] = []
                question_conf[sub_key] = 0.0
                continue

            picked, confidence = choose_mark(
                option_scores,
                min_score=mark_min_conf,
                min_gap=mark_min_gap,
                normalize=True,
                temperature=mark_temperature,
                multi_delta=mark_multi_delta
            )
            question_conf[sub_key] = float(confidence)

            selected = []
            if len(picked) == 1:
                top = option_scores[picked[0]]
                core_dominant = top["core_fill"] >= (top["ring_fill"] * min_core_vs_ring)
                pass_template = (
                    top["template_delta"] >= template_min_for_pick
                    if template_min_for_pick > 0
                    else True
                )
                is_mark = (
                    top["core_fill"] >= min_core_fill
                    and top["ring_noise"] <= max_ring_noise
                    and core_dominant
                    and pass_template
                )
                if is_mark:
                    selected = [int(top["col"])]
            elif len(picked) > 1:
                warnings.append(
                    f"TF cau {question_key} y {sub_key}: mo ho D/S (confidence={confidence:.2f})"
                )

            question_result[sub_key] = selected

        for missing_idx in range(len(question_result), len(sub_labels)):
            question_result[sub_labels[missing_idx]] = []
            question_conf[sub_labels[missing_idx]] = 0.0

        tf_result[question_key] = question_result
        tf_confidence[question_key] = question_conf

    return tf_result, calibrated_groups, tf_confidence, errors, warnings


def extract_dg_mark_scorer(binary, config, grayscale=None):
    """
    DG extraction using mark_scorer directly with normalized confidence
    per digit-column.
    """
    dg_config = config["regions"]["dg"]
    dg_labels = dg_config["labels"]
    dg_rows = dg_config["rows"]
    dg_det_cfg = config.get("dg_detection", {})
    margin = config.get("dg_bubble_margin", config.get("bubble_margin", 0.15))

    fill_threshold = dg_det_cfg.get("mark_min_fill_ratio", config.get("fill_threshold", 0.40))
    max_fill_noise = dg_det_cfg.get("max_fill_noise", 0.90)
    min_core_fill = dg_det_cfg.get("min_core_fill", 0.12)
    min_core_vs_ring = dg_det_cfg.get("min_core_vs_ring", 0.95)
    max_ring_noise = dg_det_cfg.get("mark_max_ring_noise", 0.58)
    max_mark_mean = dg_det_cfg.get("max_mark_mean", 248.0)

    mark_min_conf = dg_det_cfg.get("mark_min_confidence", 0.10)
    mark_min_gap = dg_det_cfg.get("mark_min_confidence_gap", 0.008)
    mark_multi_delta = dg_det_cfg.get("mark_multi_delta", 0.006)
    mark_temperature = dg_det_cfg.get("mark_temperature", 1.0)

    row11_guard = dg_det_cfg.get("row11_guard", True)
    row11_min_conf = dg_det_cfg.get("row11_min_confidence", 0.12)
    core_ratio = dg_det_cfg.get("core_ratio", 0.64)
    refine_center = dg_det_cfg.get("refine_center", True)
    refine_max_shift_ratio = dg_det_cfg.get("refine_max_shift_ratio", 0.22)
    refine_shrink_ratio = dg_det_cfg.get("refine_shrink_ratio", 0.92)

    h, w = binary.shape[:2]
    dg_result = {}
    dg_marks = {}
    dg_regions = {}
    dg_confidence = {}
    errors = []
    warnings = []

    for group in dg_config["groups"]:
        q_num = group["question"]
        cols = group["cols"]
        answer = ""
        column_marks = []
        column_conf = []

        region = {
            "x1": group["x1"], "y1": group["y1"],
            "x2": group["x2"], "y2": group["y2"],
            "rows": dg_rows, "cols": cols
        }

        for col_idx in range(cols):
            row_scores = []
            for row_idx in range(dg_rows):
                base_rect = get_bubble_rect(region, row_idx, col_idx, w, h, margin)
                rect = base_rect
                if refine_center and grayscale is not None:
                    rect = _refine_bubble_rect(
                        binary,
                        grayscale,
                        base_rect,
                        max_shift_ratio=refine_max_shift_ratio,
                        shrink_ratio=refine_shrink_ratio
                    )

                score_payload = _score_bubble_mark(
                    binary,
                    grayscale,
                    rect,
                    core_ratio=core_ratio
                )
                score_payload["row"] = int(row_idx)
                row_scores.append(score_payload)

            if not row_scores:
                column_marks.append(None)
                column_conf.append(0.0)
                continue

            picked, confidence = choose_mark(
                row_scores,
                min_score=mark_min_conf,
                min_gap=mark_min_gap,
                normalize=True,
                temperature=mark_temperature,
                multi_delta=mark_multi_delta
            )
            column_conf.append(float(confidence))

            mark_row = None
            if len(picked) == 1:
                top = row_scores[picked[0]]
                core_dominant = top["core_fill"] >= (top["ring_fill"] * min_core_vs_ring)
                gray_guard = (top["mean_intensity"] <= max_mark_mean) if grayscale is not None else True
                is_mark = (
                    top["fill_ratio"] >= fill_threshold
                    and top["fill_ratio"] <= max_fill_noise
                    and top["core_fill"] >= min_core_fill
                    and top["ring_noise"] <= max_ring_noise
                    and core_dominant
                    and gray_guard
                )

                if (
                    row11_guard
                    and int(top["row"]) == dg_rows - 1
                    and is_mark
                    and confidence < row11_min_conf
                ):
                    is_mark = False

                if is_mark:
                    mark_row = int(top["row"])
            elif len(picked) > 1:
                warnings.append(
                    f"DG cau {q_num} cot {col_idx + 1}: mo ho nhieu hang (confidence={confidence:.2f})"
                )

            if mark_row is not None and mark_row < len(dg_labels):
                answer += dg_labels[mark_row]
                column_marks.append(mark_row)
            else:
                column_marks.append(None)

        dg_result[str(q_num)] = answer
        dg_marks[str(q_num)] = column_marks
        dg_regions[str(q_num)] = dict(region)
        dg_confidence[str(q_num)] = column_conf

    return dg_result, dg_marks, dg_regions, dg_confidence, errors, warnings


def _merge_layout_regions(config, layout_regions):
    """Merge dynamic layout regions into runtime config (fallback-safe)."""
    merged = copy.deepcopy(config)
    if not isinstance(layout_regions, dict):
        return merged

    merged_regions = merged.get("regions", {})

    for key in ("sbd", "mdt"):
        override = layout_regions.get(key)
        if isinstance(override, dict):
            base = dict(merged_regions.get(key, {}))
            base.update(override)
            merged_regions[key] = base

    for key in ("fc", "tf", "dg"):
        override = layout_regions.get(key)
        if isinstance(override, dict):
            base = dict(merged_regions.get(key, {}))
            if isinstance(base.get("groups"), list) and isinstance(override.get("groups"), list):
                base["groups"] = override["groups"]
            if "rows" in override:
                base["rows"] = override["rows"]
            if "labels" in override:
                base["labels"] = override["labels"]
            merged_regions[key] = base

    merged["regions"] = merged_regions
    return merged


def extract_all(binary, config, grayscale=None, layout_regions=None):
    """
    Trích xuất toàn bộ đáp án từ ảnh đã warp.
    Trả về (results_dict, errors_list, warnings_list).
    """
    all_errors = []
    all_warnings = []

    runtime_config = _merge_layout_regions(config, layout_regions)

    sbd, sbd_marks, sbd_region = extract_sbd(binary, runtime_config, grayscale)
    mdt, mdt_marks, mdt_region = extract_mdt(binary, runtime_config, grayscale)

    fc, fc_err, fc_warn = extract_fc(binary, runtime_config, grayscale)
    all_errors.extend(fc_err)
    all_warnings.extend(fc_warn)

    tf, tf_groups, tf_confidence, tf_err, tf_warn = extract_tf_mark_scorer(
        binary,
        runtime_config,
        grayscale
    )
    all_errors.extend(tf_err)
    all_warnings.extend(tf_warn)

    dg, dg_marks, dg_regions, dg_confidence, dg_err, dg_warn = extract_dg_mark_scorer(
        binary,
        runtime_config,
        grayscale
    )
    all_errors.extend(dg_err)
    all_warnings.extend(dg_warn)

    results = {
        "sbd": sbd,
        "mdt": mdt,
        "fc": fc,
        "tf": tf,
        "dg": dg,
        "_fc_groups": [dict(group) for group in runtime_config["regions"]["fc"]["groups"]],
        "_sbd_marks": sbd_marks,
        "_mdt_marks": mdt_marks,
        "_dg_marks": dg_marks,
        "_tf_confidence": tf_confidence,
        "_dg_confidence": dg_confidence,
        "_tf_groups": [dict(group) for group in tf_groups],
        "_dg_regions": {key: dict(val) for key, val in dg_regions.items()},
        "_sbd_region": dict(sbd_region),
        "_mdt_region": dict(mdt_region),
    }

    return results, all_errors, all_warnings
