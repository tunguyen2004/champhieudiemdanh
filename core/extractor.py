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


def extract_sbd(binary, config, grayscale=None):
    """Trích xuất Số Báo Danh (6 chữ số). Dùng so sánh cường độ điểm ảnh."""
    region = config["regions"]["sbd"]
    h, w = binary.shape[:2]
    margin = config.get("bubble_margin", 0.15)

    # Auto-calibrate vị trí y của grid
    region = auto_calibrate_grid_y(grayscale, region, h, w, margin)

    sbd = ""
    for col in range(region["cols"]):
        if grayscale is not None:
            # Dùng cường độ xám (thấp = tối = đã tô)
            intensities = []
            for row in range(region["rows"]):
                rect = get_bubble_rect(region, row, col, w, h, margin)
                x1, y1, x2, y2 = rect
                roi = grayscale[y1:y2, x1:x2]
                mean_val = np.mean(roi) if roi.size > 0 else 255
                intensities.append(mean_val)
            min_val = min(intensities)
            min_row = intensities.index(min_val)
            sorted_i = sorted(intensities)
            second_min = sorted_i[1] if len(sorted_i) > 1 else 255
            if second_min > 0 and (second_min - min_val) > 10:
                sbd += region["labels"][min_row]
            else:
                sbd += "?"
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
            else:
                sbd += "?"
    return sbd


def extract_mdt(binary, config, grayscale=None):
    """Trích xuất Mã Đề Thi (3 chữ số). Dùng so sánh cường độ điểm ảnh."""
    region = config["regions"]["mdt"]
    h, w = binary.shape[:2]
    margin = config.get("bubble_margin", 0.15)

    # Auto-calibrate vị trí y của grid
    region = auto_calibrate_grid_y(grayscale, region, h, w, margin)

    mdt = ""
    for col in range(region["cols"]):
        if grayscale is not None:
            intensities = []
            for row in range(region["rows"]):
                rect = get_bubble_rect(region, row, col, w, h, margin)
                x1, y1, x2, y2 = rect
                roi = grayscale[y1:y2, x1:x2]
                mean_val = np.mean(roi) if roi.size > 0 else 255
                intensities.append(mean_val)
            min_val = min(intensities)
            min_row = intensities.index(min_val)
            sorted_i = sorted(intensities)
            second_min = sorted_i[1] if len(sorted_i) > 1 else 255
            if second_min > 0 and (second_min - min_val) > 10:
                mdt += region["labels"][min_row]
            else:
                mdt += "?"
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
            else:
                mdt += "?"
    return mdt


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


def extract_tf(binary, config, grayscale=None):
    """
    Trích xuất Phần II - Đúng/Sai (32 câu, 2 lựa chọn).

    Phương pháp chính: Intensity-based comparison (so sánh cường độ xám
    tương đối giữa 2 cột Đúng/Sai). Cột nào tối hơn rõ rệt → cột đó
    được tô. Chính xác hơn fill_ratio cho grid nhỏ, sát nhau.

    Fallback: robust_bubble_detection khi không có grayscale.

    Output: {"1": [0], "2": [1], ...}  (0=Đúng, 1=Sai)
    """
    tf_config = config["regions"]["tf"]
    fill_threshold = config.get("fill_threshold", 0.40)
    margin = config.get("tf_bubble_margin", config.get("bubble_margin", 0.15))
    h, w = binary.shape[:2]

    # Auto-calibrate y cho từng TF group
    calibrated_groups = []
    for group in tf_config["groups"]:
        cal_group = auto_calibrate_grid_y(grayscale, group, h, w, margin)
        calibrated_groups.append(cal_group)

    tf_result = {}
    errors = []
    warnings = []

    for group in calibrated_groups:
        questions = group.get("questions")
        start_q = group.get("start_question", 1)

        for r in range(group["rows"]):
            if questions and r < len(questions):
                q_num = questions[r]
            else:
                q_num = start_q + r

            if grayscale is not None:
                # === PHƯƠNG PHÁP CHÍNH: Intensity-based comparison ===
                # So sánh cường độ xám giữa 2 cột (Đúng/Sai)
                # Cột được tô sẽ có mean intensity THẤP hơn (tối hơn)
                intensities = []
                for c in range(group["cols"]):
                    rect = get_bubble_rect(group, r, c, w, h, margin)
                    x1, y1, x2, y2 = rect
                    roi = grayscale[y1:y2, x1:x2]
                    mean_val = np.mean(roi) if roi.size > 0 else 255
                    intensities.append((c, mean_val))

                # Sắp xếp: cột tối nhất trước
                sorted_by_intensity = sorted(intensities, key=lambda x: x[1])
                darkest_col, darkest_val = sorted_by_intensity[0]
                lightest_col, lightest_val = sorted_by_intensity[-1]

                # Chênh lệch intensity giữa 2 cột
                intensity_diff = lightest_val - darkest_val

                # Kiểm tra cột tối nhất có thực sự được tô không
                # (không phải chỉ do viền/nhiễu)
                darkest_rect = get_bubble_rect(group, r, darkest_col, w, h, margin)
                darkest_fill = compute_fill_ratio(binary, darkest_rect)

                if intensity_diff > 10 and darkest_val < 210:
                    # Chênh lệch rõ ràng → cột tối hơn được tô
                    filled = [darkest_col]
                elif intensity_diff > 5 and darkest_fill >= fill_threshold:
                    # Chênh lệch vừa phải nhưng fill_ratio cũng xác nhận
                    filled = [darkest_col]
                elif darkest_fill >= fill_threshold and darkest_val < 195:
                    # Fill ratio cao + cột đủ tối → chấp nhận
                    # Kiểm tra cột còn lại không bị tô
                    lightest_rect = get_bubble_rect(group, r, lightest_col, w, h, margin)
                    lightest_fill = compute_fill_ratio(binary, lightest_rect)
                    if darkest_fill / max(lightest_fill, 0.001) >= 1.2:
                        filled = [darkest_col]
                    elif lightest_fill >= fill_threshold:
                        # Cả 2 đều cao → nhiễu, bỏ qua
                        filled = []
                        warnings.append(f"TF câu {q_num}: Nhiễu biên (cả 2 cột đều cao)")
                    else:
                        filled = [darkest_col]
                else:
                    # Không đủ chênh lệch → coi như trống
                    filled = []
            else:
                # === FALLBACK: robust_bubble_detection (không có grayscale) ===
                statuses = []
                for c in range(group["cols"]):
                    rect = get_bubble_rect(group, r, c, w, h, margin)
                    status, fill = robust_bubble_detection(binary, grayscale, rect, config)
                    statuses.append((c, status, fill))

                vals = [fill for _, _, fill in statuses]
                max_val = max(vals) if vals else 0
                min_val = min(vals) if vals else 0
                min_val = min_val if min_val > 0 else 0.001

                robust_filled = [c for c, st, _ in statuses if st in ("filled", "pencil")]

                if len(robust_filled) == 2 and len(vals) == 2:
                    if max_val / min_val >= 1.3:
                        filled = [c for c, _, fill in statuses if fill == max_val]
                    else:
                        filled = []
                else:
                    filled = robust_filled

            if len(filled) > 1:
                warnings.append(f"TF câu {q_num}: Tô cả Đúng và Sai")

            tf_result[str(q_num)] = filled

    return tf_result, errors, warnings


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
    fill_threshold = config.get("fill_threshold", 0.40)
    margin = config.get("bubble_margin", 0.15)
    h, w = binary.shape[:2]

    dg_result = {}
    errors = []
    warnings = []

    for group in dg_config["groups"]:
        q_num = group["question"]
        cols = group["cols"]
        answer = ""

        region = {
            "x1": group["x1"], "y1": group["y1"],
            "x2": group["x2"], "y2": group["y2"],
            "rows": dg_rows, "cols": cols
        }

        for c in range(cols):
            if grayscale is not None:
                # Dùng cường độ xám (thấp = tối = đã tô)
                intensities = []
                for r in range(dg_rows):
                    rect = get_bubble_rect(region, r, c, w, h, margin)
                    x1, y1, x2, y2 = rect
                    roi = grayscale[y1:y2, x1:x2]
                    mean_val = np.mean(roi) if roi.size > 0 else 255
                    intensities.append(mean_val)
                # Lọc nhiễu viền: bỏ qua bubble quá tối (< 50) - viền form/đường kẻ đậm
                valid = [(r, v) for r, v in enumerate(intensities) if v >= 50]
                if len(valid) < 2:
                    continue
                valid_vals = [v for _, v in valid]
                min_val = min(valid_vals)
                min_row = [r for r, v in valid if v == min_val][0]
                sorted_v = sorted(valid_vals)
                second_min = sorted_v[1] if len(sorted_v) > 1 else 255
                median_val = sorted_v[len(sorted_v)//2]
                # Điều kiện phát hiện bubble được tô:
                # 1. Median > 150 (phần lớn bubble trắng)
                # 2. Chênh lệch min vs second > 15
                # 3. Min > 60% median (loại noise cấu trúc form - quá tối so với nền)
                if (median_val > 150 and (second_min - min_val) > 15
                        and min_val > median_val * 0.60
                        and min_row < len(dg_labels)):
                    answer += dg_labels[min_row]
            else:
                # Fallback: so sánh tương đối trên binary
                ratios = []
                for r in range(dg_rows):
                    rect = get_bubble_rect(region, r, c, w, h, margin)
                    ratio = compute_fill_ratio(binary, rect)
                    ratios.append(ratio)
                # Lọc nhiễu viền: bỏ qua bubble fill > 0.9 (viền form)
                valid = [(r, v) for r, v in enumerate(ratios) if v < 0.9]
                if len(valid) < 2:
                    continue
                valid_vals = [v for _, v in valid]
                max_ratio = max(valid_vals)
                max_row = [r for r, v in valid if v == max_ratio][0]
                sorted_r = sorted(valid_vals, reverse=True)
                second = sorted_r[1] if len(sorted_r) > 1 else 0
                if max_ratio >= fill_threshold and second > 0 and max_ratio / second >= 1.5:
                    if max_row < len(dg_labels):
                        answer += dg_labels[max_row]

        dg_result[str(q_num)] = answer

    return dg_result, errors, warnings


def extract_all(binary, config, grayscale=None):
    """
    Trích xuất toàn bộ đáp án từ ảnh đã warp.
    Trả về (results_dict, errors_list, warnings_list).
    """
    all_errors = []
    all_warnings = []

    sbd = extract_sbd(binary, config, grayscale)
    mdt = extract_mdt(binary, config, grayscale)

    fc, fc_err, fc_warn = extract_fc(binary, config, grayscale)
    all_errors.extend(fc_err)
    all_warnings.extend(fc_warn)

    tf, tf_err, tf_warn = extract_tf(binary, config, grayscale)
    all_errors.extend(tf_err)
    all_warnings.extend(tf_warn)

    dg, dg_err, dg_warn = extract_dg(binary, config, grayscale)
    all_errors.extend(dg_err)
    all_warnings.extend(dg_warn)

    results = {
        "sbd": sbd,
        "mdt": mdt,
        "fc": fc,
        "tf": tf,
        "dg": dg
    }

    return results, all_errors, all_warnings
