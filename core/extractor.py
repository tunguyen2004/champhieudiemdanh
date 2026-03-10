"""
Module trích xuất đáp án từ phiếu trắc nghiệm đã warp.
Hỗ trợ 3 loại: FC (nhiều lựa chọn), TF (đúng/sai), DG (trả lời ngắn - grid số).
"""
import cv2
import numpy as np


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


def extract_fc(binary, config):
    """
    Trích xuất Phần I - Nhiều lựa chọn (40 câu, 4 đáp án A/B/C/D).
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
            ratios = []
            for c in range(group["cols"]):
                rect = get_bubble_rect(group, r, c, w, h, margin)
                ratio = compute_fill_ratio(binary, rect)
                ratios.append((c, ratio))

            filled = [c for c, ratio in ratios if ratio >= fill_threshold]

            if len(filled) > 1:
                warnings.append(f"FC câu {q_num}: Tô {len(filled)} đáp án")
            elif len(filled) == 0:
                # Kiểm tra có bubble nào gần ngưỡng không
                near = [c for c, ratio in ratios if ratio >= double_threshold]
                if len(near) > 1:
                    warnings.append(f"FC câu {q_num}: Nghi ngờ tô nhiều đáp án")

            fc_result[str(q_num)] = filled

    return fc_result, errors, warnings


def extract_tf(binary, config):
    """
    Trích xuất Phần II - Đúng/Sai (32 câu, 2 lựa chọn).
    Hỗ trợ 2 kiểu đánh số:
      - "start_question" + row index (tuần tự)
      - "questions" array (chỉ định số câu cho từng dòng)
    Output: {"1": [0], "2": [1], ...}  (0=Đúng, 1=Sai)
    """
    tf_config = config["regions"]["tf"]
    fill_threshold = config.get("fill_threshold", 0.40)
    double_threshold = config.get("double_fill_threshold", 0.30)
    margin = config.get("bubble_margin", 0.15)
    h, w = binary.shape[:2]

    tf_result = {}
    errors = []
    warnings = []

    for group in tf_config["groups"]:
        questions = group.get("questions")
        start_q = group.get("start_question", 1)

        for r in range(group["rows"]):
            if questions and r < len(questions):
                q_num = questions[r]
            else:
                q_num = start_q + r

            ratios = []
            for c in range(group["cols"]):
                rect = get_bubble_rect(group, r, c, w, h, margin)
                ratio = compute_fill_ratio(binary, rect)
                ratios.append((c, ratio))

            vals = [ratio for _, ratio in ratios]
            max_val = max(vals)
            min_val = min(vals) if min(vals) > 0 else 0.001

            if max_val >= fill_threshold:
                if len(vals) == 2 and all(v >= fill_threshold for v in vals):
                    # Cả 2 cột đều cao - kiểm tra tương đối
                    if max_val / min_val >= 1.3:
                        # Chỉ 1 cột thực sự được tô
                        filled = [c for c, ratio in ratios if ratio == max_val]
                    else:
                        # Nhiễu (cả 2 gần bằng nhau) - bỏ qua
                        filled = []
                else:
                    filled = [c for c, ratio in ratios if ratio >= fill_threshold]
            else:
                filled = []

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

    fc, fc_err, fc_warn = extract_fc(binary, config)
    all_errors.extend(fc_err)
    all_warnings.extend(fc_warn)

    tf, tf_err, tf_warn = extract_tf(binary, config)
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
