"""
Core package - Pipeline xử lý chính cho hệ thống chấm phiếu trắc nghiệm THPT 2025.
"""
import cv2
import json
import os
import numpy as np

from .preprocessor import preprocess_to_binary, load_image
from .detector import find_corners, warp_perspective, deskew, align_to_template
from .extractor import extract_all
from .visualizer import create_debug_image, create_mark_layer
from .layout_dynamic import analyze_layout


def load_config(config_path="config.json"):
    """Đọc file cấu hình JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rotate_input_image(img, rotation_deg):
    if rotation_deg == 0:
        return img
    if rotation_deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {rotation_deg}")


def _undo_input_rotation(img, rotation_deg):
    """Undo discrete 90Â° rotation so output keeps original input orientation."""
    if rotation_deg == 0:
        return img
    if rotation_deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"Unsupported rotation: {rotation_deg}")


def _compose_homography(*matrices):
    """Compose source->dest transforms as Hn * ... * H2 * H1."""
    h_total = np.eye(3, dtype=np.float32)
    for mat in matrices:
        if mat is None:
            continue
        h_total = mat.astype(np.float32) @ h_total
    return h_total


def _project_marks_to_input(input_img, mark_layer, forward_h):
    """
    Project mark layer from aligned space back to input image coordinates.
    `forward_h` maps input -> aligned.
    """
    if mark_layer is None:
        return input_img.copy()

    h_img, w_img = input_img.shape[:2]
    try:
        inverse_h = np.linalg.inv(forward_h.astype(np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        return input_img.copy()

    projected = cv2.warpPerspective(
        mark_layer,
        inverse_h,
        (w_img, h_img),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    mask = cv2.cvtColor(projected, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(mask) == 0:
        return input_img.copy()

    blended = input_img.copy()
    blended_mix = cv2.addWeighted(input_img, 0.55, projected, 0.95, 0)
    use = mask > 0
    blended[use] = blended_mix[use]
    return blended


def _score_output(output, config):
    if output.get("res") is None:
        return -1000

    score = 0
    sbd = str(output.get("sbd", ""))
    mdt = str(output.get("mdt", ""))
    sbd_expected = config["regions"]["sbd"]["cols"]
    mdt_expected = config["regions"]["mdt"]["cols"]
    fc = output.get("res", {}).get("fc", {})
    tf = output.get("res", {}).get("tf", {})
    dg = output.get("res", {}).get("dg", {})

    sbd = sbd[:sbd_expected].ljust(sbd_expected, "?")
    mdt = mdt[:mdt_expected].ljust(mdt_expected, "?")

    score += sum(60 for ch in sbd if ch != "?")
    score -= sum(80 for ch in sbd if ch == "?")
    score += sum(60 for ch in mdt if ch != "?")
    score -= sum(80 for ch in mdt if ch == "?")
    score += sum(1 for ans in fc.values() if len(ans) == 1) * 2
    score -= sum(1 for ans in fc.values() if len(ans) > 1) * 6

    for question_data in tf.values():
        if isinstance(question_data, dict):
            score += sum(1 for ans in question_data.values() if len(ans) == 1)
            score -= sum(2 for ans in question_data.values() if len(ans) > 1)

    score += sum(len(value) for value in dg.values()) * 2
    score -= len(output.get("err", [])) * 20

    warn_text = output.get("warn", "")
    if warn_text:
        warn_items = [item.strip() for item in warn_text.split(";") if item.strip()]
        non_dynamic_warn = [
            item for item in warn_items if not item.lower().startswith("dynamic layout")
        ]
        score -= len(non_dynamic_warn) * 2

    if output.get("_layout_dynamic"):
        score += 6
        resolved_sections = output.get("_layout_resolved_sections", [])
        if isinstance(resolved_sections, list):
            score += len(resolved_sections) * 2

    if "aligned" in output.get("_detection_method", ""):
        score += 5

    return score


def _is_confident_output(output, config):
    sbd_expected = config["regions"]["sbd"]["cols"]
    mdt_expected = config["regions"]["mdt"]["cols"]
    sbd = str(output.get("sbd", ""))
    mdt = str(output.get("mdt", ""))
    return (
        len(sbd) >= sbd_expected
        and len(mdt) >= mdt_expected
        and "?" not in sbd[:sbd_expected]
        and "?" not in mdt[:mdt_expected]
    )


def _extraction_quality(results, errors, warnings, config):
    """Quality score for comparing dynamic-vs-static extraction on same warped image."""
    score = 0.0
    sbd_expected = int(config["regions"]["sbd"]["cols"])
    mdt_expected = int(config["regions"]["mdt"]["cols"])

    sbd = str(results.get("sbd", ""))[:sbd_expected].ljust(sbd_expected, "?")
    mdt = str(results.get("mdt", ""))[:mdt_expected].ljust(mdt_expected, "?")
    score += sum(8.0 for ch in sbd if ch != "?")
    score -= sum(12.0 for ch in sbd if ch == "?")
    score += sum(8.0 for ch in mdt if ch != "?")
    score -= sum(12.0 for ch in mdt if ch == "?")

    fc = results.get("fc", {})
    for i in range(1, 41):
        ans = fc.get(str(i), [])
        if len(ans) == 1:
            score += 2.5
        elif len(ans) > 1:
            score -= 7.0
        else:
            score -= 0.2

    tf = results.get("tf", {})
    for q in tf.values():
        if not isinstance(q, dict):
            continue
        for ans in q.values():
            if len(ans) == 1:
                score += 1.0
            elif len(ans) > 1:
                score -= 2.5
            else:
                score -= 0.15

    dg = results.get("dg", {})
    for val in dg.values():
        score += min(len(str(val)), 4) * 1.5

    score -= len(errors) * 18.0
    score -= len(warnings) * 0.6
    return score


def _process_loaded_image(img, image_path, config, debug=False, rotation_deg=0):
    binary = preprocess_to_binary(img, config)

    corners, method = find_corners(img, binary, config)
    warp_w = config.get("warp_width", 1800)
    warp_h = config.get("warp_height", 2500)

    warped, warp_matrix = warp_perspective(
        img, corners, warp_w, warp_h, return_matrix=True
    )

    warped, skew_angle, deskew_matrix = deskew(warped, return_matrix=True)

    warped, aligned, align_matrix = align_to_template(
        warped, config, return_matrix=True
    )
    if aligned:
        method += "+aligned"
    if rotation_deg:
        method += f"+rot{rotation_deg}"

    forward_h = _compose_homography(warp_matrix, deskew_matrix, align_matrix)

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe_e = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_e = clahe_e.apply(warped_gray)
    blurred_e = cv2.GaussianBlur(enhanced_e, (5, 5), 0)
    _, warped_binary = cv2.threshold(
        blurred_e, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    dynamic_cfg = config.get("dynamic_layout", {})
    layout_regions = None
    layout_warnings = []
    layout_diagnostics = {}
    dynamic_applied = False
    if dynamic_cfg.get("enabled", False):
        layout_regions, layout_warnings, layout_diagnostics = analyze_layout(
            warped_binary, warped_gray, config
        )
    baseline_results, baseline_errors, baseline_warnings = extract_all(
        warped_binary,
        config,
        warped_gray,
        layout_regions=None
    )
    results = baseline_results
    errors = baseline_errors
    warnings = list(baseline_warnings)

    if dynamic_cfg.get("enabled", False):
        if layout_regions:
            dynamic_results, dynamic_errors, dynamic_warnings = extract_all(
                warped_binary,
                config,
                warped_gray,
                layout_regions=layout_regions
            )
            baseline_quality = _extraction_quality(
                baseline_results, baseline_errors, baseline_warnings, config
            )
            dynamic_quality = _extraction_quality(
                dynamic_results, dynamic_errors, dynamic_warnings, config
            )
            quality_margin = float(dynamic_cfg.get("quality_margin", -2.0))
            if dynamic_quality >= (baseline_quality + quality_margin):
                results = dynamic_results
                errors = dynamic_errors
                warnings = list(dynamic_warnings)
                dynamic_applied = True
                method += "+dynlayout"
                layout_diagnostics["dynamic_quality"] = round(dynamic_quality, 2)
                layout_diagnostics["baseline_quality"] = round(baseline_quality, 2)
            else:
                method += "+dynreject"
                warnings.append(
                    f"Dynamic layout rejected (quality {dynamic_quality:.1f} < baseline {baseline_quality:.1f})."
                )
        else:
            method += "+dynfallback"

    if layout_warnings:
        warnings = list(layout_warnings) + list(warnings)

    mark_layer = create_mark_layer(warped.shape, results, config)
    marked_rotated = _project_marks_to_input(img, mark_layer, forward_h)
    marked = _undo_input_rotation(marked_rotated, rotation_deg)

    debug_img = None
    if debug:
        debug_img = create_debug_image(warped, warped_binary, config)

    if dynamic_cfg.get("enabled", False):
        if dynamic_applied:
            layout_mode = "dynamic"
        elif layout_regions:
            layout_mode = "rejected"
        else:
            layout_mode = "fallback"
    else:
        layout_mode = "disabled"

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
        "_skew_angle": round(skew_angle, 2),
        "_rotation_deg": rotation_deg,
        "_layout_dynamic": bool(dynamic_applied),
        "_layout_mode": layout_mode,
        "_layout_candidate_count": int(layout_diagnostics.get("candidate_count", 0)),
        "_layout_component_count": int(layout_diagnostics.get("component_count", 0)),
        "_layout_anchor_count": int(layout_diagnostics.get("anchor_count", 0)),
        "_layout_resolved_sections_raw": list(layout_diagnostics.get("resolved_sections", [])),
        "_layout_resolved_sections": list(layout_diagnostics.get("resolved_sections", [])) if dynamic_applied else [],
        "_tf_confidence": results.get("_tf_confidence", {}),
        "_dg_confidence": results.get("_dg_confidence", {}),
    }

    return output, marked, debug_img


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

    best_output, best_marked, best_debug = _process_loaded_image(
        img, image_path, config, debug=debug, rotation_deg=0
    )
    if _is_confident_output(best_output, config):
        return best_output, best_marked, best_debug

    best_score = _score_output(best_output, config)
    for rotation_deg in (90, 180, 270):
        rotated_img = _rotate_input_image(img, rotation_deg)
        output, marked, debug_img = _process_loaded_image(
            rotated_img, image_path, config, debug=debug, rotation_deg=rotation_deg
        )
        score = _score_output(output, config)
        if score > best_score:
            best_output, best_marked, best_debug = output, marked, debug_img
            best_score = score

    return best_output, best_marked, best_debug



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
    tf_count = 0
    for g_idx, group in enumerate(tf_groups, start=1):
        cau_num = str(group.get("question", g_idx))
        parts = []

        question_data = tf.get(cau_num, {})
        if isinstance(question_data, dict):
            for sub in sub_labels:
                ans = question_data.get(sub, [])
                if ans:
                    txt = ",".join(tf_labels[a] for a in ans if a < len(tf_labels))
                    tf_count += 1
                else:
                    txt = "_"
                parts.append(f"{sub}={txt}")
        else:
            questions = group.get("questions", [])
            for r_idx in range(len(sub_labels)):
                sub = sub_labels[r_idx]
                if questions and r_idx < len(questions):
                    ans = tf.get(str(questions[r_idx]), [])
                else:
                    ans = []
                if ans:
                    txt = ",".join(tf_labels[a] for a in ans if a < len(tf_labels))
                    tf_count += 1
                else:
                    txt = "_"
                parts.append(f"{sub}={txt}")

        lines.append(f"  Câu {cau_num}: " + " | ".join(parts))
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
    per_file_dir = os.path.join(output_dir, "per_file_results")
    os.makedirs(per_file_dir, exist_ok=True)
    all_results = []

    for i, path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Đang xử lý: {os.path.basename(path)}")

        output, marked, debug_img = process_image(path, config, debug)

        base_name = os.path.splitext(os.path.basename(path))[0]
        ext_tag = os.path.splitext(os.path.basename(path))[1].lower().lstrip(".") or "img"
        result_stem = f"{base_name}_{ext_tag}"

        # Lưu ảnh đã đánh dấu
        if marked is not None:
            out_path = os.path.join(output_dir, f"marked_{result_stem}.jpg")
            cv2.imwrite(out_path, marked)
            output["out"] = out_path

            if debug and debug_img is not None:
                debug_dir = os.path.join(output_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                dbg_path = os.path.join(debug_dir, f"debug_{result_stem}.jpg")
                cv2.imwrite(dbg_path, debug_img)

        # Lưu JSON riêng cho từng ảnh
        per_file_json_path = os.path.join(per_file_dir, f"result_{result_stem}.json")
        with open(per_file_json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        all_results.append(output)

    # Lưu JSON tổng hợp
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nHoàn tất! Kết quả lưu tại: {output_dir}")
    for r in all_results:
        print(format_results(r, config))
    return all_results
